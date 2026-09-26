"""Reference-based image-editing backends (the primary, commercial path).

Two implementations share one code path because both take *(person image,
garment image, instruction)* and return an edited person:

:class:`QwenEditBackend`
    ``Qwen/Qwen-Image-Edit-2511``, Apache-2.0. Highest fidelity, native
    multi-image reference conditioning, supports LoRA (used in Phase 3 for
    saree-specific fine-tuning).

:class:`FluxKleinBackend`
    ``black-forest-labs/FLUX.2-klein-4B``, Apache-2.0. Roughly half the VRAM
    and noticeably faster; useful as the low-cost tier and for A/B comparison.

Why editing rather than warping
-------------------------------
A saree is a single 5-9 metre rectangle of cloth. It has no sleeves, no
shoulder seams, and no fixed 2-D pattern that could be warped onto a body
region - the garment's final shape exists only as a function of how it is
wrapped. Garment-warping models learn a flat-garment-to-body correspondence,
so they cannot represent that. Reference editing can, because the drape is
described in language and synthesised rather than transformed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ai_trial_room.backends.base import TryOnBackend, TryOnRequest, TryOnResult
from ai_trial_room.backends.prompts import (
    DRAPE_SPECS,
    build_negative_prompt,
    build_prompt,
    build_refine_prompt,
)
from ai_trial_room.config import CONFIG, BackendId, Category
from ai_trial_room.postprocessing.blend import feather_composite
from ai_trial_room.preprocessing.regions import (
    GarmentRegion,
    derive_regions,
    restrict_to_mask,
)
from ai_trial_room.utils.device import detect_hardware, resolve_quantization
from ai_trial_room.utils.errors import ModelLoadError
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)


class _EditBackendBase(TryOnBackend):
    """Shared logic for reference-editing backends."""

    family = "edit"

    #: Name of the diffusers pipeline class to import from ``diffusers``.
    pipeline_class_name: str

    @property
    def supported_categories(self) -> frozenset[Category]:
        """Editing backends handle every category, draped included."""
        return frozenset(Category)

    # -- loading ----------------------------------------------------------- #

    def _import_pipeline_class(self) -> Any:
        """Import this backend's diffusers pipeline class.

        Raises
        ------
        ModelLoadError
            When diffusers is missing or too old to expose the class.
        """
        try:
            import diffusers
        except ImportError as exc:
            raise ModelLoadError(
                "The diffusers library is not installed. Run "
                "`pip install -r requirements.txt`.",
                detail=str(exc),
            ) from exc

        cls = getattr(diffusers, self.pipeline_class_name, None)
        if cls is None:
            raise ModelLoadError(
                f"Your diffusers version ({diffusers.__version__}) does not "
                f"provide {self.pipeline_class_name}. Upgrade with "
                "`pip install -U diffusers`.",
                detail=f"missing {self.pipeline_class_name}",
            )
        return cls

    def _load(self) -> Any:
        """Build the pipeline, applying quantization and memory savers."""
        pipeline_cls = self._import_pipeline_class()
        dtype = self._torch_dtype()
        quantization = resolve_quantization(CONFIG.runtime.quantize)

        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if CONFIG.hf_token:
            kwargs["token"] = CONFIG.hf_token

        if quantization == "int4":
            transformer = self._load_int4_transformer()
            if transformer is not None:
                kwargs["transformer"] = transformer

        try:
            pipeline = pipeline_cls.from_pretrained(self.spec.repo_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - hub/network/format failures vary
            raise ModelLoadError(
                f"Could not load {self.spec.repo_id}.",
                detail=f"{type(exc).__name__}: {exc}"[:400],
            ) from exc

        if quantization == "fp8":
            self._cast_transformer_fp8(pipeline)

        return self._apply_memory_savers(pipeline)

    def _load_int4_transformer(self) -> Any | None:
        """Try to load a Nunchaku SVDQuant INT4/FP4 transformer.

        Nunchaku shrinks the 12B transformer from ~40 GB (bf16) to ~7 GB and,
        with its own async CPU offload, runs in about 3-4 GB of VRAM. It is an
        optional dependency: when absent we simply fall back to bf16 plus
        diffusers offloading.

        Returns
        -------
        Any or None
            The quantized transformer, or ``None`` if unavailable.
        """
        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel
            from nunchaku.utils import get_precision
        except ImportError:
            logger.info(
                "nunchaku not installed; using %s with CPU offload instead of INT4. "
                "Install nunchaku for a large VRAM saving.",
                CONFIG.runtime.dtype,
            )
            return None

        try:
            precision = get_precision()  # "int4" on Ampere/Ada, "fp4" on Blackwell
            repo = "nunchaku-tech/nunchaku-qwen-image-edit-2511"
            filename = f"svdq-{precision}_r128-qwen-image-edit-2511.safetensors"
            logger.info("Loading Nunchaku %s transformer: %s/%s", precision, repo, filename)

            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
                f"{repo}/{filename}"
            )
            # Nunchaku manages its own per-layer offload; 1 GB pinned budget
            # keeps peak VRAM near 4 GB on an 8 GB laptop card.
            if hasattr(transformer, "set_offload"):
                transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
            return transformer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Nunchaku INT4 load failed (%s); falling back to bf16.", exc)
            return None

    def _cast_transformer_fp8(self, pipeline: Any) -> None:
        """Cast transformer weights to FP8 in place, if the GPU supports it.

        FP8 halves transformer memory versus bf16 at a small quality cost, and
        needs no extra packages. Requires Hopper/Ada/Blackwell
        (compute capability >= 8.9); silently skipped elsewhere.
        """
        import torch

        hardware = detect_hardware()
        capability = hardware.compute_capability or (0, 0)
        if capability < (8, 9):
            logger.info(
                "FP8 needs compute capability >= 8.9 (have %s.%s); keeping %s.",
                *capability,
                CONFIG.runtime.dtype,
            )
            return

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            return
        try:
            transformer.to(torch.float8_e4m3fn)
            logger.info("Cast transformer to float8_e4m3fn.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FP8 cast failed (%s); keeping original dtype.", exc)

    # -- generation -------------------------------------------------------- #

    def _build_prompt(self, request: TryOnRequest) -> str:
        """Compose the editing instruction for this request."""
        return build_prompt(
            request.category,
            drape_style=request.options.drape_style,
            dupatta_style=request.options.dupatta_style,
            dominant_shoulder=request.person.pose.dominant_shoulder,
            framing=request.person.framing,
            looks_unstitched=request.garment.looks_unstitched,
            extra=request.options.extra_prompt,
        )

    def _build_negative(self, request: TryOnRequest) -> str:
        """Compose the layered negative prompt for this request."""
        return build_negative_prompt(
            request.category,
            drape_style=request.options.drape_style,
        )

    def _pipeline_kwargs(self, request: TryOnRequest, prompt: str, seed: int) -> dict[str, Any]:
        """Assemble keyword arguments for the pipeline call. Override per model."""
        import torch

        options = request.options
        steps = options.steps or self.spec.default_steps
        true_cfg = options.true_cfg_scale or self.spec.default_true_cfg
        width, height = CONFIG.runtime.size

        return {
            "image": [request.person.image, request.garment.image],
            "prompt": prompt,
            "negative_prompt": self._build_negative(request),
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg,
            "guidance_scale": options.guidance_scale or 1.0,
            "width": width,
            "height": height,
            "num_images_per_prompt": 1,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }

    def _generate(self, request: TryOnRequest) -> TryOnResult:
        """Run the editing pipeline and return the try-on image."""
        prompt = self._build_prompt(request)
        seed = request.options.resolved_seed()
        kwargs = self._pipeline_kwargs(request, prompt, seed)
        steps = int(kwargs.get("num_inference_steps", self.spec.default_steps))

        logger.debug("Prompt (%d chars): %s", len(prompt), prompt)
        request.progress(0.15, f"Generating {request.category.label.lower()}...")

        callback = _make_step_callback(request, steps)
        if callback is not None:
            kwargs["callback_on_step_end"] = callback

        assert self._pipeline is not None  # guaranteed by TryOnBackend.generate
        output = self._pipeline(**kwargs)
        image: Image.Image = output.images[0]

        refined_regions: list[str] = []
        if request.options.refine:
            image, refined_regions = self._refine_regions(request, image, seed)

        return TryOnResult(
            image=image,
            backend_id=self.backend_id,
            prompt=prompt,
            seed=seed,
            steps=steps,
            duration_s=0.0,  # filled in by TryOnBackend.generate
            refined_regions=refined_regions,
            metadata={
                "family": self.family,
                "repo_id": self.spec.repo_id,
                "quantization": resolve_quantization(CONFIG.runtime.quantize),
                "drape_style": request.options.drape_style.value,
                "dupatta_style": request.options.dupatta_style.value,
                "framing": request.person.framing,
                "dominant_shoulder": request.person.pose.dominant_shoulder,
                "background_removed": request.garment.background_removed,
            },
        )

    # -- refinement -------------------------------------------------------- #

    #: Regions worth a second pass, in priority order. The pallu carries most of
    #: a saree's design value, so it is refined first and alone by default.
    _REFINE_PRIORITY: tuple[GarmentRegion, ...] = (GarmentRegion.PALLU,)

    def _refine_regions(
        self,
        request: TryOnRequest,
        image: Image.Image,
        seed: int,
    ) -> tuple[Image.Image, list[str]]:
        """Run a short, focused second pass over high-value garment regions.

        How it works, and why it works this way
        ---------------------------------------
        ``QwenImageEditPlusPipeline`` has no mask argument - it is an instruction
        editor, not an inpainter. So we cannot ask it to touch only the pallu.

        Instead: re-run the pipeline using the *first-pass output* as reference 1
        (so the drape and pose are already established and only detail is at
        stake), with a prompt that talks about nothing but the fabric detail of
        one region, at fewer steps and lower guidance. Then composite only that
        region back through its feathered mask. Everything outside the region is
        bit-identical to the first pass.

        That makes the pass safe: a bad refinement can only degrade the region it
        was aimed at, never the face, background or silhouette.

        Parameters
        ----------
        request:
            The originating request, for pose, category and options.
        image:
            First-pass output.
        seed:
            Base seed. The refinement uses ``seed + 1`` so it explores a
            different sample rather than reproducing the same detail.

        Returns
        -------
        tuple
            ``(image, refined_region_labels)``. On any failure the input image is
            returned unchanged with an empty label list.
        """
        options = request.options
        regions = derive_regions(
            request.person.pose,
            request.person.parse,
            request.category,
            drape_style=options.drape_style,
            pallu_shoulder=DRAPE_SPECS[options.drape_style].shoulder
            if request.category is Category.SAREE
            else request.person.pose.dominant_shoulder,
        )

        refined: list[str] = []
        current = image

        for region in self._REFINE_PRIORITY:
            mask = regions.get(region)
            if mask is None or not regions.is_usable(region):
                logger.info(
                    "Skipping %s refinement (coverage %.2f%% too small).",
                    region.value,
                    regions.coverage(region) * 100,
                )
                continue

            # Keep the geometric region inside what we were allowed to repaint,
            # so refinement cannot leak onto the face or the background.
            mask = restrict_to_mask(mask, request.person.inpaint_mask)

            try:
                current = self._refine_one(request, current, mask, region, seed)
                refined.append(region.label)
            except Exception as exc:  # noqa: BLE001 - refinement is optional polish
                logger.warning(
                    "Refinement of %s failed (%s); keeping the first pass.",
                    region.value,
                    exc,
                )
                break

        return current, refined

    def _refine_one(
        self,
        request: TryOnRequest,
        image: Image.Image,
        mask: Image.Image,
        region: GarmentRegion,
        seed: int,
    ) -> Image.Image:
        """Refine a single region and composite it back.

        Parameters
        ----------
        request:
            Originating request.
        image:
            Current image to improve.
        mask:
            Feathered ``L`` mask of the region.
        region:
            Which region is being refined, for the prompt and logs.
        seed:
            Base seed; ``seed + 1`` is used for this pass.

        Returns
        -------
        Image.Image
            The image with ``region`` replaced by the refined version.
        """
        import torch

        options = request.options
        steps = options.refine_steps or max(12, (options.steps or self.spec.default_steps) // 2)
        prompt = build_refine_prompt(request.category, region.label, drape_style=options.drape_style)

        request.progress(0.80, f"Refining {region.label}...")
        logger.info("Refining %s: %d steps, strength %.2f", region.value, steps, options.refine_strength)

        width, height = CONFIG.runtime.size
        kwargs: dict[str, Any] = {
            # Reference 1 is the first-pass result, not the original photo: the
            # drape is already correct and only detail is being improved.
            "image": [image, request.garment.image],
            "prompt": prompt,
            "negative_prompt": self._build_negative(request),
            "num_inference_steps": steps,
            "true_cfg_scale": max(1.5, (options.true_cfg_scale or self.spec.default_true_cfg) - 1.0),
            "guidance_scale": 1.0,
            "width": width,
            "height": height,
            "num_images_per_prompt": 1,
            "generator": torch.Generator(device="cpu").manual_seed(seed + 1),
        }
        if self.backend_id is BackendId.FLUX_KLEIN:
            kwargs.pop("negative_prompt", None)
            kwargs.pop("true_cfg_scale", None)

        assert self._pipeline is not None
        refined_full: Image.Image = self._pipeline(**kwargs).images[0]

        # Scale the mask by refine_strength so the pass blends in rather than
        # fully replacing the region - this is the knob that keeps it subtle.
        strength = min(max(options.refine_strength, 0.0), 1.0)
        scaled = Image.fromarray(
            (np.asarray(mask.convert("L"), dtype=np.float32) * strength).astype(np.uint8),
            mode="L",
        )
        return feather_composite(refined_full, image, scaled, sigma=4.0)


def _make_step_callback(request: TryOnRequest, total_steps: int) -> Any | None:
    """Build a diffusers ``callback_on_step_end`` that reports UI progress.

    Maps denoising steps onto the 0.15-0.90 slice of the progress bar, leaving
    room for loading before and postprocessing after.

    Returns
    -------
    Callable or None
        ``None`` when no progress sink was supplied.
    """
    if request.progress is None:
        return None

    def _callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        fraction = 0.15 + 0.75 * ((step + 1) / max(1, total_steps))
        request.progress(min(fraction, 0.90), f"Step {step + 1}/{total_steps}")
        return callback_kwargs

    return _callback


class QwenEditBackend(_EditBackendBase):
    """``Qwen/Qwen-Image-Edit-2511`` - Apache-2.0, primary backend.

    Notes
    -----
    Uses ``true_cfg_scale`` (real classifier-free guidance against the negative
    prompt) while holding ``guidance_scale`` at 1.0, which is how the Qwen
    pipeline expects to be driven. Recommended defaults from the model card are
    40 steps at ``true_cfg_scale=4.0``; the UI defaults to 30 steps to keep a
    T4 demo responsive.
    """

    backend_id = BackendId.QWEN_EDIT
    pipeline_class_name = "QwenImageEditPlusPipeline"


class FluxKleinBackend(_EditBackendBase):
    """``black-forest-labs/FLUX.2-klein-4B`` - Apache-2.0, fast tier.

    Notes
    -----
    The Flux 2 pipeline takes ``guidance_scale`` rather than ``true_cfg_scale``
    and does not accept a negative prompt, so :meth:`_pipeline_kwargs` is
    overridden to drop those arguments.
    """

    backend_id = BackendId.FLUX_KLEIN
    pipeline_class_name = "Flux2KleinPipeline"

    def _pipeline_kwargs(self, request: TryOnRequest, prompt: str, seed: int) -> dict[str, Any]:
        """Adapt the shared kwargs to the Flux 2 signature."""
        kwargs = super()._pipeline_kwargs(request, prompt, seed)
        kwargs.pop("negative_prompt", None)
        kwargs.pop("true_cfg_scale", None)
        kwargs["guidance_scale"] = (
            request.options.guidance_scale or self.spec.default_true_cfg
        )
        return kwargs
