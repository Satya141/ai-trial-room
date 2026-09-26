"""Garment-warping try-on backend (CatVTON) - RESEARCH USE ONLY.

.. warning::

   CatVTON's code, checkpoints and demo are released under **CC BY-NC-SA
   4.0**. You may not use this backend, or images it produces, in a product you
   sell. It is disabled unless ``ALLOW_NONCOMMERCIAL=1`` is exported, and
   :meth:`~ai_trial_room.backends.base.TryOnBackend.check_license` enforces
   that before any weights are fetched.

   The same restriction applies to IDM-VTON and OOTDiffusion. As of this
   writing there is no permissively licensed garment-warping try-on model, which
   is why :mod:`ai_trial_room.backends.edit_backend` is the production path.

Why keep it at all?
-------------------
Two reasons, both useful for a portfolio and for product decisions:

1. **Texture-fidelity baseline.** Warping copies real garment pixels rather
   than synthesising them, so on a plain stitched kurti it reproduces a printed
   motif more exactly than any editing model. Having the comparison on hand
   tells you how much fidelity the commercial path costs you.
2. **Speed baseline.** CatVTON is 899 M parameters against Qwen's 12 B and runs
   in under 8 GB, so it shows what a purpose-built small model can do.

The comparison grid produced by ``scripts/compare_backends.py`` is built for
exactly this.

Installation
------------
CatVTON is not on PyPI and is not a diffusers pipeline. To enable it::

    git clone https://github.com/Zheng-Chong/CatVTON third_party/CatVTON
    export ALLOW_NONCOMMERCIAL=1
    export AITR_CATVTON_PATH=third_party/CatVTON

Without those steps this backend raises
:class:`~ai_trial_room.utils.errors.BackendUnavailableError`, which the router
catches and reports cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from ai_trial_room.backends.base import TryOnBackend, TryOnRequest, TryOnResult
from ai_trial_room.config import CONFIG, BackendId, Category
from ai_trial_room.utils.errors import BackendUnavailableError, ModelLoadError
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Where the cloned CatVTON repository lives.
CATVTON_PATH_ENV = "AITR_CATVTON_PATH"


class CatVtonBackend(TryOnBackend):
    """``zhengchong/CatVTON`` - CC BY-NC-SA 4.0, research only.

    Notes
    -----
    Only non-draped, stitched garments are supported. Requesting a saree or
    lehenga raises
    :class:`~ai_trial_room.utils.errors.BackendError` from the base class,
    because a warping model has no mechanism to produce a drape.
    """

    backend_id = BackendId.CATVTON
    family = "vton"

    #: CatVTON's own naming for the garment region it warps into.
    _CLOTH_TYPE_BY_CATEGORY = {
        Category.KURTI: "upper",
        Category.KURTA: "upper",
        Category.DRESS: "overall",
    }

    @property
    def supported_categories(self) -> frozenset[Category]:
        """Stitched garments only - draped garments are structurally impossible."""
        return frozenset({Category.DRESS, Category.KURTI, Category.KURTA})

    def _repo_path(self) -> Path:
        """Locate the cloned CatVTON repository.

        Raises
        ------
        BackendUnavailableError
            If the path is unset or does not contain the expected module.
        """
        raw = os.environ.get(CATVTON_PATH_ENV, "third_party/CatVTON")
        path = (Path(raw) if Path(raw).is_absolute() else Path.cwd() / raw).resolve()

        if not (path / "model" / "pipeline.py").exists():
            raise BackendUnavailableError(
                "CatVTON is not installed. It is a research-licensed comparison "
                "backend and is optional - see the README section 'Optional: "
                "CatVTON baseline'.",
                detail=f"expected {path / 'model' / 'pipeline.py'}",
            )
        return path

    def _load(self) -> Any:
        """Import CatVTON from its cloned repo and build its pipeline."""
        repo_path = self._repo_path()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from model.pipeline import CatVTONPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailableError(
                "CatVTON's dependencies are missing. Install them from its own "
                "requirements file inside the cloned repository.",
                detail=str(exc),
            ) from exc

        from huggingface_hub import snapshot_download

        dtype = self._torch_dtype()
        try:
            weights = snapshot_download(
                repo_id=self.spec.repo_id,
                token=CONFIG.hf_token,
            )
            pipeline = CatVTONPipeline(
                base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                attn_ckpt=weights,
                attn_ckpt_version="mix",
                weight_dtype=dtype,
                use_tf32=True,
                device="cuda" if self._torch_available_cuda() else "cpu",
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Could not initialise CatVTON.",
                detail=f"{type(exc).__name__}: {exc}"[:400],
            ) from exc

        # CatVTON manages its own device placement, so we only apply the
        # slicing optimisations that are safe to layer on top.
        return pipeline

    @staticmethod
    def _torch_available_cuda() -> bool:
        """True when a CUDA device is usable."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:  # pragma: no cover
            return False

    def _generate(self, request: TryOnRequest) -> TryOnResult:
        """Warp the garment onto the masked body region."""
        import torch

        options = request.options
        steps = options.steps or self.spec.default_steps
        guidance = options.guidance_scale or self.spec.default_true_cfg
        seed = options.resolved_seed()
        width, height = CONFIG.runtime.size

        request.progress(0.2, "Warping garment onto body...")

        assert self._pipeline is not None
        images = self._pipeline(
            image=request.person.image,
            condition_image=request.garment.image,
            mask=request.person.inpaint_mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        result_image: Image.Image = images[0] if isinstance(images, (list, tuple)) else images

        return TryOnResult(
            image=result_image,
            backend_id=self.backend_id,
            prompt="(warping model - no text prompt)",
            seed=seed,
            steps=steps,
            duration_s=0.0,
            metadata={
                "family": self.family,
                "repo_id": self.spec.repo_id,
                "cloth_type": self._CLOTH_TYPE_BY_CATEGORY.get(request.category, "upper"),
                "license_warning": "CC BY-NC-SA 4.0 - non-commercial use only",
            },
        )
