"""Abstract backend interface and shared request/result types.

A *backend* is any model that can put a garment onto a person. Two families
exist, with very different mechanics:

``vton``
    Garment-warping models (CatVTON, IDM-VTON, OOTDiffusion). They warp a
    flat garment onto a parsed body region. Excellent texture fidelity on
    stitched upper-body garments; structurally incapable of draping a saree.
    Every such model released to date is non-commercially licensed.

``edit``
    Reference-based image editors (Qwen-Image-Edit, FLUX.2 klein). They take
    the person and garment as reference images plus an instruction. Handles
    draped garments, and the leading options are Apache-2.0.

Both implement :class:`TryOnBackend`, so :mod:`ai_trial_room.router` can swap
them without the UI knowing which is in use.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final

from PIL import Image

from ai_trial_room.config import (
    CONFIG,
    MODEL_SPECS,
    BackendId,
    Category,
    DrapeStyle,
    LicenseClass,
    ModelSpec,
)
from ai_trial_room.preprocessing.garment import GarmentAssets
from ai_trial_room.preprocessing.person import PersonAssets
from ai_trial_room.utils.device import (
    detect_hardware,
    free_vram,
    is_oom_error,
    resolve_dtype,
    resolve_quantization,
    vram_report,
)
from ai_trial_room.utils.errors import (
    BackendError,
    LicenseRestrictedError,
    OutOfMemoryError,
)
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Signature of the progress callback the UI passes in: ``(fraction, message)``.
ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    """Default progress sink used when the caller supplies none."""


# --------------------------------------------------------------------------- #
# Request / result
# --------------------------------------------------------------------------- #


@dataclass
class TryOnOptions:
    """User-adjustable generation settings.

    Values of ``None`` mean "use the backend's default", which lets the UI
    expose an Advanced panel without hard-coding every model's preferences.
    """

    steps: int | None = None
    guidance_scale: float | None = None
    true_cfg_scale: float | None = None
    #: ``-1`` (or ``None``) requests a fresh random seed each run.
    seed: int | None = None
    drape_style: DrapeStyle = DrapeStyle.NIVI
    #: Extra instruction appended to the generated prompt.
    extra_prompt: str = ""
    #: Blend the original face back in after generation (Phase 2).
    preserve_face: bool = True
    #: Match output colour/lighting to the source photo (Phase 2).
    harmonize_colors: bool = True

    def resolved_seed(self) -> int:
        """Return a concrete seed, generating one when unset or negative."""
        if self.seed is None or self.seed < 0:
            return int(time.time_ns() % (2**31 - 1))
        return int(self.seed)


@dataclass
class TryOnRequest:
    """Everything one generation needs."""

    person: PersonAssets
    garment: GarmentAssets
    category: Category
    options: TryOnOptions = field(default_factory=TryOnOptions)
    progress: ProgressCallback = _noop_progress


@dataclass
class TryOnResult:
    """One generated try-on plus provenance for the UI and comparison grids."""

    image: Image.Image
    backend_id: BackendId
    prompt: str
    seed: int
    steps: int
    duration_s: float
    #: Free-form extras: intermediate masks, warp previews, timings.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> ModelSpec:
        """Static metadata for the backend that produced this result."""
        return MODEL_SPECS[self.backend_id]

    def summary(self) -> str:
        """Markdown caption shown under the output image."""
        return (
            f"**{self.spec.repo_id}** · {self.steps} steps · seed `{self.seed}` · "
            f"{self.duration_s:.1f}s · license {self.spec.license_name}"
        )


# --------------------------------------------------------------------------- #
# Abstract backend
# --------------------------------------------------------------------------- #


class TryOnBackend(abc.ABC):
    """Base class for every try-on model.

    Subclasses implement :meth:`_load` and :meth:`_generate`. The base class
    owns the cross-cutting concerns: lazy loading, license gating, OOM
    translation, timing and VRAM release.
    """

    #: Which backend this class implements. Set by each subclass.
    backend_id: BackendId

    #: ``"vton"`` or ``"edit"``. Used by the router and the comparison script.
    family: str = "edit"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._loaded_at: float | None = None

    # -- introspection ----------------------------------------------------- #

    @property
    def spec(self) -> ModelSpec:
        """Static metadata (repo id, license, defaults) for this backend."""
        return MODEL_SPECS[self.backend_id]

    @property
    def is_loaded(self) -> bool:
        """True when weights are resident."""
        return self._pipeline is not None

    @property
    def supported_categories(self) -> frozenset[Category]:
        """Categories this backend can handle. Override in subclasses."""
        return frozenset(Category)

    def supports(self, category: Category) -> bool:
        """True when ``category`` is in :attr:`supported_categories`."""
        return category in self.supported_categories

    # -- license gate ------------------------------------------------------ #

    def check_license(self) -> None:
        """Refuse to load a non-commercial backend unless opted in.

        This is the guard that keeps a CC BY-NC-SA model out of a product you
        sell. It runs before any weights are fetched.

        Raises
        ------
        LicenseRestrictedError
        """
        if self.spec.license_class is LicenseClass.COMMERCIAL:
            return
        if CONFIG.allow_noncommercial:
            logger.warning(
                "Loading NON-COMMERCIAL backend %s (%s). This output may not be "
                "used in a commercial product.",
                self.backend_id.value,
                self.spec.license_name,
            )
            return
        raise LicenseRestrictedError(
            f"{self.spec.repo_id} is licensed {self.spec.license_name} and is "
            "disabled by default. Export ALLOW_NONCOMMERCIAL=1 to enable it for "
            "research or personal use.",
            detail=f"backend={self.backend_id.value}",
        )

    # -- lifecycle --------------------------------------------------------- #

    @abc.abstractmethod
    def _load(self) -> Any:
        """Instantiate and return the underlying pipeline. Called once."""

    def load(self) -> None:
        """Load weights if not already resident.

        Raises
        ------
        LicenseRestrictedError, ModelLoadError
        """
        if self.is_loaded:
            return
        self.check_license()

        hardware = detect_hardware()
        logger.info(
            "Loading %s (%s) on %s | quantize=%s | VRAM before: %s",
            self.backend_id.value,
            self.spec.repo_id,
            hardware.device,
            resolve_quantization(CONFIG.runtime.quantize),
            vram_report(),
        )
        started = time.perf_counter()
        self._pipeline = self._load()
        self._loaded_at = time.time()
        logger.info(
            "Loaded %s in %.1fs | VRAM after: %s",
            self.backend_id.value,
            time.perf_counter() - started,
            vram_report(),
        )

    def unload(self) -> None:
        """Drop the pipeline and release VRAM.

        Called by the registry before a different heavy backend loads, so the
        two never co-reside on a 16 GB card.
        """
        if not self.is_loaded:
            return
        logger.info("Unloading %s | VRAM before: %s", self.backend_id.value, vram_report())
        self._pipeline = None
        self._loaded_at = None
        free_vram()
        logger.info("Unloaded %s | VRAM after: %s", self.backend_id.value, vram_report())

    # -- generation -------------------------------------------------------- #

    @abc.abstractmethod
    def _generate(self, request: TryOnRequest) -> TryOnResult:
        """Do the actual work. Weights are guaranteed loaded."""

    def generate(self, request: TryOnRequest) -> TryOnResult:
        """Generate a try-on image.

        Wraps :meth:`_generate` with lazy loading, timing and error
        translation, so callers only ever see :class:`TrialRoomError`
        subclasses with friendly messages.

        Parameters
        ----------
        request:
            Preprocessed person, garment, category and options.

        Returns
        -------
        TryOnResult

        Raises
        ------
        BackendError
            Including :class:`OutOfMemoryError` and
            :class:`LicenseRestrictedError`.
        """
        if not self.supports(request.category):
            raise BackendError(
                f"{self.spec.repo_id} does not support {request.category.label}.",
                detail=f"supported={sorted(c.value for c in self.supported_categories)}",
            )

        request.progress(0.05, f"Loading {self.backend_id.value}...")
        self.load()

        started = time.perf_counter()
        try:
            result = self._generate(request)
        except Exception as exc:
            if is_oom_error(exc):
                # Free what we can so the next attempt has a chance.
                self.unload()
                raise OutOfMemoryError(detail=str(exc)[:400]) from exc
            if isinstance(exc, BackendError):
                raise
            raise BackendError(detail=f"{type(exc).__name__}: {exc}"[:400]) from exc

        result.duration_s = time.perf_counter() - started
        logger.info(
            "%s generated in %.1fs (seed=%s, steps=%s)",
            self.backend_id.value,
            result.duration_s,
            result.seed,
            result.steps,
        )
        request.progress(0.95, "Finishing up...")
        return result

    # -- helpers for subclasses -------------------------------------------- #

    def _torch_dtype(self) -> Any:
        """Resolve the working torch dtype for this machine."""
        return resolve_dtype(CONFIG.runtime.dtype)

    def _apply_memory_savers(self, pipeline: Any) -> Any:
        """Enable offloading and slicing according to the runtime config.

        Every call is guarded because support varies by pipeline class and by
        diffusers version; a missing method must not break loading.

        Parameters
        ----------
        pipeline:
            A diffusers pipeline instance.

        Returns
        -------
        Any
            The same pipeline, mutated in place.
        """
        runtime = CONFIG.runtime
        hardware = detect_hardware()

        if not hardware.is_cuda:
            logger.warning("No CUDA device; skipping memory optimisations.")
            return pipeline

        # Sequential offload is the most aggressive and is mutually exclusive
        # with model offload; prefer it on very small cards.
        if runtime.sequential_cpu_offload:
            _try_call(pipeline, "enable_sequential_cpu_offload")
        elif runtime.model_cpu_offload:
            _try_call(pipeline, "enable_model_cpu_offload")
        else:
            _try_call(pipeline, "to", "cuda")

        if runtime.attention_slicing:
            _try_call(pipeline, "enable_attention_slicing", "auto")
        if runtime.vae_slicing:
            _try_call(pipeline, "enable_vae_slicing")
        if runtime.vae_tiling:
            _try_call(pipeline, "enable_vae_tiling")

        return pipeline


def _try_call(obj: Any, method: str, *args: Any) -> bool:
    """Call ``obj.method(*args)`` if it exists, logging and swallowing failures.

    Returns
    -------
    bool
        True when the call succeeded.
    """
    func = getattr(obj, method, None)
    if func is None:
        logger.debug("%s has no %s(); skipping.", type(obj).__name__, method)
        return False
    try:
        func(*args)
        logger.debug("Applied %s(%s)", method, ", ".join(map(repr, args)))
        return True
    except Exception as exc:  # noqa: BLE001 - optimisations are best-effort
        logger.debug("%s() failed: %s", method, exc)
        return False


#: Default negative prompt shared by the editing backends.
NEGATIVE_PROMPT: Final[str] = (
    "different face, different person, changed facial features, distorted face, "
    "extra limbs, extra arms, missing limbs, deformed hands, fused fingers, "
    "changed body proportions, changed background, blurry, low resolution, "
    "watermark, text, logo, oversaturated, plastic skin, cartoon, 3d render"
)
