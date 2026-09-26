"""Central configuration for AI Trial Room.

Every tunable knob lives here so that deployment targets (local dev box,
Kaggle T4, Hugging Face Space) can be switched with environment variables
rather than code edits.

Licensing note
--------------
This project is designed to be *commercially deployable*. Backends are tagged
with a :class:`LicenseClass` and the router refuses to serve a non-commercial
backend unless ``ALLOW_NONCOMMERCIAL=1`` is exported. See ``README.md`` for the
full license matrix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"
EXAMPLES_DIR: Final[Path] = ASSETS_DIR / "examples"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
CACHE_DIR: Final[Path] = Path(os.environ.get("AITR_CACHE_DIR", PROJECT_ROOT / ".cache"))


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable tolerantly."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an int environment variable, falling back on malformed input."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on malformed input."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class Category(str, Enum):
    """Garment categories supported by the app."""

    DRESS = "dress"
    KURTI = "kurti"
    KURTA = "kurta"
    SAREE = "saree"
    LEHENGA = "lehenga"

    @property
    def label(self) -> str:
        """Human-readable label for the Gradio dropdown."""
        return _CATEGORY_LABELS[self]

    @classmethod
    def from_label(cls, label: str) -> Category:
        """Resolve a UI label (``"Saree"``) back to a :class:`Category`."""
        try:
            return cls(label.strip().lower())
        except ValueError as exc:  # pragma: no cover - defensive
            valid = ", ".join(c.label for c in cls)
            raise ValueError(
                f"Unknown category {label!r}. Expected one of: {valid}"
            ) from exc

    @property
    def is_draped(self) -> bool:
        """True for garments that are draped rather than warped onto the body.

        Draped garments (saree, lehenga) have no fixed 2-D cut, so
        garment-warping models cannot represent them. These always route to the
        reference-editing backend.
        """
        return self in {Category.SAREE, Category.LEHENGA}


_CATEGORY_LABELS: Final[dict[Category, str]] = {
    Category.DRESS: "Dress",
    Category.KURTI: "Kurti",
    Category.KURTA: "Kurta",
    Category.SAREE: "Saree",
    Category.LEHENGA: "Lehenga",
}


class DrapeStyle(str, Enum):
    """Regional saree draping styles (Phase 2 turns these into prompts)."""

    NIVI = "nivi"
    BENGALI = "bengali"
    GUJARATI = "gujarati"
    NAUVARI = "nauvari"

    @property
    def label(self) -> str:
        """Human-readable label for the Gradio dropdown."""
        return _DRAPE_LABELS[self]

    @classmethod
    def from_label(cls, label: str) -> DrapeStyle:
        """Resolve a UI label back to a :class:`DrapeStyle`."""
        key = label.strip().lower().split(" ")[0]
        try:
            return cls(key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown drape style {label!r}") from exc


_DRAPE_LABELS: Final[dict[DrapeStyle, str]] = {
    DrapeStyle.NIVI: "Nivi (Andhra / modern standard)",
    DrapeStyle.BENGALI: "Bengali (Atpoure)",
    DrapeStyle.GUJARATI: "Gujarati (Seedha pallu)",
    DrapeStyle.NAUVARI: "Nauvari (Maharashtrian)",
}


class LicenseClass(str, Enum):
    """Coarse license buckets used to gate backends."""

    COMMERCIAL = "commercial"
    """Permissive (Apache-2.0 / MIT). Safe to sell."""

    NON_COMMERCIAL = "non_commercial"
    """CC BY-NC-SA or similar. Research / portfolio only."""


class BackendId(str, Enum):
    """Registered backend identifiers."""

    QWEN_EDIT = "qwen_edit"
    FLUX_KLEIN = "flux_klein"
    CATVTON = "catvton"


# --------------------------------------------------------------------------- #
# Model settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for one generative backend."""

    backend_id: BackendId
    repo_id: str
    pipeline_class: str
    license_name: str
    license_class: LicenseClass
    license_url: str
    #: Approximate VRAM in GB at bf16 with model CPU offload enabled.
    vram_gb_bf16: float
    #: Default sampler settings, overridable from the UI.
    default_steps: int
    default_true_cfg: float
    notes: str = ""

    @property
    def is_commercial(self) -> bool:
        """True when this backend may be used in a product you sell."""
        return self.license_class is LicenseClass.COMMERCIAL


MODEL_SPECS: Final[dict[BackendId, ModelSpec]] = {
    BackendId.QWEN_EDIT: ModelSpec(
        backend_id=BackendId.QWEN_EDIT,
        repo_id=os.environ.get("AITR_QWEN_REPO", "Qwen/Qwen-Image-Edit-2511"),
        pipeline_class="QwenImageEditPlusPipeline",
        license_name="Apache-2.0",
        license_class=LicenseClass.COMMERCIAL,
        license_url="https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
        vram_gb_bf16=16.0,
        default_steps=40,
        default_true_cfg=4.0,
        notes=(
            "Primary backend. Native 2-image (person + garment) reference "
            "conditioning; handles draped garments that warping models cannot."
        ),
    ),
    BackendId.FLUX_KLEIN: ModelSpec(
        backend_id=BackendId.FLUX_KLEIN,
        repo_id=os.environ.get("AITR_FLUX_REPO", "black-forest-labs/FLUX.2-klein-4B"),
        pipeline_class="Flux2KleinPipeline",
        license_name="Apache-2.0",
        license_class=LicenseClass.COMMERCIAL,
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        vram_gb_bf16=13.0,
        default_steps=28,
        default_true_cfg=4.0,
        notes="Faster commercial alternative. Lower VRAM, slightly less detail.",
    ),
    BackendId.CATVTON: ModelSpec(
        backend_id=BackendId.CATVTON,
        repo_id=os.environ.get("AITR_CATVTON_REPO", "zhengchong/CatVTON"),
        pipeline_class="CatVTONPipeline",
        license_name="CC BY-NC-SA 4.0",
        license_class=LicenseClass.NON_COMMERCIAL,
        license_url="https://github.com/Zheng-Chong/CatVTON",
        vram_gb_bf16=8.0,
        default_steps=50,
        default_true_cfg=2.5,
        notes=(
            "RESEARCH ONLY. Garment-warping baseline for quality comparison. "
            "Blocked unless ALLOW_NONCOMMERCIAL=1."
        ),
    ),
}

#: Human-parsing model used to build body / clothing masks.
HUMAN_PARSING_REPO: Final[str] = os.environ.get(
    "AITR_PARSING_REPO", "mattmdjaga/segformer_b2_clothes"
)
HUMAN_PARSING_LICENSE: Final[str] = "NVIDIA Source Code License (research)"

#: rembg model used for garment background removal (MIT tool, Apache weights).
REMBG_MODEL: Final[str] = os.environ.get("AITR_REMBG_MODEL", "u2net")


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeConfig:
    """Resolution, precision and memory-saving switches.

    Defaults target a 16 GB NVIDIA T4 (Kaggle / Colab free tier). On smaller
    cards set ``AITR_QUANTIZE=int4`` and keep ``sequential_cpu_offload`` on.
    """

    #: Working resolution as ``(width, height)``. 3:4 portrait suits full-body.
    width: int = field(default_factory=lambda: _env_int("AITR_WIDTH", 768))
    height: int = field(default_factory=lambda: _env_int("AITR_HEIGHT", 1024))

    #: Torch dtype name: ``bfloat16`` on Ampere+, ``float16`` on Turing (T4).
    dtype: str = field(default_factory=lambda: os.environ.get("AITR_DTYPE", "auto"))

    #: ``none`` | ``fp8`` | ``int4``. ``int4`` requires the nunchaku package.
    quantize: str = field(default_factory=lambda: os.environ.get("AITR_QUANTIZE", "auto"))

    # --- memory savers ---------------------------------------------------- #
    model_cpu_offload: bool = field(
        default_factory=lambda: _env_bool("AITR_MODEL_CPU_OFFLOAD", True)
    )
    sequential_cpu_offload: bool = field(
        default_factory=lambda: _env_bool("AITR_SEQ_CPU_OFFLOAD", False)
    )
    attention_slicing: bool = field(
        default_factory=lambda: _env_bool("AITR_ATTENTION_SLICING", True)
    )
    vae_slicing: bool = field(default_factory=lambda: _env_bool("AITR_VAE_SLICING", True))
    vae_tiling: bool = field(default_factory=lambda: _env_bool("AITR_VAE_TILING", True))

    #: Only one heavy backend may hold GPU memory at a time.
    max_resident_backends: int = field(
        default_factory=lambda: _env_int("AITR_MAX_RESIDENT_BACKENDS", 1)
    )

    # --- sampling --------------------------------------------------------- #
    default_steps: int = field(default_factory=lambda: _env_int("AITR_STEPS", 30))
    default_true_cfg: float = field(
        default_factory=lambda: _env_float("AITR_TRUE_CFG", 4.0)
    )
    default_seed: int = field(default_factory=lambda: _env_int("AITR_SEED", -1))

    @property
    def size(self) -> tuple[int, int]:
        """Working resolution as ``(width, height)``."""
        return self.width, self.height


@dataclass
class PreprocessConfig:
    """Thresholds for the preprocessing stage."""

    #: Reject person photos smaller than this on the short edge.
    min_person_short_edge: int = 384
    #: Reject garment photos smaller than this on the short edge.
    min_garment_short_edge: int = 256
    #: MediaPipe pose landmark confidence needed to accept a detection.
    pose_min_confidence: float = 0.5
    #: Fraction of pose landmarks that must be visible.
    pose_min_visible_ratio: float = 0.35
    #: Pixels of transparent padding kept around a cut-out garment.
    garment_margin_px: int = 24
    #: Dilation applied to the clothing mask, in pixels.
    mask_dilate_px: int = 12
    #: Gaussian blur sigma used to feather masks.
    mask_feather_sigma: float = 4.0


@dataclass
class PrivacyConfig:
    """Privacy guarantees surfaced in the UI and enforced in code."""

    #: Never write uploads to a durable location.
    persist_uploads: bool = False
    #: Delete per-request temp dirs when the request finishes.
    delete_temp_on_exit: bool = True
    #: Also delete generated outputs after they are handed to the browser.
    delete_outputs_on_exit: bool = field(
        default_factory=lambda: _env_bool("AITR_DELETE_OUTPUTS", False)
    )
    notice: str = "Photos are processed in memory and never stored."


@dataclass
class AppConfig:
    """Top-level configuration object consumed by ``app.py``."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    #: Default backend for every category unless overridden.
    default_backend: BackendId = BackendId.QWEN_EDIT
    #: Allow research-licensed backends to load. Never enable in production.
    allow_noncommercial: bool = field(
        default_factory=lambda: _env_bool("ALLOW_NONCOMMERCIAL", False)
    )
    #: Read from the environment only - never commit a token.
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)
    log_level: str = field(default_factory=lambda: os.environ.get("AITR_LOG_LEVEL", "INFO"))
    share_gradio: bool = field(default_factory=lambda: _env_bool("AITR_SHARE", False))
    server_port: int = field(default_factory=lambda: _env_int("AITR_PORT", 7860))

    def resolved_backend(self, category: Category) -> BackendId:
        """Return the backend that should serve ``category``.

        Draped garments always use the editing backend; non-draped garments use
        the configured default (also the editing backend by default, since every
        warping model available today is non-commercially licensed).
        """
        if category.is_draped:
            return BackendId.QWEN_EDIT
        return self.default_backend


#: Import this singleton rather than constructing :class:`AppConfig` yourself.
CONFIG: Final[AppConfig] = AppConfig()

APP_TITLE: Final[str] = "AI Trial Room"
APP_TAGLINE: Final[str] = "Virtual try-on built for Indian wear"
CONSENT_TEXT: Final[str] = "I confirm this is my photo, or I have permission to use it."
