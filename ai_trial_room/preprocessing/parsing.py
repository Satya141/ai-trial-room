"""Human parsing with a pluggable, licence-aware provider.

Why this module exists
----------------------
Phase 1 used ``mattmdjaga/segformer_b2_clothes`` for the garment mask. It works
well, and it inherits NVIDIA's **research-only** SegFormer licence - which made it
the single blocker between this project and something you can invoice for.

This module fixes that. The default provider is now **MediaPipe Selfie
Multiclass** (Apache-2.0, 106 K parameters, 447 KB), which segments
``background / hair / body-skin / face-skin / clothes / accessories``.

The catch, and how it is handled
--------------------------------
MediaPipe gives **one** ``clothes`` class. It cannot tell a kurti from the
trousers under it, and its ``body-skin`` does not separate arms from legs. Used
naively, a kurti mask would repaint the customer's jeans.

So :func:`_split_by_pose` recovers the distinction geometrically, using the hip
landmarks this pipeline already computes: clothes above the hip line become
``UPPER_CLOTHES``, clothes below become ``PANTS``, skin above becomes arms, skin
below becomes legs. That restores every label
:data:`~ai_trial_room.preprocessing.person._CATEGORY_GARMENT_LABELS` needs, from a
permissively licensed model.

Providers
---------
========================  ==============  =======================================
Provider                  Licence         Notes
========================  ==============  =======================================
``mediapipe`` (default)    Apache-2.0 ✅   Coarse classes, split by pose geometry
``segformer``              NVIDIA SCL ⚠️   Finer labels; research use only, gated
``pose_only``              none ✅         No model at all; geometric fallback
========================  ==============  =======================================

``segformer`` is gated behind ``ALLOW_NONCOMMERCIAL=1`` exactly like CatVTON, so
a commercial deployment cannot reach for it by accident.
"""

from __future__ import annotations

import functools
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from ai_trial_room.config import CACHE_DIR, CONFIG, HUMAN_PARSING_REPO, LicenseClass
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Apache-2.0 model from Google, downloaded once and cached.
MEDIAPIPE_MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

MEDIAPIPE_MODEL_NAME: Final[str] = "selfie_multiclass_256x256.tflite"

#: Seconds to wait for the model download before giving up.
_DOWNLOAD_TIMEOUT: Final[int] = 60


class SelfieClass(IntEnum):
    """Classes emitted by MediaPipe Selfie Multiclass."""

    BACKGROUND = 0
    HAIR = 1
    BODY_SKIN = 2
    FACE_SKIN = 3
    CLOTHES = 4
    ACCESSORIES = 5


class ParsingProvider(str, Enum):
    """Which human-parsing model to use."""

    MEDIAPIPE = "mediapipe"
    SEGFORMER = "segformer"
    POSE_ONLY = "pose_only"

    @property
    def license_name(self) -> str:
        """Licence of this provider's weights."""
        return _PROVIDER_LICENSES[self][0]

    @property
    def license_class(self) -> LicenseClass:
        """Whether this provider may be used commercially."""
        return _PROVIDER_LICENSES[self][1]

    @property
    def is_commercial(self) -> bool:
        """True when this provider is safe for a product you sell."""
        return self.license_class is LicenseClass.COMMERCIAL

    @property
    def description(self) -> str:
        """One-line summary for logs and the About panel."""
        return _PROVIDER_DESCRIPTIONS[self]


_PROVIDER_LICENSES: Final[dict[ParsingProvider, tuple[str, LicenseClass]]] = {
    ParsingProvider.MEDIAPIPE: ("Apache-2.0", LicenseClass.COMMERCIAL),
    ParsingProvider.SEGFORMER: (
        "NVIDIA Source Code License (research)",
        LicenseClass.NON_COMMERCIAL,
    ),
    ParsingProvider.POSE_ONLY: ("n/a (no model)", LicenseClass.COMMERCIAL),
}

_PROVIDER_DESCRIPTIONS: Final[dict[ParsingProvider, str]] = {
    ParsingProvider.MEDIAPIPE: (
        "MediaPipe Selfie Multiclass; coarse classes refined with pose geometry"
    ),
    ParsingProvider.SEGFORMER: (
        "SegFormer/ATR; 18 fine labels but research-licensed weights"
    ),
    ParsingProvider.POSE_ONLY: "no segmentation model; geometric masks only",
}


@dataclass(frozen=True)
class ParsingOutcome:
    """A parsed label map plus which provider actually produced it."""

    label_map: np.ndarray
    """``(H, W)`` ``uint8`` array of
    :class:`~ai_trial_room.preprocessing.person.AtrLabel` values."""

    provider: ParsingProvider
    available: bool
    """False when no model ran and the caller should use a geometric fallback."""


def resolve_provider(requested: str | None = None) -> ParsingProvider:
    """Decide which provider to use.

    Parameters
    ----------
    requested:
        ``"auto"``, a provider name, or ``None`` to read the config.

    Returns
    -------
    ParsingProvider
        Never a non-commercial provider unless ``ALLOW_NONCOMMERCIAL=1``.
    """
    raw = (requested or CONFIG.parsing_provider or "auto").strip().lower()

    if raw != "auto":
        try:
            provider = ParsingProvider(raw)
        except ValueError:
            logger.warning(
                "Unknown parsing provider %r; falling back to %s.",
                raw,
                ParsingProvider.MEDIAPIPE.value,
            )
            return ParsingProvider.MEDIAPIPE

        if not provider.is_commercial and not CONFIG.allow_noncommercial:
            logger.warning(
                "Parsing provider %s is licensed %s and is disabled. Using %s "
                "instead. Export ALLOW_NONCOMMERCIAL=1 to enable it for research.",
                provider.value,
                provider.license_name,
                ParsingProvider.MEDIAPIPE.value,
            )
            return ParsingProvider.MEDIAPIPE
        return provider

    return ParsingProvider.MEDIAPIPE


# --------------------------------------------------------------------------- #
# MediaPipe provider
# --------------------------------------------------------------------------- #


def _model_path() -> Path | None:
    """Return the cached MediaPipe model, downloading it if needed.

    Returns
    -------
    Path or None
        ``None`` when the download failed, so the caller can degrade.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / MEDIAPIPE_MODEL_NAME

    if destination.is_file() and destination.stat().st_size > 10_000:
        return destination

    logger.info("Downloading %s (447 KB, Apache-2.0)...", MEDIAPIPE_MODEL_NAME)
    partial = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(MEDIAPIPE_MODEL_URL, timeout=_DOWNLOAD_TIMEOUT) as response:
            partial.write_bytes(response.read())
        partial.replace(destination)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning("Could not download the parsing model: %s", exc)
        partial.unlink(missing_ok=True)
        return None

    logger.info("Cached parsing model at %s", destination)
    return destination


@functools.lru_cache(maxsize=1)
def _load_mediapipe_segmenter() -> Any | None:
    """Load and cache the MediaPipe image segmenter.

    Returns
    -------
    Any or None
        An ``ImageSegmenter``, or ``None`` when MediaPipe or the weights are
        unavailable.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe is not installed; human parsing unavailable.")
        return None

    model = _model_path()
    if model is None:
        return None

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model))
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
    except Exception as exc:  # noqa: BLE001 - task API surface varies by version
        logger.warning("Could not create the MediaPipe segmenter: %s", exc)
        return None

    logger.info("Loaded MediaPipe Selfie Multiclass (Apache-2.0).")
    return segmenter


def parse_with_mediapipe(image: Image.Image, pose: Any | None = None) -> ParsingOutcome:
    """Segment with MediaPipe, then split coarse classes using pose geometry.

    Parameters
    ----------
    image:
        RGB image to parse.
    pose:
        :class:`~ai_trial_room.preprocessing.person.PoseResult` used to split
        ``clothes`` and ``body-skin`` at the hip line. Without it the split is
        skipped and clothing is labelled upper-body only.

    Returns
    -------
    ParsingOutcome
    """
    from ai_trial_room.preprocessing.person import AtrLabel

    width, height = image.size
    segmenter = _load_mediapipe_segmenter()

    if segmenter is None:
        return ParsingOutcome(
            label_map=np.zeros((height, width), np.uint8),
            provider=ParsingProvider.MEDIAPIPE,
            available=False,
        )

    try:
        import mediapipe as mp

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)),
        )
        result = segmenter.segment(mp_image)
        categories = np.asarray(result.category_mask.numpy_view(), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MediaPipe segmentation failed: %s", exc)
        return ParsingOutcome(
            label_map=np.zeros((height, width), np.uint8),
            provider=ParsingProvider.MEDIAPIPE,
            available=False,
        )

    # The model works at 256x256; resize its output back with nearest-neighbour
    # so class ids are never interpolated into nonexistent labels.
    if categories.shape != (height, width):
        categories = np.asarray(
            Image.fromarray(categories, mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )

    label_map = np.zeros((height, width), np.uint8)
    label_map[categories == SelfieClass.HAIR] = int(AtrLabel.HAIR)
    label_map[categories == SelfieClass.FACE_SKIN] = int(AtrLabel.FACE)
    label_map[categories == SelfieClass.ACCESSORIES] = int(AtrLabel.BAG)
    label_map[categories == SelfieClass.CLOTHES] = int(AtrLabel.UPPER_CLOTHES)
    label_map[categories == SelfieClass.BODY_SKIN] = int(AtrLabel.LEFT_ARM)

    if pose is not None:
        _split_by_pose(label_map, categories, pose)

    coverage = float((label_map != int(AtrLabel.BACKGROUND)).mean())
    logger.info(
        "MediaPipe parsing done: person covers %.1f%% of frame%s",
        coverage * 100,
        "" if pose is None else " (clothes split at the hip line)",
    )
    return ParsingOutcome(
        label_map=label_map, provider=ParsingProvider.MEDIAPIPE, available=coverage > 0.01
    )


def _split_by_pose(label_map: np.ndarray, categories: np.ndarray, pose: Any) -> None:
    """Split MediaPipe's coarse classes at the hip line, in place.

    MediaPipe has a single ``clothes`` class, so a kurti and the trousers beneath
    it are indistinguishable - and a kurti mask built from that would repaint the
    customer's jeans. The hip landmarks give the boundary for free.

    Parameters
    ----------
    label_map:
        Label map to refine, modified in place.
    categories:
        Raw MediaPipe category mask.
    pose:
        Pose result supplying hip landmarks.
    """
    from ai_trial_room.preprocessing.person import AtrLabel, PoseLandmark

    left_hip = pose.get(PoseLandmark.LEFT_HIP)
    right_hip = pose.get(PoseLandmark.RIGHT_HIP)
    if left_hip is None or right_hip is None:
        logger.debug("No hip landmarks; leaving MediaPipe classes unsplit.")
        return

    hip_y = int(round(min(left_hip[1], right_hip[1])))
    hip_y = max(0, min(hip_y, label_map.shape[0] - 1))

    below = np.zeros_like(label_map, dtype=bool)
    below[hip_y:, :] = True

    clothes = categories == SelfieClass.CLOTHES
    skin = categories == SelfieClass.BODY_SKIN

    # Lower-body clothing: labelled PANTS so a kurti mask leaves it alone while a
    # saree mask (which includes PANTS) repaints it.
    label_map[clothes & below] = int(AtrLabel.PANTS)
    # Skin below the hips is legs, which draped masks must cover.
    label_map[skin & below] = int(AtrLabel.LEFT_LEG)


# --------------------------------------------------------------------------- #
# SegFormer provider (research only)
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _load_segformer() -> tuple[Any, Any] | None:
    """Load and cache the SegFormer human-parsing model.

    Returns
    -------
    tuple or None
        ``(processor, model)``, or ``None`` when unavailable.
    """
    try:
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        logger.warning("transformers unavailable; SegFormer parsing disabled.")
        return None

    try:
        processor = SegformerImageProcessor.from_pretrained(HUMAN_PARSING_REPO)
        model = AutoModelForSemanticSegmentation.from_pretrained(HUMAN_PARSING_REPO)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %s: %s", HUMAN_PARSING_REPO, exc)
        return None

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    logger.warning(
        "Loaded RESEARCH-LICENSED parsing model %s. Output must not be used "
        "commercially.",
        HUMAN_PARSING_REPO,
    )
    return processor, model


def parse_with_segformer(image: Image.Image) -> ParsingOutcome:
    """Segment with SegFormer/ATR, giving 18 fine-grained labels.

    Research use only - see this module's docstring.
    """
    width, height = image.size
    loaded = _load_segformer()

    if loaded is None:
        return ParsingOutcome(
            label_map=np.zeros((height, width), np.uint8),
            provider=ParsingProvider.SEGFORMER,
            available=False,
        )

    processor, model = loaded
    try:
        import torch

        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        label_map = upsampled.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SegFormer parsing failed: %s", exc)
        return ParsingOutcome(
            label_map=np.zeros((height, width), np.uint8),
            provider=ParsingProvider.SEGFORMER,
            available=False,
        )

    coverage = float((label_map != 0).mean())
    logger.info("SegFormer parsing done: person covers %.1f%% of frame", coverage * 100)
    return ParsingOutcome(
        label_map=label_map, provider=ParsingProvider.SEGFORMER, available=coverage > 0.01
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_human(
    image: Image.Image,
    pose: Any | None = None,
    *,
    provider: str | None = None,
) -> ParsingOutcome:
    """Parse a person image using the configured provider.

    Falls back down the chain rather than failing: if the chosen provider cannot
    run, MediaPipe is tried, and finally an empty map is returned so the caller
    uses geometric masks.

    Parameters
    ----------
    image:
        RGB image to parse.
    pose:
        Pose result, used to refine MediaPipe's coarse classes.
    provider:
        Override the configured provider.

    Returns
    -------
    ParsingOutcome
    """
    chosen = resolve_provider(provider)

    if chosen is ParsingProvider.POSE_ONLY:
        logger.info("Parsing provider is pose_only; skipping segmentation.")
        return ParsingOutcome(
            label_map=np.zeros(image.size[::-1], np.uint8),
            provider=chosen,
            available=False,
        )

    if chosen is ParsingProvider.SEGFORMER:
        outcome = parse_with_segformer(image)
        if outcome.available:
            return outcome
        logger.info("SegFormer unavailable; trying MediaPipe.")
        chosen = ParsingProvider.MEDIAPIPE

    return parse_with_mediapipe(image, pose)


def provider_report() -> dict[str, str]:
    """Describe the active provider, for the UI's About panel."""
    provider = resolve_provider()
    return {
        "Provider": provider.value,
        "Model": provider.description,
        "Licence": provider.license_name,
        "Commercial use": "yes" if provider.is_commercial else "NO - research only",
    }
