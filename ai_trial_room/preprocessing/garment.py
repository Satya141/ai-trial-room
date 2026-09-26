"""Garment photo preprocessing: background removal, crop, centre, resize.

Shop catalogue photos arrive in every form imaginable - on a mannequin, held
up by hand, flat on a bed, on a hanger against a cluttered wall. Normalising
them to a clean, centred cut-out on white measurably improves how faithfully
the editing backend reproduces the print and colour.

For Indian wear there is one extra wrinkle: an unstitched saree photographed
folded shows almost none of the drape, while the *pallu* (the decorated end)
carries most of the design. We keep the whole garment rather than cropping to
a torso box, and record aspect ratio so prompts can mention it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_trial_room.config import CONFIG, REMBG_MODEL, Category, PreprocessConfig
from ai_trial_room.utils.errors import GarmentNotFoundError
from ai_trial_room.utils.image_io import (
    assert_min_size,
    flatten_on_color,
    letterbox,
    load_image,
    trim_alpha,
)
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Below this fraction of opaque pixels we assume background removal failed.
_MIN_FOREGROUND_RATIO: float = 0.02

#: Above this we assume it removed nothing (i.e. the whole frame stayed opaque
#: on a photo that clearly had a background).
_MAX_FOREGROUND_RATIO: float = 0.995


@dataclass(frozen=True)
class GarmentAssets:
    """Normalised garment ready to be fed to a backend.

    Attributes
    ----------
    image:
        RGB garment on a white canvas at the working resolution.
    cutout:
        RGBA cut-out with the background removed, tightly cropped.
    original:
        The uploaded image after EXIF correction.
    category:
        Category the garment was prepared for.
    background_removed:
        False when removal was skipped or failed and the raw photo was used.
    aspect_ratio:
        ``width / height`` of the tight cut-out. A long, narrow shape suggests
        an unstitched saree length rather than a worn garment.
    """

    image: Image.Image
    cutout: Image.Image
    original: Image.Image
    category: Category
    background_removed: bool
    aspect_ratio: float

    @property
    def looks_unstitched(self) -> bool:
        """Heuristic: an unstitched drape length photographed folded or spread.

        Used to add "as an unstitched fabric length" hints to the prompt so the
        model drapes the fabric instead of trying to fit a stitched outfit.
        """
        return self.category.is_draped and (self.aspect_ratio > 1.6 or self.aspect_ratio < 0.45)


@functools.lru_cache(maxsize=1)
def _load_rembg_session() -> Any | None:
    """Create and cache a rembg session.

    Returns
    -------
    Any or None
        A ``rembg`` session, or ``None`` when rembg is not installed.
    """
    try:
        from rembg import new_session
    except ImportError:
        logger.warning("rembg not installed; garment background removal disabled.")
        return None

    try:
        session = new_session(REMBG_MODEL)
    except Exception as exc:  # noqa: BLE001 - onnx/download failures vary
        logger.warning("Could not create rembg session %s: %s", REMBG_MODEL, exc)
        return None

    logger.info("Loaded rembg model %s", REMBG_MODEL)
    return session


def remove_background(image: Image.Image) -> tuple[Image.Image, bool]:
    """Remove the garment photo's background.

    Parameters
    ----------
    image:
        RGB garment photo.

    Returns
    -------
    tuple
        ``(rgba_image, succeeded)``. On failure the input is returned as opaque
        RGBA so the caller can still proceed.
    """
    session = _load_rembg_session()
    if session is None:
        return image.convert("RGBA"), False

    try:
        from rembg import remove

        result = remove(image, session=session).convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background removal failed: %s", exc)
        return image.convert("RGBA"), False

    alpha = np.asarray(result.getchannel("A"))
    foreground_ratio = float((alpha > 8).mean())

    if foreground_ratio < _MIN_FOREGROUND_RATIO:
        logger.warning(
            "Background removal kept only %.2f%% of pixels; discarding result.",
            foreground_ratio * 100,
        )
        return image.convert("RGBA"), False

    if foreground_ratio > _MAX_FOREGROUND_RATIO:
        logger.info("Background removal was a no-op (already isolated).")
        return result, True

    logger.info("Background removed; garment covers %.1f%% of frame.", foreground_ratio * 100)
    return result, True


def prepare_garment(
    source: str | Path | Image.Image | np.ndarray,
    category: Category,
    *,
    target_size: tuple[int, int] | None = None,
    cfg: PreprocessConfig | None = None,
    remove_bg: bool = True,
) -> GarmentAssets:
    """Run the full garment preprocessing pipeline.

    Parameters
    ----------
    source:
        Uploaded garment photo.
    category:
        Target category, used for heuristics and logging.
    target_size:
        Working resolution ``(width, height)``. Defaults to the runtime config.
    cfg:
        Preprocessing thresholds.
    remove_bg:
        Set False to skip background removal for an already-clean product shot.

    Returns
    -------
    GarmentAssets

    Raises
    ------
    MissingImageError, ImageTooSmallError, GarmentNotFoundError
    """
    cfg = cfg or CONFIG.preprocess
    target_size = target_size or CONFIG.runtime.size

    original = load_image(source, label="garment image")
    assert_min_size(original, cfg.min_garment_short_edge, label="garment image")

    if remove_bg:
        rgba, removed = remove_background(original)
    else:
        rgba, removed = original.convert("RGBA"), False

    cutout = trim_alpha(rgba, margin=cfg.garment_margin_px)
    if min(cutout.size) < 32:
        raise GarmentNotFoundError(detail=f"cutout collapsed to {cutout.size}")

    aspect_ratio = cutout.width / cutout.height

    # Letterbox onto white: the editing backend sees RGB, and a neutral
    # backdrop stops it borrowing colour from the garment photo's surroundings.
    flat = flatten_on_color(cutout, (255, 255, 255))
    image, _ = letterbox(flat, target_size, fill=(255, 255, 255))

    assets = GarmentAssets(
        image=image,
        cutout=cutout,
        original=original,
        category=category,
        background_removed=removed,
        aspect_ratio=aspect_ratio,
    )
    logger.info(
        "Garment prepared: %sx%s -> cutout %sx%s (ar=%.2f, bg_removed=%s, unstitched=%s)",
        *original.size,
        *cutout.size,
        aspect_ratio,
        removed,
        assets.looks_unstitched,
    )
    return assets
