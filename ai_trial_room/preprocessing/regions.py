"""Semantic sub-regions of a draped garment, derived from pose and parsing.

Why this module exists
----------------------
Phase 1 produced a single "repaint this" mask. That is enough to generate a
saree, but not enough to *improve* one, because the failure modes are
region-specific:

* The **pallu** is the decorated end carried over a shoulder. It holds most of a
  saree's design value and is where customers look first - and it is also where
  diffusion output softens most, because it is a large diagonal expanse of
  patterned fabric.
* The **pleats** at the front waist are what make a drape read as real. Models
  frequently render a smooth wrap with no pleat structure at all.
* The **blouse / choli** is a small, fitted region that must not inherit the
  saree's full print.
* The **skirt / lower drape** governs silhouette and hem behaviour.

Isolating these lets the refinement pass in
:mod:`ai_trial_room.backends.edit_backend` spend a second, shorter denoise on
just the pallu, and lets selective sharpening target the fabric rather than skin.

Everything here is geometric, derived from MediaPipe landmarks plus the ATR
parsing map. No extra model, so it costs microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ai_trial_room.config import Category, DrapeStyle
from ai_trial_room.preprocessing.person import (
    AtrLabel,
    ParseResult,
    PoseLandmark,
    PoseResult,
)
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Feather applied to every region mask, in pixels. Regions are used for
#: compositing, so a hard edge would show.
_FEATHER_PX: Final[float] = 6.0


class GarmentRegion(str, Enum):
    """Named sub-regions of a draped outfit."""

    PALLU = "pallu"
    """Decorated end over the shoulder (saree) or dupatta (lehenga)."""

    PLEATS = "pleats"
    """Front waist fan of folds."""

    BLOUSE = "blouse"
    """Fitted upper garment (choli)."""

    SKIRT = "skirt"
    """Lower drape or ghagra."""

    @property
    def label(self) -> str:
        """Human-readable name for logs and the UI."""
        return _REGION_LABELS[self]


_REGION_LABELS: Final[dict[GarmentRegion, str]] = {
    GarmentRegion.PALLU: "pallu / dupatta",
    GarmentRegion.PLEATS: "front pleats",
    GarmentRegion.BLOUSE: "blouse / choli",
    GarmentRegion.SKIRT: "skirt / lower drape",
}


@dataclass(frozen=True)
class RegionMasks:
    """Feathered masks for each sub-region of one outfit.

    Every mask is an 8-bit ``L`` image the same size as the working image, where
    255 means "inside this region".
    """

    masks: dict[GarmentRegion, Image.Image]
    image_size: tuple[int, int]

    def get(self, region: GarmentRegion) -> Image.Image | None:
        """Return the mask for ``region``, or ``None`` if it was not derived."""
        return self.masks.get(region)

    def coverage(self, region: GarmentRegion) -> float:
        """Fraction of the frame covered by ``region``, 0-1."""
        mask = self.masks.get(region)
        if mask is None:
            return 0.0
        return float(np.asarray(mask, dtype=np.float32).mean() / 255.0)

    def is_usable(self, region: GarmentRegion, *, min_coverage: float = 0.004) -> bool:
        """True when ``region`` covers enough pixels to be worth refining.

        Refining a 20-pixel sliver costs a full denoise pass and changes nothing
        visible, so the refinement path skips regions below this threshold.
        """
        return self.coverage(region) >= min_coverage


def _feather(mask: Image.Image, sigma: float = _FEATHER_PX) -> Image.Image:
    """Blur a binary mask so composites through it have no visible edge."""
    return mask.filter(ImageFilter.GaussianBlur(sigma))


def _polygon_mask(
    size: tuple[int, int], points: list[tuple[float, float]]
) -> Image.Image:
    """Rasterise a polygon into a feathered 8-bit mask."""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=255)
    return _feather(mask)


#: Pallu band half-width at the shoulder, as a fraction of shoulder span.
_PALLU_HALF_WIDTH_TOP: Final[float] = 0.28

#: Pallu band half-width at the loose end - it flares as it falls.
_PALLU_HALF_WIDTH_BOTTOM: Final[float] = 0.42


def _pallu_polygon(
    pose: PoseResult,
    size: tuple[int, int],
    shoulder: str,
) -> list[tuple[float, float]]:
    """Trace the pallu as a diagonal band across the torso.

    Anatomy this encodes
    --------------------
    In a Nivi-style drape the pallu is tucked at the front waist on one side,
    rises diagonally across the torso, passes over the opposite shoulder and
    falls down the back. Seen from the front, that is a *band* running from one
    shoulder down toward the opposite hip - not a panel covering the whole chest.

    So the band is built along the shoulder-to-opposite-hip axis, with a width
    proportional to shoulder span, flaring toward the loose end. Getting this
    right matters because the band is what the refinement pass repaints: a mask
    covering the whole torso would let a second denoise pass alter the blouse and
    the silhouette, which is exactly what refinement must not touch.

    Parameters
    ----------
    pose:
        Body geometry.
    size:
        ``(width, height)`` of the working image.
    shoulder:
        ``"left"`` or ``"right"`` - which shoulder the pallu passes over.

    Returns
    -------
    list
        Four polygon vertices, or an empty list when landmarks are missing.
    """
    width, height = size
    near = PoseLandmark.LEFT_SHOULDER if shoulder == "left" else PoseLandmark.RIGHT_SHOULDER
    far = PoseLandmark.RIGHT_SHOULDER if shoulder == "left" else PoseLandmark.LEFT_SHOULDER
    far_hip = PoseLandmark.RIGHT_HIP if shoulder == "left" else PoseLandmark.LEFT_HIP

    points = [pose.get(k) for k in (near, far, far_hip)]
    if any(p is None for p in points):
        return []

    near_sh, far_sh, far_hp = points  # type: ignore[misc]
    span = abs(near_sh[0] - far_sh[0]) or 1.0

    # Push the band slightly off the shoulder so the fabric reads as hanging
    # over it rather than painted onto the body.
    outward = 0.30 * span * (-1 if shoulder == "left" else 1)

    start = (near_sh[0] + outward, max(0.0, near_sh[1] - 0.15 * span))
    end = (
        far_hp[0] + outward * 0.5,
        min(float(height), far_hp[1] + 0.50 * (height - far_hp[1])),
    )

    # Unit vector along the band, and its perpendicular.
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    half_top = _PALLU_HALF_WIDTH_TOP * span
    half_bottom = _PALLU_HALF_WIDTH_BOTTOM * span

    polygon = [
        (start[0] + px * half_top, start[1] + py * half_top),
        (start[0] - px * half_top, start[1] - py * half_top),
        (end[0] - px * half_bottom, end[1] - py * half_bottom),
        (end[0] + px * half_bottom, end[1] + py * half_bottom),
    ]
    return [(min(max(x, 0.0), width), min(max(y, 0.0), height)) for x, y in polygon]


def _spread_pallu_polygon(
    pose: PoseResult, size: tuple[int, int]
) -> list[tuple[float, float]]:
    """Pallu spread open across the chest - the Gujarati *seedha pallu* look.

    Here the decorated end faces forward rather than hanging behind, so the
    region is a broad chest panel rather than a diagonal band.
    """
    width, height = size
    keys = (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP,
    )
    points = [pose.get(k) for k in keys]
    if any(p is None for p in points):
        return []

    left_sh, right_sh, left_hp, right_hp = points  # type: ignore[misc]
    span = abs(left_sh[0] - right_sh[0]) or 1.0
    pad = 0.22 * span
    top = max(0.0, min(left_sh[1], right_sh[1]) - 0.12 * span)
    bottom = min(float(height), max(left_hp[1], right_hp[1]) + 0.25 * span)
    x0 = max(0.0, min(left_sh[0], right_sh[0]) - pad)
    x1 = min(float(width), max(left_sh[0], right_sh[0]) + pad)

    return [(x0, top), (x1, top), (x1, bottom), (x0, bottom)]


def _pleat_polygon(pose: PoseResult, size: tuple[int, int]) -> list[tuple[float, float]]:
    """Trace the fan of pleats tucked at the front waist.

    A trapezoid from the waist down to roughly knee height, flaring outward - the
    pleats fan as they descend.
    """
    width, height = size
    left_hip = pose.get(PoseLandmark.LEFT_HIP)
    right_hip = pose.get(PoseLandmark.RIGHT_HIP)
    if left_hip is None or right_hip is None:
        return []

    hip_span = abs(left_hip[0] - right_hip[0]) or 1.0
    centre = (left_hip[0] + right_hip[0]) / 2.0
    waist_y = min(left_hip[1], right_hip[1])

    knee = pose.get(PoseLandmark.LEFT_KNEE) or pose.get(PoseLandmark.RIGHT_KNEE)
    bottom_y = min(float(height), knee[1] if knee else waist_y + 2.2 * hip_span)

    top_half = 0.52 * hip_span
    bottom_half = 0.95 * hip_span

    polygon = [
        (centre - top_half, waist_y),
        (centre + top_half, waist_y),
        (centre + bottom_half, bottom_y),
        (centre - bottom_half, bottom_y),
    ]
    return [(min(max(x, 0.0), width), min(max(y, 0.0), height)) for x, y in polygon]


def _blouse_polygon(pose: PoseResult, size: tuple[int, int]) -> list[tuple[float, float]]:
    """Trace the fitted blouse: shoulders down to just above the waist."""
    width, height = size
    keys = (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP,
    )
    points = [pose.get(k) for k in keys]
    if any(p is None for p in points):
        return []

    left_sh, right_sh, left_hp, right_hp = points  # type: ignore[misc]
    span = abs(left_sh[0] - right_sh[0]) or 1.0
    shoulder_y = min(left_sh[1], right_sh[1])
    hip_y = min(left_hp[1], right_hp[1])
    # A choli ends well above the waist; stop at ~55% of torso height.
    bottom_y = shoulder_y + 0.55 * (hip_y - shoulder_y)
    pad = 0.14 * span

    x0 = max(0.0, min(left_sh[0], right_sh[0]) - pad)
    x1 = min(float(width), max(left_sh[0], right_sh[0]) + pad)
    return [(x0, shoulder_y), (x1, shoulder_y), (x1, bottom_y), (x0, bottom_y)]


def derive_regions(
    pose: PoseResult,
    parse: ParseResult,
    category: Category,
    *,
    drape_style: DrapeStyle = DrapeStyle.NIVI,
    pallu_shoulder: str = "left",
) -> RegionMasks:
    """Derive sub-region masks for a draped outfit.

    Parameters
    ----------
    pose:
        Body geometry from :func:`~ai_trial_room.preprocessing.person.detect_pose`.
    parse:
        Human parsing output, used to trim regions to the person's silhouette.
    category:
        Garment category. Non-draped categories get only a blouse-equivalent
        upper region, since they have no pallu or pleats.
    drape_style:
        Saree drape - determines whether the pallu hangs diagonally or is spread
        across the chest.
    pallu_shoulder:
        Which shoulder carries the pallu, from the drape spec.

    Returns
    -------
    RegionMasks
    """
    size = pose.image_size
    masks: dict[GarmentRegion, Image.Image] = {}

    if category.is_draped:
        if category is Category.SAREE and drape_style is DrapeStyle.GUJARATI:
            pallu_points = _spread_pallu_polygon(pose, size)
        else:
            pallu_points = _pallu_polygon(pose, size, pallu_shoulder)

        if pallu_points:
            masks[GarmentRegion.PALLU] = _polygon_mask(size, pallu_points)

        pleat_points = _pleat_polygon(pose, size)
        if pleat_points:
            masks[GarmentRegion.PLEATS] = _polygon_mask(size, pleat_points)

    blouse_points = _blouse_polygon(pose, size)
    if blouse_points:
        masks[GarmentRegion.BLOUSE] = _polygon_mask(size, blouse_points)

    if parse.available:
        skirt = parse.mask_for_labels(
            [AtrLabel.SKIRT, AtrLabel.DRESS, AtrLabel.PANTS, AtrLabel.LEFT_LEG, AtrLabel.RIGHT_LEG]
        )
        if skirt.any():
            masks[GarmentRegion.SKIRT] = _feather(
                Image.fromarray((skirt * 255).astype(np.uint8), mode="L")
            )

    regions = RegionMasks(masks=masks, image_size=size)
    logger.info(
        "Derived regions for %s: %s",
        category.value,
        ", ".join(
            f"{region.value}={regions.coverage(region) * 100:.1f}%" for region in masks
        )
        or "none",
    )
    return regions


def restrict_to_mask(region: Image.Image, allowed: Image.Image) -> Image.Image:
    """Clip ``region`` to the area where ``allowed`` is non-zero.

    Used to keep a geometric region inside the actual inpaint mask, so a pallu
    polygon cannot spill onto the background or the person's face.

    Parameters
    ----------
    region:
        Region mask to clip.
    allowed:
        Mask defining the permitted area, typically the inpaint mask.

    Returns
    -------
    Image.Image
        The clipped region mask.
    """
    if region.size != allowed.size:
        allowed = allowed.resize(region.size, Image.Resampling.LANCZOS)
    clipped = np.minimum(
        np.asarray(region.convert("L"), dtype=np.uint8),
        np.asarray(allowed.convert("L"), dtype=np.uint8),
    )
    return Image.fromarray(clipped, mode="L")
