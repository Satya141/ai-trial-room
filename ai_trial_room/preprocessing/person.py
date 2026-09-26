"""Person photo preprocessing: geometry, pose and human parsing.

The pipeline is deliberately fail-fast. A bad person photo is the single
largest cause of poor try-on results, so we detect and explain problems here
rather than letting the diffusion model produce something unusable.

Stages
------
1. Load, apply EXIF rotation, validate minimum resolution.
2. Letterbox to the model's working resolution (pad, never crop - cropping
   would cut off a saree hem).
3. Detect pose with MediaPipe and derive body geometry (shoulders, hips,
   visible extent).
4. Run human parsing (SegFormer/ATR) to get per-pixel garment and body labels.
5. Build the inpainting mask for the requested category.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
from PIL import Image, ImageFilter

from ai_trial_room.config import CONFIG, HUMAN_PARSING_REPO, Category, PreprocessConfig
from ai_trial_room.utils.errors import (
    NoPersonDetectedError,
    PersonPartiallyVisibleError,
)
from ai_trial_room.utils.image_io import (
    LetterboxInfo,
    assert_min_size,
    letterbox,
    load_image,
)
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Label schemas
# --------------------------------------------------------------------------- #


class AtrLabel(IntEnum):
    """Semantic classes emitted by ``mattmdjaga/segformer_b2_clothes`` (ATR)."""

    BACKGROUND = 0
    HAT = 1
    HAIR = 2
    SUNGLASSES = 3
    UPPER_CLOTHES = 4
    SKIRT = 5
    PANTS = 6
    DRESS = 7
    BELT = 8
    LEFT_SHOE = 9
    RIGHT_SHOE = 10
    FACE = 11
    LEFT_LEG = 12
    RIGHT_LEG = 13
    LEFT_ARM = 14
    RIGHT_ARM = 15
    BAG = 16
    SCARF = 17


#: Labels that must never be repainted - they carry the person's identity.
IDENTITY_LABELS: Final[frozenset[AtrLabel]] = frozenset(
    {AtrLabel.FACE, AtrLabel.HAIR, AtrLabel.HAT, AtrLabel.SUNGLASSES}
)

#: Existing-garment labels to erase, per target category.
#:
#: A saree or lehenga covers the legs and one shoulder, so their masks must
#: include the legs and skirt regions that an upper-body mask would leave
#: behind. Getting this wrong is what makes naive try-on leave jeans showing
#: under a saree.
_CATEGORY_GARMENT_LABELS: Final[dict[Category, frozenset[AtrLabel]]] = {
    Category.KURTI: frozenset({AtrLabel.UPPER_CLOTHES, AtrLabel.DRESS, AtrLabel.SCARF}),
    Category.KURTA: frozenset({AtrLabel.UPPER_CLOTHES, AtrLabel.DRESS, AtrLabel.SCARF}),
    Category.DRESS: frozenset(
        {
            AtrLabel.UPPER_CLOTHES,
            AtrLabel.DRESS,
            AtrLabel.SKIRT,
            AtrLabel.BELT,
            AtrLabel.SCARF,
        }
    ),
    Category.SAREE: frozenset(
        {
            AtrLabel.UPPER_CLOTHES,
            AtrLabel.DRESS,
            AtrLabel.SKIRT,
            AtrLabel.PANTS,
            AtrLabel.BELT,
            AtrLabel.SCARF,
            AtrLabel.LEFT_LEG,
            AtrLabel.RIGHT_LEG,
        }
    ),
    Category.LEHENGA: frozenset(
        {
            AtrLabel.UPPER_CLOTHES,
            AtrLabel.DRESS,
            AtrLabel.SKIRT,
            AtrLabel.PANTS,
            AtrLabel.BELT,
            AtrLabel.SCARF,
            AtrLabel.LEFT_LEG,
            AtrLabel.RIGHT_LEG,
        }
    ),
}


class PoseLandmark(IntEnum):
    """The subset of MediaPipe's 33 pose landmarks this project uses."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PoseResult:
    """Body geometry derived from pose landmarks.

    Coordinates are absolute pixels in the letterboxed image.
    """

    landmarks: dict[PoseLandmark, tuple[float, float, float]]
    """Landmark -> ``(x, y, visibility)``."""

    image_size: tuple[int, int]
    """``(width, height)`` of the image the landmarks refer to."""

    visible_ratio: float
    """Fraction of the 33 landmarks detected above the visibility threshold."""

    def get(self, landmark: PoseLandmark) -> tuple[float, float, float] | None:
        """Return ``(x, y, visibility)`` for ``landmark`` if it was detected."""
        return self.landmarks.get(landmark)

    def is_visible(self, landmark: PoseLandmark, threshold: float = 0.5) -> bool:
        """True when ``landmark`` was detected above ``threshold`` visibility."""
        point = self.landmarks.get(landmark)
        return point is not None and point[2] >= threshold

    @property
    def shoulder_width_px(self) -> float | None:
        """Horizontal distance between the shoulders, or ``None``."""
        left = self.get(PoseLandmark.LEFT_SHOULDER)
        right = self.get(PoseLandmark.RIGHT_SHOULDER)
        if left is None or right is None:
            return None
        return float(abs(left[0] - right[0]))

    @property
    def dominant_shoulder(self) -> str:
        """Which shoulder faces the camera - drives saree pallu placement.

        In a Nivi drape the pallu falls over the *left* shoulder. When the
        subject is turned, draping over the shoulder nearer the camera reads as
        more natural, so this picks the shoulder with higher visibility.
        """
        left = self.get(PoseLandmark.LEFT_SHOULDER)
        right = self.get(PoseLandmark.RIGHT_SHOULDER)
        if left is None and right is None:
            return "left"
        if left is None:
            return "right"
        if right is None:
            return "left"
        return "left" if left[2] >= right[2] else "right"

    @property
    def framing(self) -> str:
        """Classify visible body extent as ``full``, ``three_quarter`` or ``half``."""
        if self.is_visible(PoseLandmark.LEFT_ANKLE, 0.3) or self.is_visible(
            PoseLandmark.RIGHT_ANKLE, 0.3
        ):
            return "full"
        if self.is_visible(PoseLandmark.LEFT_KNEE, 0.3) or self.is_visible(
            PoseLandmark.RIGHT_KNEE, 0.3
        ):
            return "three_quarter"
        return "half"

    def torso_box(self) -> tuple[int, int, int, int] | None:
        """Bounding box ``(left, top, right, bottom)`` around shoulders and hips."""
        keys = (
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP,
        )
        points = [self.get(k) for k in keys]
        if any(p is None for p in points):
            return None
        xs = [p[0] for p in points if p]
        ys = [p[1] for p in points if p]
        width, height = self.image_size
        return (
            max(0, int(min(xs))),
            max(0, int(min(ys))),
            min(width, int(max(xs))),
            min(height, int(max(ys))),
        )


@dataclass(frozen=True)
class ParseResult:
    """Per-pixel human parsing output."""

    label_map: np.ndarray
    """``(H, W)`` ``uint8`` array of :class:`AtrLabel` values."""

    available: bool
    """False when the parsing model could not be loaded and a pose-derived
    fallback mask was used instead."""

    def mask_for_labels(self, labels: Sequence[AtrLabel]) -> np.ndarray:
        """Return a boolean mask that is True where any of ``labels`` appears."""
        wanted = np.array([int(label) for label in labels], dtype=self.label_map.dtype)
        return np.isin(self.label_map, wanted)

    @property
    def person_mask(self) -> np.ndarray:
        """Boolean mask of every non-background pixel."""
        return self.label_map != int(AtrLabel.BACKGROUND)


@dataclass(frozen=True)
class PersonAssets:
    """Everything the backends need about the person photo.

    Attributes
    ----------
    image:
        Letterboxed RGB image at the model's working resolution.
    original:
        The uploaded photo after EXIF correction, at its native size.
    letterbox_info:
        Transform needed to map results back to ``original``.
    pose:
        Derived body geometry.
    parse:
        Human parsing label map.
    inpaint_mask:
        Feathered 8-bit mask (white = repaint) for the requested category.
    category:
        The category this mask was built for.
    """

    image: Image.Image
    original: Image.Image
    letterbox_info: LetterboxInfo
    pose: PoseResult
    parse: ParseResult
    inpaint_mask: Image.Image
    category: Category

    @property
    def framing(self) -> str:
        """Shorthand for ``pose.framing``."""
        return self.pose.framing


# --------------------------------------------------------------------------- #
# Pose detection
# --------------------------------------------------------------------------- #


def detect_pose(image: Image.Image, cfg: PreprocessConfig | None = None) -> PoseResult:
    """Detect body landmarks with MediaPipe Pose.

    Parameters
    ----------
    image:
        RGB image to analyse.
    cfg:
        Preprocessing thresholds. Defaults to the global config.

    Returns
    -------
    PoseResult

    Raises
    ------
    NoPersonDetectedError
        If MediaPipe finds no pose, or too few landmarks are visible.
    """
    cfg = cfg or CONFIG.preprocess
    width, height = image.size

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise NoPersonDetectedError(
            "Pose detection is unavailable in this environment.",
            detail="mediapipe is not installed",
        ) from exc

    array = np.asarray(image.convert("RGB"))
    # static_image_mode=True disables temporal smoothing, which is what we want
    # for one-shot photos; model_complexity=2 is the most accurate variant and
    # costs ~100 ms on CPU, negligible next to diffusion.
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=cfg.pose_min_confidence,
    ) as detector:
        result = detector.process(array)

    if result.pose_landmarks is None:
        raise NoPersonDetectedError(detail="mediapipe returned no pose_landmarks")

    raw = result.pose_landmarks.landmark
    visible_count = sum(1 for lm in raw if lm.visibility >= cfg.pose_min_confidence)
    visible_ratio = visible_count / len(raw)

    landmarks: dict[PoseLandmark, tuple[float, float, float]] = {}
    for landmark in PoseLandmark:
        point = raw[int(landmark)]
        if point.visibility < 0.1:
            continue
        landmarks[landmark] = (point.x * width, point.y * height, float(point.visibility))

    pose = PoseResult(landmarks=landmarks, image_size=(width, height), visible_ratio=visible_ratio)

    if visible_ratio < cfg.pose_min_visible_ratio:
        raise NoPersonDetectedError(
            "The person in the photo is unclear. Use a brighter, front-facing "
            "photo where the upper body is fully in frame.",
            detail=f"visible_ratio={visible_ratio:.2f} < {cfg.pose_min_visible_ratio}",
        )

    if not (
        pose.is_visible(PoseLandmark.LEFT_SHOULDER, 0.3)
        or pose.is_visible(PoseLandmark.RIGHT_SHOULDER, 0.3)
    ):
        raise NoPersonDetectedError(
            "Could not find the shoulders. Make sure the upper body is visible.",
            detail="no shoulder landmark above 0.3 visibility",
        )

    logger.info(
        "Pose detected: framing=%s visible=%.0f%% shoulder=%s",
        pose.framing,
        visible_ratio * 100,
        pose.dominant_shoulder,
    )
    return pose


def require_framing(pose: PoseResult, category: Category) -> None:
    """Enforce the minimum body extent a category needs.

    A saree or lehenga is defined by its full-length drape. Generating one from
    a headshot produces a hallucinated lower body that will not match the
    customer, so we refuse rather than disappoint.

    Raises
    ------
    PersonPartiallyVisibleError
    """
    if category.is_draped and pose.framing == "half":
        raise PersonPartiallyVisibleError(
            f"A {category.label.lower()} needs a photo showing the person from "
            "head to at least the knees. Please upload a full-length photo.",
            detail=f"framing={pose.framing} category={category.value}",
        )


# --------------------------------------------------------------------------- #
# Human parsing
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _load_parser() -> tuple[Any, Any] | None:
    """Load and cache the SegFormer human-parsing model.

    Returns
    -------
    tuple or None
        ``(processor, model)``, or ``None`` when transformers or the weights
        are unavailable - callers then fall back to a pose-derived mask.
    """
    try:
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        logger.warning("transformers unavailable; human parsing disabled.")
        return None

    try:
        processor = SegformerImageProcessor.from_pretrained(HUMAN_PARSING_REPO)
        model = AutoModelForSemanticSegmentation.from_pretrained(HUMAN_PARSING_REPO)
    except Exception as exc:  # noqa: BLE001 - network/hub failures are varied
        logger.warning("Could not load human parsing model %s: %s", HUMAN_PARSING_REPO, exc)
        return None

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    logger.info("Loaded human parsing model %s", HUMAN_PARSING_REPO)
    return processor, model


def parse_human(image: Image.Image) -> ParseResult:
    """Produce a per-pixel garment/body label map.

    Falls back to an empty map (``available=False``) if the parsing model
    cannot be loaded, so the app degrades instead of crashing.
    """
    loaded = _load_parser()
    width, height = image.size

    if loaded is None:
        return ParseResult(label_map=np.zeros((height, width), dtype=np.uint8), available=False)

    processor, model = loaded
    import torch

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=(height, width), mode="bilinear", align_corners=False
    )
    label_map = upsampled.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)

    coverage = float((label_map != int(AtrLabel.BACKGROUND)).mean())
    logger.info("Human parsing done: person covers %.1f%% of frame", coverage * 100)
    return ParseResult(label_map=label_map, available=True)


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #


def build_inpaint_mask(
    parse: ParseResult,
    pose: PoseResult,
    category: Category,
    cfg: PreprocessConfig | None = None,
) -> Image.Image:
    """Build the region the model is allowed to repaint.

    The mask is the union of the existing-garment labels for ``category``,
    dilated to cover garment edges, minus the identity region (face, hair) so
    the customer's likeness survives. It is then feathered, which avoids the
    hard seams that give cheap try-on demos away.

    Parameters
    ----------
    parse:
        Human parsing output. When unavailable, a pose-derived box is used.
    pose:
        Body geometry, used for the fallback and to extend draped masks.
    category:
        Target garment category.
    cfg:
        Preprocessing thresholds.

    Returns
    -------
    Image.Image
        8-bit ``L`` mask: 255 = repaint, 0 = keep.
    """
    cfg = cfg or CONFIG.preprocess
    height, width = parse.label_map.shape

    if parse.available:
        labels = _CATEGORY_GARMENT_LABELS[category]
        mask = parse.mask_for_labels(sorted(labels))
        if category.is_draped:
            mask |= _draped_extension_mask(pose, (width, height))
    else:
        logger.warning("Human parsing unavailable; using pose-derived fallback mask.")
        mask = _pose_fallback_mask(pose, (width, height), category)

    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    if cfg.mask_dilate_px > 0:
        # MaxFilter's kernel must be odd; size 2r+1 dilates by r pixels.
        mask_image = mask_image.filter(ImageFilter.MaxFilter(cfg.mask_dilate_px * 2 + 1))

    if parse.available:
        identity = parse.mask_for_labels(sorted(IDENTITY_LABELS))
        if identity.any():
            keep = Image.fromarray(((~identity) * 255).astype(np.uint8), mode="L")
            mask_image = Image.fromarray(
                np.minimum(np.asarray(mask_image), np.asarray(keep)).astype(np.uint8), mode="L"
            )

    if cfg.mask_feather_sigma > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(cfg.mask_feather_sigma))

    coverage = float(np.asarray(mask_image).mean() / 255.0)
    logger.info("Inpaint mask for %s covers %.1f%% of frame", category.value, coverage * 100)
    return mask_image


def _draped_extension_mask(pose: PoseResult, size: tuple[int, int]) -> np.ndarray:
    """Extend a mask downward from the hips for sarees and lehengas.

    Human parsing labels only the *existing* clothing. A person wearing jeans
    has no pixels labelled ``SKIRT``, so a saree generated from that mask has
    nowhere to put its lower drape. This adds a trapezoid from the hips to the
    bottom of the frame, flaring outward to allow for pleats and a lehenga's
    volume.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=bool)

    left_hip = pose.get(PoseLandmark.LEFT_HIP)
    right_hip = pose.get(PoseLandmark.RIGHT_HIP)
    if left_hip is None or right_hip is None:
        return mask

    hip_y = int(min(left_hip[1], right_hip[1]))
    hip_left = int(min(left_hip[0], right_hip[0]))
    hip_right = int(max(left_hip[0], right_hip[0]))
    hip_width = max(1, hip_right - hip_left)

    # Flare from 1.4x hip width at the waist to 2.6x at the hem.
    for y in range(max(0, hip_y), height):
        progress = (y - hip_y) / max(1, height - hip_y)
        half_width = hip_width * (1.4 + 1.2 * progress) / 2.0
        centre = (hip_left + hip_right) / 2.0
        x0 = max(0, int(centre - half_width))
        x1 = min(width, int(centre + half_width))
        mask[y, x0:x1] = True

    return mask


def _pose_fallback_mask(
    pose: PoseResult, size: tuple[int, int], category: Category
) -> np.ndarray:
    """Approximate a garment mask from pose alone.

    Only used when the parsing model is unavailable. Coarse, but better than
    refusing to run.
    """
    width, height = size
    mask = np.zeros((height, width), dtype=bool)

    torso = pose.torso_box()
    if torso is None:
        return mask

    left, top, right, bottom = torso
    pad_x = int((right - left) * 0.35)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - int((bottom - top) * 0.15))

    if category.is_draped or category is Category.DRESS:
        bottom = height
    else:
        bottom = min(height, bottom + int((bottom - top) * 0.25))

    mask[top:bottom, left:right] = True
    return mask


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def prepare_person(
    source: str | Path | Image.Image | np.ndarray,
    category: Category,
    *,
    target_size: tuple[int, int] | None = None,
    cfg: PreprocessConfig | None = None,
) -> PersonAssets:
    """Run the full person preprocessing pipeline.

    Parameters
    ----------
    source:
        Uploaded person photo (path, PIL image or numpy array).
    category:
        Target garment category - determines the mask and framing rules.
    target_size:
        Working resolution ``(width, height)``. Defaults to the runtime config.
    cfg:
        Preprocessing thresholds.

    Returns
    -------
    PersonAssets

    Raises
    ------
    MissingImageError, ImageTooSmallError, NoPersonDetectedError,
    PersonPartiallyVisibleError
    """
    cfg = cfg or CONFIG.preprocess
    target_size = target_size or CONFIG.runtime.size

    original = load_image(source, label="photo of the person")
    assert_min_size(original, cfg.min_person_short_edge, label="photo of the person")

    image, info = letterbox(original, target_size, fill=(255, 255, 255))
    logger.info(
        "Person photo %sx%s -> %sx%s for category=%s",
        *original.size,
        *target_size,
        category.value,
    )

    pose = detect_pose(image, cfg)
    require_framing(pose, category)

    parse = parse_human(image)
    inpaint_mask = build_inpaint_mask(parse, pose, category, cfg)

    return PersonAssets(
        image=image,
        original=original,
        letterbox_info=info,
        pose=pose,
        parse=parse,
        inpaint_mask=inpaint_mask,
        category=category,
    )
