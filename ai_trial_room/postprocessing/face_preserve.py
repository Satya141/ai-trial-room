"""Identity preservation by blending the original face back into the output.

The problem
-----------
Diffusion editing regenerates every pixel it touches. Even with an explicit
"keep the face unchanged" instruction, a 12 B transformer drifts: eye spacing
shifts a few pixels, the jawline softens, skin texture smooths. Customers
notice immediately - "that's a nice saree but it isn't me" - and for a shop
deployment that is the difference between a tool they use and one they don't.

The fix
-------
Composite the *original* face pixels over the generated image through a
feathered mask derived from face landmarks. Because the pose, camera and
framing are identical (we letterboxed once and never moved the person), the
face lands in the same place, and a wide feather hides the seam.

Guard rails
-----------
Blending is skipped, with a warning, when:

* no face is found in either image (nothing to blend);
* the two face centroids are more than
  :data:`MAX_CENTROID_DRIFT_RATIO` of the face width apart, which means the
  model moved the head and a naive paste would produce a two-headed artefact.

Licensing note: this uses MediaPipe FaceMesh (Apache-2.0) rather than
InsightFace, whose models are licensed for non-commercial research only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ai_trial_room.config import CONFIG, IdentityConfig
from ai_trial_room.postprocessing.blend import laplacian_blend, match_region_tone
from ai_trial_room.preprocessing.person import PersonAssets
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Reject the blend if the face centroid moved further than this fraction of
#: the detected face width between input and output.
MAX_CENTROID_DRIFT_RATIO: float = 0.35

#: Feather radius as a fraction of face width. Generous on purpose - a hard
#: edge around a pasted face is the classic giveaway of a cheap try-on demo.
FEATHER_RATIO: float = 0.12

#: Grow the landmark hull by this fraction of face width before feathering, so
#: the mask covers the hairline and jaw rather than stopping at the landmarks.
HULL_EXPAND_RATIO: float = 0.10


@dataclass(frozen=True)
class FaceRegion:
    """A detected face as a polygon plus derived geometry."""

    polygon: list[tuple[float, float]]
    """Convex hull of the face landmarks, in absolute pixels."""

    image_size: tuple[int, int]
    """``(width, height)`` of the image the polygon refers to."""

    @property
    def centroid(self) -> tuple[float, float]:
        """Arithmetic centre of the hull."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @property
    def width(self) -> float:
        """Horizontal extent of the hull in pixels."""
        xs = [p[0] for p in self.polygon]
        return max(xs) - min(xs)

    @property
    def height(self) -> float:
        """Vertical extent of the hull in pixels."""
        ys = [p[1] for p in self.polygon]
        return max(ys) - min(ys)


def detect_face(image: Image.Image) -> FaceRegion | None:
    """Locate the most prominent face using MediaPipe FaceMesh.

    Parameters
    ----------
    image:
        RGB image to search.

    Returns
    -------
    FaceRegion or None
        ``None`` when MediaPipe is unavailable or no face is found.
    """
    try:
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe unavailable; cannot preserve face.")
        return None

    width, height = image.size
    array = np.asarray(image.convert("RGB"))

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.4,
    ) as mesh:
        result = mesh.process(array)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    points = [(lm.x * width, lm.y * height) for lm in landmarks]
    hull = _convex_hull(points)
    if len(hull) < 3:
        return None

    return FaceRegion(polygon=hull, image_size=(width, height))


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the convex hull of ``points`` using a monotone chain.

    Implemented locally to avoid pulling in scipy for twenty lines of geometry.

    Parameters
    ----------
    points:
        At least three ``(x, y)`` pairs.

    Returns
    -------
    list[tuple[float, float]]
        Hull vertices in counter-clockwise order.
    """
    unique = sorted(set(points))
    if len(unique) < 3:
        return list(unique)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _expand_polygon(
    polygon: Sequence[tuple[float, float]],
    amount: float,
    *,
    downward_factor: float = 0.35,
) -> list[tuple[float, float]]:
    """Scale a polygon outward from its centroid by ``amount`` pixels.

    A cheap approximation of polygon offsetting: adequate here because the face
    hull is roughly convex and roughly circular.

    Expansion is *anisotropic*. Growing upward and sideways picks up the hairline
    and jaw, which is wanted. Growing downward by the same amount would push the
    mask onto the neck and the blouse neckline - and pasting the original
    neckline back over a newly generated choli is a very visible failure. So
    downward growth is scaled by ``downward_factor``.

    Parameters
    ----------
    polygon:
        Hull vertices.
    amount:
        Outward offset in pixels.
    downward_factor:
        Multiplier applied to ``amount`` for vertices below the centroid.

    Returns
    -------
    list[tuple[float, float]]
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    expanded: list[tuple[float, float]] = []
    for x, y in polygon:
        dx, dy = x - cx, y - cy
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        scale = amount * (downward_factor if dy > 0 else 1.0)
        expanded.append((x + dx / length * scale, y + dy / length * scale))
    return expanded


def build_face_mask(
    region: FaceRegion,
    *,
    feather: float | None = None,
    strength: float = 1.0,
) -> Image.Image:
    """Render a feathered 8-bit mask over ``region``.

    Parameters
    ----------
    region:
        Detected face.
    feather:
        Blur sigma in pixels. Defaults to :data:`FEATHER_RATIO` of face width.
    strength:
        Peak mask opacity, 0-1. Values below 1 let some of the generated face
        through, which can look more natural under heavy relighting.

    Returns
    -------
    Image.Image
        ``L`` mask where 255 means "take the original pixel".
    """
    width, height = region.image_size
    sigma = feather if feather is not None else max(4.0, region.width * FEATHER_RATIO)

    polygon = _expand_polygon(region.polygon, region.width * HULL_EXPAND_RATIO)

    peak = int(round(255 * min(max(strength, 0.0), 1.0)))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in polygon], fill=peak)
    return mask.filter(ImageFilter.GaussianBlur(sigma))


def restore_face(
    generated: Image.Image,
    person: PersonAssets,
    *,
    config: IdentityConfig | None = None,
) -> Image.Image:
    """Blend the original face over ``generated``.

    Pipeline, each step guarded and skippable:

    1. Detect the face in both images.
    2. Abort if the head moved - see :data:`MAX_CENTROID_DRIFT_RATIO`.
    3. Optionally shift the original face's colour toward the generated frame's,
       so a relit result does not get a mismatched face pasted in.
    4. Blend through a feathered hull, using a Laplacian pyramid so lighting
       cross-fades smoothly while facial detail stays sharp.

    Parameters
    ----------
    generated:
        Model output, same size as ``person.image``.
    person:
        Preprocessed person assets; ``person.image`` supplies the source face.
    config:
        Identity settings. Defaults to :attr:`AppConfig.identity`.

    Returns
    -------
    Image.Image
        The blended image, or ``generated`` unchanged when blending would be
        unsafe. Never raises - identity preservation is an enhancement, and a
        failure here must not lose the user's result.
    """
    config = config or CONFIG.identity

    source = person.image
    if generated.size != source.size:
        logger.warning(
            "Size mismatch (%s vs %s); resizing generated image before face blend.",
            generated.size,
            source.size,
        )
        generated = generated.resize(source.size, Image.Resampling.LANCZOS)

    source_face = detect_face(source)
    if source_face is None:
        logger.info("No face found in the source photo; skipping face preservation.")
        return generated

    generated_face = detect_face(generated)
    if generated_face is None:
        logger.info("No face found in the generated image; skipping face preservation.")
        return generated

    sx, sy = source_face.centroid
    gx, gy = generated_face.centroid
    drift = ((sx - gx) ** 2 + (sy - gy) ** 2) ** 0.5
    allowed = source_face.width * config.max_drift_ratio

    if drift > allowed:
        logger.warning(
            "Face moved %.0f px (limit %.0f px); skipping blend to avoid artefacts.",
            drift,
            allowed,
        )
        return generated

    mask = build_face_mask(source_face, strength=config.strength)
    face_source = source.convert("RGB")

    if config.match_skin_tone:
        face_source = match_region_tone(face_source, generated, mask, strength=0.7)

    try:
        if config.laplacian:
            blended = laplacian_blend(
                face_source,
                generated.convert("RGB"),
                mask,
                levels=config.pyramid_levels,
            )
            method = f"laplacian x{config.pyramid_levels}"
        else:
            blended = Image.composite(face_source, generated.convert("RGB"), mask)
            method = "alpha"
    except Exception as exc:  # noqa: BLE001 - never lose the result over polish
        logger.warning("Face blend failed (%s); returning unblended output.", exc)
        return generated

    logger.info(
        "Face preserved via %s (drift %.1f px, strength %.2f, tone_match=%s).",
        method,
        drift,
        config.strength,
        config.match_skin_tone,
    )
    return blended
