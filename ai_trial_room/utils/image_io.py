"""Image loading, geometry helpers and privacy-safe temporary storage.

All functions are pure with respect to their inputs: they return new
:class:`PIL.Image.Image` objects and never mutate an argument in place.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

from ai_trial_room.utils.errors import ImageTooSmallError, MissingImageError
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Diffusion VAEs need both edges divisible by this.
LATENT_MULTIPLE: int = 16


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def load_image(source: str | Path | Image.Image | np.ndarray, *, label: str = "image") -> Image.Image:
    """Coerce any supported input into an RGB :class:`~PIL.Image.Image`.

    EXIF orientation is applied, so photos taken on a phone are upright.

    Parameters
    ----------
    source:
        A path, a PIL image, or an HxWx3 / HxWx4 numpy array.
    label:
        Name used in error messages, e.g. ``"person photo"``.

    Raises
    ------
    MissingImageError
        If ``source`` is ``None``.
    """
    if source is None:
        raise MissingImageError(f"Please upload a {label}.")

    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source.astype(np.uint8))
    else:
        path = Path(source)
        if not path.exists():
            raise MissingImageError(f"Could not read the {label}.", detail=str(path))
        image = Image.open(path)

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_rgba(source: str | Path | Image.Image | np.ndarray, *, label: str = "image") -> Image.Image:
    """Like :func:`load_image` but preserves an alpha channel."""
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGBA")
    return load_image(source, label=label).convert("RGBA")


def assert_min_size(image: Image.Image, min_short_edge: int, *, label: str) -> None:
    """Reject images whose short edge is below ``min_short_edge``.

    Raises
    ------
    ImageTooSmallError
    """
    short_edge = min(image.size)
    if short_edge < min_short_edge:
        raise ImageTooSmallError(
            f"The {label} is too small ({image.width}x{image.height}). "
            f"Please use an image at least {min_short_edge}px on its shorter side.",
            detail=f"short_edge={short_edge} < {min_short_edge}",
        )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LetterboxInfo:
    """Records how an image was letterboxed, so the result can be undone.

    Attributes
    ----------
    original_size:
        ``(width, height)`` before any resizing.
    scaled_size:
        ``(width, height)`` of the content after aspect-preserving resize.
    offset:
        ``(left, top)`` position of the content inside the padded canvas.
    canvas_size:
        ``(width, height)`` of the final padded canvas.
    """

    original_size: tuple[int, int]
    scaled_size: tuple[int, int]
    offset: tuple[int, int]
    canvas_size: tuple[int, int]

    @property
    def scale(self) -> float:
        """Uniform scale factor applied to the original image."""
        return self.scaled_size[0] / self.original_size[0]

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """Content region inside the canvas as ``(left, top, right, bottom)``."""
        left, top = self.offset
        return left, top, left + self.scaled_size[0], top + self.scaled_size[1]


def round_to_multiple(value: int, multiple: int = LATENT_MULTIPLE) -> int:
    """Round ``value`` up to the nearest positive multiple of ``multiple``."""
    if value <= 0:
        return multiple
    return int(np.ceil(value / multiple) * multiple)


def letterbox(
    image: Image.Image,
    target: tuple[int, int],
    *,
    fill: tuple[int, int, int] | tuple[int, int, int, int] = (255, 255, 255),
    resample: int = Image.Resampling.LANCZOS,
) -> tuple[Image.Image, LetterboxInfo]:
    """Resize preserving aspect ratio, then pad to exactly ``target``.

    Padding rather than cropping matters here: cropping a full-body photo to a
    fixed aspect ratio would cut off the hem of a saree or lehenga.

    Parameters
    ----------
    image:
        Source image.
    target:
        Desired ``(width, height)``.
    fill:
        Padding colour. Must match the image's band count.
    resample:
        PIL resampling filter.

    Returns
    -------
    tuple
        The padded image and a :class:`LetterboxInfo` describing the transform.
    """
    target_w, target_h = target
    src_w, src_h = image.size

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), resample)

    if len(fill) != len(resized.getbands()):
        fill = fill[: len(resized.getbands())] if len(fill) > len(resized.getbands()) else fill * 2
        fill = tuple(fill[: len(resized.getbands())])  # type: ignore[assignment]

    canvas = Image.new(resized.mode, (target_w, target_h), fill)  # type: ignore[arg-type]
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)

    info = LetterboxInfo(
        original_size=(src_w, src_h),
        scaled_size=(new_w, new_h),
        offset=offset,
        canvas_size=(target_w, target_h),
    )
    logger.debug("Letterboxed %sx%s -> %sx%s at offset %s", src_w, src_h, new_w, new_h, offset)
    return canvas, info


def unletterbox(image: Image.Image, info: LetterboxInfo) -> Image.Image:
    """Invert :func:`letterbox`, returning the image at its original size.

    Used so the generated try-on is handed back to the user at the same
    dimensions as the photo they uploaded.
    """
    cropped = image.crop(info.content_box)
    if cropped.size == info.original_size:
        return cropped
    return cropped.resize(info.original_size, Image.Resampling.LANCZOS)


def trim_alpha(image: Image.Image, *, margin: int = 0, alpha_threshold: int = 8) -> Image.Image:
    """Crop an RGBA image to its opaque content, keeping ``margin`` px around it.

    Parameters
    ----------
    image:
        RGBA image, typically the output of background removal.
    margin:
        Transparent padding to retain on every side.
    alpha_threshold:
        Alpha values at or below this count as fully transparent.

    Returns
    -------
    Image.Image
        The cropped RGBA image, or the input unchanged if it is fully
        transparent.
    """
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    opaque = alpha > alpha_threshold
    if not opaque.any():
        return rgba

    rows = np.flatnonzero(opaque.any(axis=1))
    cols = np.flatnonzero(opaque.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(rgba.width, right + margin)
    bottom = min(rgba.height, bottom + margin)
    return rgba.crop((left, top, right, bottom))


def flatten_on_color(
    image: Image.Image, color: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Composite an RGBA image onto a solid background and return RGB.

    Reference-editing backends take RGB input, and a neutral white backdrop
    keeps the model's attention on the garment rather than on checkerboard
    transparency artefacts.
    """
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*color, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def make_side_by_side(
    left: Image.Image,
    right: Image.Image,
    *,
    gap: int = 12,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Compose two images into a single before/after strip of equal height."""
    height = max(left.height, right.height)

    def _scaled(img: Image.Image) -> Image.Image:
        if img.height == height:
            return img.convert("RGB")
        width = max(1, int(round(img.width * height / img.height)))
        return img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    a, b = _scaled(left), _scaled(right)
    canvas = Image.new("RGB", (a.width + gap + b.width, height), background)
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas


# --------------------------------------------------------------------------- #
# Privacy-safe temporary storage
# --------------------------------------------------------------------------- #


@contextmanager
def ephemeral_dir(prefix: str = "aitr-") -> Iterator[Path]:
    """Yield a temp directory that is force-deleted on exit.

    Uploaded photos are biometric data. Use this for any intermediate file that
    is consumed entirely within one request - even if the request raises.

    .. note::

       This is *not* suitable for a file the browser must download later: the
       directory is gone the moment the block exits, which is before the user
       clicks the download button. Use :func:`download_cache_dir` for those.

    Yields
    ------
    Path
        An existing, writable, short-lived directory.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        logger.debug("Purged ephemeral dir %s", path)


#: Process-lifetime directory holding files the browser still needs to fetch.
_download_dir: Path | None = None

#: Download files older than this are purged on the next request.
DOWNLOAD_TTL_SECONDS: float = 900.0


def download_cache_dir() -> Path:
    """Return the temp directory used for browser-downloadable results.

    A download cannot live in an :func:`ephemeral_dir`, because Gradio hands the
    browser a path that is fetched *after* the handler returns. So results go
    here instead: a temp directory created once per process, removed at exit,
    and swept by :func:`purge_download_cache` on every request.

    That still satisfies "photos are not stored" - nothing lands in the project
    tree, nothing survives the process, and files are deleted within
    :data:`DOWNLOAD_TTL_SECONDS` - while keeping the download button working.

    Returns
    -------
    Path
        An existing, writable directory.
    """
    global _download_dir

    if _download_dir is None or not _download_dir.exists():
        _download_dir = Path(tempfile.mkdtemp(prefix="aitr-downloads-"))
        atexit.register(shutil.rmtree, _download_dir, True)
        logger.debug("Created download cache %s", _download_dir)

    return _download_dir


def purge_download_cache(max_age_s: float = DOWNLOAD_TTL_SECONDS) -> int:
    """Delete cached download files older than ``max_age_s``.

    Called at the start of each generation, so a result lives only as long as
    it plausibly needs to and a long demo session cannot accumulate photos.

    Parameters
    ----------
    max_age_s:
        Age threshold in seconds.

    Returns
    -------
    int
        Number of files deleted.
    """
    if _download_dir is None or not _download_dir.exists():
        return 0

    cutoff = time.time() - max_age_s
    removed = 0
    for path in _download_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:  # pragma: no cover - racing with another sweep
            continue

    if removed:
        logger.debug("Purged %d expired download(s)", removed)
    return removed


def scrub_metadata(image: Image.Image) -> Image.Image:
    """Return a copy with all EXIF/ICC metadata dropped.

    Phone photos embed GPS coordinates and device IDs. Stripping them means a
    downloaded result cannot leak where the photo was taken.
    """
    clean = Image.new(image.mode, image.size)
    clean.putdata(list(image.getdata()))
    return clean
