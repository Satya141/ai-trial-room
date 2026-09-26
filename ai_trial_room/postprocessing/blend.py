"""Colour and lighting harmonization between the generated image and the source.

Reference-editing models tend to return output that is subtly brighter and
more saturated than the input photo - they have a prior toward well-lit studio
imagery. In a shop, photos are taken under warm tube lights or on a phone near
a window, and a result that does not match that lighting reads as fake even
when the garment is perfect.

:func:`harmonize` corrects this by matching the generated image's global
statistics to the source in LAB space, where lightness is separated from
colour, so exposure and white balance can be corrected without touching hue
relationships within the garment.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: How strongly to pull the output toward the source statistics. 1.0 is a full
#: match, which can wash out a deliberately vivid garment; 0.6 keeps the
#: garment's character while fixing exposure and cast.
DEFAULT_STRENGTH: float = 0.6

#: Clamp per-channel gain so a near-monochrome source cannot blow up contrast.
_MAX_GAIN: float = 1.6
_MIN_GAIN: float = 0.6


def _to_lab(image: Image.Image) -> np.ndarray:
    """Convert an RGB image to a float32 LAB array.

    Uses PIL's built-in ``LAB`` conversion, which keeps this dependency-free.
    Channel ranges are 0-255 per PIL's convention, not CIE units - fine here
    because we only ever compare and rescale, never interpret absolutely.
    """
    return np.asarray(image.convert("RGB").convert("LAB"), dtype=np.float32)


def _from_lab(array: np.ndarray) -> Image.Image:
    """Convert a float32 LAB array back to an RGB image."""
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="LAB").convert("RGB")


def match_statistics(
    generated: Image.Image,
    reference: Image.Image,
    *,
    strength: float = DEFAULT_STRENGTH,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Match ``generated``'s LAB mean and standard deviation to ``reference``.

    Parameters
    ----------
    generated:
        Model output to correct.
    reference:
        Source photo whose lighting should be matched.
    strength:
        0.0 leaves ``generated`` untouched, 1.0 matches fully.
    mask:
        Optional ``L`` mask selecting which pixels of ``reference`` define the
        target statistics. Passing the inverse of the garment mask makes the
        match ignore the new garment and key off skin and background instead,
        which is usually what you want.

    Returns
    -------
    Image.Image
        The corrected image.
    """
    if strength <= 0.0:
        return generated

    source = _to_lab(generated)
    target = _to_lab(reference)

    if mask is not None:
        weights = np.asarray(mask.convert("L").resize(reference.size), dtype=np.float32) / 255.0
        if weights.sum() < 64:  # too few pixels to be meaningful
            weights = None
    else:
        weights = None

    corrected = source.copy()
    for channel in range(3):
        src_channel = source[..., channel]
        tgt_channel = target[..., channel]

        if weights is not None:
            total = weights.sum()
            tgt_mean = float((tgt_channel * weights).sum() / total)
            tgt_std = float(
                np.sqrt(((tgt_channel - tgt_mean) ** 2 * weights).sum() / total)
            )
        else:
            tgt_mean = float(tgt_channel.mean())
            tgt_std = float(tgt_channel.std())

        src_mean = float(src_channel.mean())
        src_std = float(src_channel.std())

        gain = 1.0 if src_std < 1e-3 else float(np.clip(tgt_std / src_std, _MIN_GAIN, _MAX_GAIN))
        matched = (src_channel - src_mean) * gain + tgt_mean
        corrected[..., channel] = src_channel + (matched - src_channel) * strength

    return _from_lab(corrected)


def harmonize(
    generated: Image.Image,
    reference: Image.Image,
    *,
    strength: float = DEFAULT_STRENGTH,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Match colour and lighting, returning ``generated`` on any failure.

    Thin, non-raising wrapper around :func:`match_statistics`: colour matching
    is a polish step and must never cost the user their result.
    """
    try:
        if generated.size != reference.size:
            reference = reference.resize(generated.size, Image.Resampling.LANCZOS)
        result = match_statistics(generated, reference, strength=strength, mask=mask)
        logger.debug("Harmonized colours at strength %.2f", strength)
        return result
    except Exception as exc:  # noqa: BLE001 - polish must not break the pipeline
        logger.warning("Colour harmonization failed (%s); returning raw output.", exc)
        return generated


def feather_composite(
    foreground: Image.Image,
    background: Image.Image,
    mask: Image.Image,
    *,
    sigma: float = 3.0,
) -> Image.Image:
    """Composite ``foreground`` over ``background`` through a blurred ``mask``.

    Parameters
    ----------
    foreground:
        Image to place on top.
    background:
        Image underneath.
    mask:
        ``L`` mask; 255 selects ``foreground``.
    sigma:
        Gaussian blur applied to the mask before compositing, which removes the
        stair-stepping you get from a hard segmentation edge.

    Returns
    -------
    Image.Image
    """
    if mask.size != background.size:
        mask = mask.resize(background.size, Image.Resampling.LANCZOS)
    if foreground.size != background.size:
        foreground = foreground.resize(background.size, Image.Resampling.LANCZOS)

    soft = mask.convert("L").filter(ImageFilter.GaussianBlur(sigma))
    return Image.composite(foreground.convert("RGB"), background.convert("RGB"), soft)


def laplacian_blend(
    foreground: Image.Image,
    background: Image.Image,
    mask: Image.Image,
    *,
    levels: int = 5,
) -> Image.Image:
    """Blend two images through ``mask`` using a Laplacian pyramid.

    Why this beats a feathered alpha composite
    ------------------------------------------
    A straight composite blends every spatial frequency at the same rate, so a
    wide feather needed to hide the seam also cross-fades facial detail, and a
    narrow feather leaves a visible patch edge whenever the two images differ in
    brightness. That is exactly the case here: the generated image has been
    relit by the model, so the original face is a different exposure.

    A Laplacian blend decomposes both images by frequency and blends each band
    with a correspondingly blurred mask. Low frequencies (lighting, colour)
    cross-fade gradually so the transition is invisible; high frequencies
    (pores, eyelashes, hair) switch sharply so the face stays crisp.

    Parameters
    ----------
    foreground:
        Image taken where ``mask`` is white - the original face.
    background:
        Image taken where ``mask`` is black - the generated try-on.
    mask:
        ``L`` mask.
    levels:
        Pyramid depth. Each level halves resolution; 5 suits faces from roughly
        64 to 1024 px wide. Clamped so the smallest level stays at least 8 px.

    Returns
    -------
    Image.Image
        The blended RGB image.
    """
    size = background.size
    if foreground.size != size:
        foreground = foreground.resize(size, Image.Resampling.LANCZOS)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.LANCZOS)

    # Keep the coarsest level usable: halving below ~8 px adds nothing.
    max_levels = max(1, int(np.floor(np.log2(max(1, min(size)) / 8.0))))
    levels = max(1, min(levels, max_levels))

    fg = np.asarray(foreground.convert("RGB"), dtype=np.float32)
    bg = np.asarray(background.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0

    fg_pyramid = _gaussian_pyramid(fg, levels)
    bg_pyramid = _gaussian_pyramid(bg, levels)
    alpha_pyramid = _gaussian_pyramid(alpha, levels)

    # Start from the blended coarsest level, then add back each detail band.
    result = (
        fg_pyramid[-1] * alpha_pyramid[-1] + bg_pyramid[-1] * (1.0 - alpha_pyramid[-1])
    )

    for level in range(levels - 2, -1, -1):
        target_shape = fg_pyramid[level].shape[:2]
        result = _upsample(result, target_shape)

        fg_detail = fg_pyramid[level] - _upsample(fg_pyramid[level + 1], target_shape)
        bg_detail = bg_pyramid[level] - _upsample(bg_pyramid[level + 1], target_shape)
        band_alpha = alpha_pyramid[level]

        result = result + fg_detail * band_alpha + bg_detail * (1.0 - band_alpha)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _gaussian_pyramid(array: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Gaussian pyramid, coarsest level last."""
    pyramid = [array]
    for _ in range(levels - 1):
        pyramid.append(_downsample(pyramid[-1]))
    return pyramid


def _blur5(array: np.ndarray) -> np.ndarray:
    """Apply a separable 5-tap binomial blur with edge replication.

    The classic ``[1 4 6 4 1] / 16`` kernel used for Gaussian pyramids. Applied
    as two 1-D passes, which is why this stays fast in pure numpy.
    """
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    out = array

    for axis in (0, 1):
        padded = np.pad(
            out,
            [(2, 2) if index == axis else (0, 0) for index in range(out.ndim)],
            mode="edge",
        )
        shape = [slice(None)] * out.ndim
        accumulated = np.zeros_like(out)
        for tap, weight in enumerate(kernel):
            shape[axis] = slice(tap, tap + out.shape[axis])
            accumulated = accumulated + padded[tuple(shape)] * weight
        out = accumulated

    return out


def _downsample(array: np.ndarray) -> np.ndarray:
    """Blur then take every second pixel, halving both spatial dimensions."""
    return _blur5(array)[::2, ::2]


def _upsample(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample to ``shape``, then blur to smooth it.

    Nearest-neighbour plus a blur is equivalent in effect to the usual
    zero-insert-and-blur expand step, and avoids an interpolation dependency.
    """
    height, width = shape
    row_index = np.minimum((np.arange(height) // 2), array.shape[0] - 1)
    col_index = np.minimum((np.arange(width) // 2), array.shape[1] - 1)
    expanded = array[row_index][:, col_index]
    return _blur5(expanded)


def match_region_tone(
    source: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    *,
    strength: float = 0.7,
) -> Image.Image:
    """Shift ``source``'s colour inside ``mask`` toward ``target``'s in that area.

    Used before face blending: if the model relit the person warmly, the original
    face needs the same warm cast or it reads as a cut-out sticker. Only the mean
    per-channel offset is transferred, not the variance, so the face keeps its own
    contrast and texture.

    Parameters
    ----------
    source:
        Image to correct - the original photo.
    target:
        Image whose local colour should be matched - the generated output.
    mask:
        ``L`` mask defining the region whose statistics are compared.
    strength:
        0.0 leaves ``source`` untouched, 1.0 fully adopts ``target``'s mean.

    Returns
    -------
    Image.Image
        The corrected RGB image.
    """
    if strength <= 0.0:
        return source

    size = source.size
    if target.size != size:
        target = target.resize(size, Image.Resampling.LANCZOS)
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.LANCZOS)

    weights = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    total = float(weights.sum())
    if total < 64.0:  # region too small for a meaningful statistic
        return source

    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    tgt = np.asarray(target.convert("RGB"), dtype=np.float32)
    weights3 = weights[..., None]

    src_mean = (src * weights3).sum(axis=(0, 1)) / total
    tgt_mean = (tgt * weights3).sum(axis=(0, 1)) / total

    # Clamp the shift: a large offset means the model changed the person, and
    # chasing it would tint the face wrongly.
    offset = np.clip((tgt_mean - src_mean) * strength, -28.0, 28.0)
    corrected = src + offset * weights3

    logger.debug("Region tone offset %s (strength %.2f)", np.round(offset, 1), strength)
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8), mode="RGB")


def sharpen_garment_region(
    image: Image.Image, mask: Image.Image, *, amount: float = 0.35
) -> Image.Image:
    """Apply unsharp masking inside ``mask`` only.

    Diffusion output is often slightly soft on fine detail - exactly where zari
    thread, mirror work and block-print edges live. Sharpening only the garment
    region avoids adding noise to skin, which would look worse.

    Parameters
    ----------
    image:
        Image to sharpen.
    mask:
        ``L`` mask of the garment region.
    amount:
        0.0 disables; values above ~0.6 start to look crunchy.

    Returns
    -------
    Image.Image
    """
    if amount <= 0.0:
        return image
    try:
        sharpened = image.convert("RGB").filter(
            ImageFilter.UnsharpMask(radius=2, percent=int(amount * 150), threshold=3)
        )
        return feather_composite(sharpened, image, mask, sigma=2.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Garment sharpening failed (%s); skipping.", exc)
        return image
