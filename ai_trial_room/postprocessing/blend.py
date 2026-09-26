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
