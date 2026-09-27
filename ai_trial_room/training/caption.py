"""Caption generation for saree LoRA datasets.

Two strategies, and the cheap one is usually better here.

``template`` (default)
    Build the caption from the drape style in the filename plus the project's own
    garment vocabulary. Free, instant, and - importantly - it uses *exactly* the
    same words as the inference prompts in
    :mod:`ai_trial_room.backends.prompts`. That alignment is the point: a LoRA
    learns to associate a visual pattern with the tokens it was captioned with,
    so captions that drift from the inference vocabulary train a trigger you
    never pull.

``vlm``
    Caption each image with Qwen2.5-VL (Apache-2.0), then prepend the drape
    vocabulary. Captures per-image specifics a template cannot - colour, border
    style, fabric, pose - at the cost of a 3-7B model download and a few seconds
    per image. Worth it above ~100 images, where template captions become
    near-identical and the LoRA has nothing to discriminate on.

A trigger token
---------------
Every caption starts with a rare trigger string (default ``aitrsaree``) so the
learned behaviour can be invoked deliberately at inference rather than always
being on. :func:`ai_trial_room.lora.apply_trigger` inserts the same token into
generation prompts.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Sequence

from PIL import Image

from ai_trial_room.backends.prompts import DRAPE_SPECS
from ai_trial_room.config import Category, DrapeStyle
from ai_trial_room.training.dataset import DrapeSample, parse_drape_from_name
from ai_trial_room.utils.image_io import load_image
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Rare token prepended to every caption, used as the LoRA trigger.
DEFAULT_TRIGGER: Final[str] = "aitrsaree"

#: Vision-language model used by the ``vlm`` strategy. Apache-2.0.
DEFAULT_VLM_REPO: Final[str] = "Qwen/Qwen2.5-VL-3B-Instruct"

#: Instruction given to the VLM. Deliberately narrow - we want garment
#: description, not a story about the person, and never identity details.
VLM_INSTRUCTION: Final[str] = (
    "Describe only the clothing in this photograph in one sentence. Name the "
    "garment type, its main colour, the border or edging, the print or "
    "embroidery style, and the fabric. Do not describe the person's face, "
    "identity, age or expression. Do not describe the background."
)


class CaptionStrategy(str, Enum):
    """How to produce captions."""

    TEMPLATE = "template"
    VLM = "vlm"

    @property
    def description(self) -> str:
        """One-line explanation for CLI help."""
        return _STRATEGY_DESCRIPTIONS[self]


_STRATEGY_DESCRIPTIONS: Final[dict[CaptionStrategy, str]] = {
    CaptionStrategy.TEMPLATE: "filename drape style + project vocabulary (free, instant)",
    CaptionStrategy.VLM: "Qwen2.5-VL description + drape vocabulary (needs a GPU)",
}


@dataclass
class CaptionResult:
    """Outcome of captioning one dataset."""

    written: list[Path]
    skipped: list[Path]
    failed: list[tuple[Path, str]]
    untagged: list[Path]
    """Images whose filename prefix did not name a known drape style."""

    def render(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [
            f"Captions written: {len(self.written)}",
            f"Skipped (exists): {len(self.skipped)}",
        ]
        if self.untagged:
            shown = ", ".join(p.name for p in self.untagged[:5])
            lines.append(
                f"No drape style in filename for {len(self.untagged)} image(s): {shown}. "
                "Rename them like nivi_001.jpg so the caption names the drape."
            )
        for path, reason in self.failed:
            lines.append(f"  FAILED {path.name}: {reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Template captions
# --------------------------------------------------------------------------- #


def template_caption(
    drape_style: DrapeStyle | None,
    *,
    category: Category = Category.SAREE,
    trigger: str = DEFAULT_TRIGGER,
    detail: str = "",
) -> str:
    """Build a caption from the drape vocabulary.

    The wording mirrors :mod:`ai_trial_room.backends.prompts` on purpose, so the
    tokens the LoRA is trained on are the tokens inference will emit.

    Parameters
    ----------
    drape_style:
        Drape to describe. ``None`` produces a generic saree caption.
    category:
        Garment category; only saree and lehenga have drape vocabulary.
    trigger:
        Rare token prepended to the caption.
    detail:
        Optional per-image description (e.g. from a VLM) inserted after the
        drape description.

    Returns
    -------
    str

    Examples
    --------
    >>> caption = template_caption(DrapeStyle.GUJARATI)
    >>> "seedha" in caption.lower() or "Gujarati" in caption
    True
    """
    parts: list[str] = [trigger] if trigger else []

    if category is Category.SAREE and drape_style is not None:
        spec = DRAPE_SPECS[drape_style]
        parts.append(
            f"a woman wearing a saree draped in the traditional {spec.name} style "
            f"of {spec.region}, with {spec.description}, and a matching fitted "
            f"blouse, the pallu over the {spec.shoulder} shoulder"
        )
    elif category is Category.LEHENGA:
        parts.append(
            "a woman wearing a lehenga: a floor-length flared ghagra skirt, a "
            "fitted choli blouse, and a draped dupatta"
        )
    else:
        parts.append(f"a person wearing a {category.label.lower()}")

    if detail:
        parts.append(detail.rstrip(". "))

    parts.append("full body photograph, natural lighting, realistic fabric folds")
    return ", ".join(parts) + "."


# --------------------------------------------------------------------------- #
# VLM captions
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _load_vlm(repo_id: str) -> tuple[Any, Any] | None:
    """Load and cache the captioning VLM.

    Returns
    -------
    tuple or None
        ``(processor, model)``, or ``None`` when transformers or the weights are
        unavailable - the caller then falls back to template captions.
    """
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError:
        logger.warning("transformers unavailable; cannot use the VLM strategy.")
        return None

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VlmClass
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as VlmClass  # type: ignore[assignment]
        except ImportError:
            logger.warning("No vision-language model class found in transformers.")
            return None

    try:
        processor = AutoProcessor.from_pretrained(repo_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = VlmClass.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except Exception as exc:  # noqa: BLE001 - hub/network/format failures vary
        logger.warning("Could not load VLM %s: %s", repo_id, exc)
        return None

    model.eval()
    logger.info("Loaded captioning VLM %s", repo_id)
    return processor, model


def vlm_describe(
    image: Image.Image,
    *,
    repo_id: str = DEFAULT_VLM_REPO,
    max_new_tokens: int = 72,
) -> str:
    """Describe an image's clothing with a vision-language model.

    Parameters
    ----------
    image:
        Image to describe.
    repo_id:
        VLM to use.
    max_new_tokens:
        Generation cap. Captions longer than ~70 tokens start to ramble, and a
        rambling caption dilutes the trigger.

    Returns
    -------
    str
        The description, or ``""`` when the model is unavailable or errors.
    """
    loaded = _load_vlm(repo_id)
    if loaded is None:
        return ""

    processor, model = loaded
    try:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VLM_INSTRUCTION},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Strip the prompt tokens so only the completion is decoded.
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return " ".join(caption.split())
    except Exception as exc:  # noqa: BLE001 - captioning is best-effort
        logger.warning("VLM captioning failed: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def caption_dataset(
    root: Path,
    *,
    strategy: CaptionStrategy = CaptionStrategy.TEMPLATE,
    category: Category = Category.SAREE,
    trigger: str = DEFAULT_TRIGGER,
    default_style: DrapeStyle | None = None,
    overwrite: bool = False,
    vlm_repo: str = DEFAULT_VLM_REPO,
    dry_run: bool = False,
) -> CaptionResult:
    """Write a ``.txt`` caption beside every image in ``root``.

    Parameters
    ----------
    root:
        Dataset directory (scanned recursively).
    strategy:
        ``template`` or ``vlm``.
    category:
        Garment category for the vocabulary.
    trigger:
        Rare token prepended to every caption.
    default_style:
        Drape assumed when the filename does not name one. ``None`` leaves such
        images with a generic caption and reports them.
    overwrite:
        Replace existing caption files rather than skipping them.
    vlm_repo:
        VLM repo id for the ``vlm`` strategy.
    dry_run:
        Print what would be written without touching disk.

    Returns
    -------
    CaptionResult
    """
    from ai_trial_room.training.dataset import iter_image_files

    images = list(iter_image_files(root))
    if not images:
        logger.error("No images found in %s", root)
        return CaptionResult([], [], [], [])

    result = CaptionResult(written=[], skipped=[], failed=[], untagged=[])
    logger.info(
        "Captioning %d image(s) in %s using the %s strategy",
        len(images),
        root,
        strategy.value,
    )

    for index, image_path in enumerate(images, start=1):
        caption_path = image_path.with_suffix(".txt")

        if caption_path.exists() and not overwrite:
            result.skipped.append(caption_path)
            continue

        style = parse_drape_from_name(image_path) or default_style
        if parse_drape_from_name(image_path) is None:
            result.untagged.append(image_path)

        detail = ""
        if strategy is CaptionStrategy.VLM:
            try:
                image = load_image(image_path, label="training image")
                detail = vlm_describe(image, repo_id=vlm_repo)
            except Exception as exc:  # noqa: BLE001
                result.failed.append((image_path, str(exc)[:120]))
                continue

        caption = template_caption(
            style, category=category, trigger=trigger, detail=detail
        )

        if dry_run:
            logger.info("[dry-run] %s -> %s", caption_path.name, caption[:110])
            result.written.append(caption_path)
            continue

        try:
            caption_path.write_text(caption + "\n", encoding="utf-8")
            result.written.append(caption_path)
        except OSError as exc:
            result.failed.append((caption_path, str(exc)[:120]))

        if index % 25 == 0:
            logger.info("  %d/%d captioned", index, len(images))

    logger.info(
        "Captioning done: %d written, %d skipped, %d failed",
        len(result.written),
        len(result.skipped),
        len(result.failed),
    )
    return result


def caption_stats(samples: Sequence[DrapeSample]) -> dict[str, float]:
    """Summarise caption length and trigger coverage.

    A LoRA trains poorly when captions are near-identical (nothing to
    discriminate on) or when the trigger is missing from some of them.

    Returns
    -------
    dict
        ``mean_words``, ``min_words``, ``max_words``, ``unique_ratio``,
        ``trigger_coverage``.
    """
    if not samples:
        return {}

    word_counts = [len(sample.caption.split()) for sample in samples]
    unique = len({sample.caption for sample in samples})
    with_trigger = sum(1 for s in samples if s.caption.startswith(DEFAULT_TRIGGER))

    return {
        "mean_words": sum(word_counts) / len(word_counts),
        "min_words": float(min(word_counts)),
        "max_words": float(max(word_counts)),
        "unique_ratio": unique / len(samples),
        "trigger_coverage": with_trigger / len(samples),
    }
