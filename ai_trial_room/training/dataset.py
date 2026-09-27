"""Dataset layouts for saree LoRA fine-tuning.

Two modes, because they need very different data and only one of them is
realistically collectable by a shop.

``drape`` (default, recommended)
    Single images of correctly draped garments, plus a caption each. The LoRA
    teaches the transformer *what a Nauvari drape looks like* - the visual
    distribution the base model is weakest on. A shop already owns this data:
    it is their catalogue photography.

``paired``
    Triplets of (person before, garment reference, person wearing it) plus an
    instruction. This trains the edit behaviour directly and is strictly better
    - but it requires photographing the same person in two outfits under the
    same lighting, for hundreds of garments. Almost nobody has this.

Why a ``drape`` LoRA still helps an *editing* model
---------------------------------------------------
Qwen-Image-Edit and FLUX.2 klein use one transformer for both generation and
editing; the edit path just adds reference-image conditioning. Teaching that
transformer the drape distribution therefore improves the edit path too, because
it is the same weights being asked the same visual question. It will not teach
the model to *transfer* a specific garment better - only ``paired`` data does
that - so expect better drapes, not better print fidelity.

Layouts
-------
``drape``::

    datasets/saree_drapes/
        nivi_001.jpg
        nivi_001.txt          <- caption for nivi_001.jpg
        gujarati_004.png
        gujarati_004.txt
        ...

The filename prefix before the first ``_`` is read as the drape style when it
matches a known one, which lets :mod:`ai_trial_room.training.caption` write
style-aware captions without any manual tagging.

``paired``::

    datasets/saree_paired/
        condition/  0001.jpg      <- the person, before
        garment/    0001.jpg      <- the garment reference (optional)
        target/     0001.jpg      <- the person wearing it
        prompts/    0001.txt      <- the instruction

Files are matched on stem, so ``condition/0001.jpg`` pairs with
``target/0001.png`` regardless of extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

from PIL import Image

from ai_trial_room.config import Category, DrapeStyle
from ai_trial_room.utils.errors import InvalidInputError
from ai_trial_room.utils.image_io import letterbox, load_image
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Extensions accepted when scanning a dataset directory.
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)

#: Subdirectory names used by the ``paired`` layout.
PAIRED_DIRS: Final[dict[str, bool]] = {
    "condition": True,  # required
    "target": True,  # required
    "garment": False,  # optional
    "prompts": False,  # optional (falls back to a generated instruction)
}

#: Below this, a LoRA overfits to the point of reproducing training images.
MIN_RECOMMENDED_SAMPLES: Final[int] = 40

#: Diminishing returns past this on a single-concept drape LoRA.
GOOD_SAMPLE_COUNT: Final[int] = 150


class DatasetMode(str, Enum):
    """Which dataset layout to read."""

    DRAPE = "drape"
    PAIRED = "paired"

    @property
    def description(self) -> str:
        """One-line explanation for CLI help and logs."""
        return _MODE_DESCRIPTIONS[self]


_MODE_DESCRIPTIONS: Final[dict[DatasetMode, str]] = {
    DatasetMode.DRAPE: "single images + captions; teaches the drape distribution",
    DatasetMode.PAIRED: "condition/target/instruction triplets; teaches the edit itself",
}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DrapeSample:
    """One captioned image in a ``drape`` dataset."""

    image_path: Path
    caption: str
    drape_style: DrapeStyle | None
    """Parsed from the filename prefix, or ``None`` when it is not recognised."""

    @property
    def caption_path(self) -> Path:
        """Where this sample's caption file lives."""
        return self.image_path.with_suffix(".txt")


@dataclass(frozen=True)
class PairedSample:
    """One edit triplet in a ``paired`` dataset."""

    condition_path: Path
    target_path: Path
    instruction: str
    garment_path: Path | None = None

    @property
    def stem(self) -> str:
        """Shared filename stem that links the triplet."""
        return self.condition_path.stem


@dataclass
class DatasetReport:
    """Result of scanning and validating a dataset directory."""

    mode: DatasetMode
    root: Path
    drape_samples: list[DrapeSample] = field(default_factory=list)
    paired_samples: list[PairedSample] = field(default_factory=list)
    #: Problems that make a file unusable; it was skipped.
    errors: list[str] = field(default_factory=list)
    #: Problems worth fixing that did not block loading.
    warnings: list[str] = field(default_factory=list)
    #: Count per drape style, for the class-balance warning.
    style_counts: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of usable samples found."""
        return len(self.drape_samples) + len(self.paired_samples)

    @property
    def ok(self) -> bool:
        """True when at least one sample loaded and nothing was fatal."""
        return self.count > 0 and not self.errors

    def render(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [
            f"Dataset: {self.root}",
            f"Mode:    {self.mode.value} ({self.mode.description})",
            f"Samples: {self.count}",
        ]
        if self.style_counts:
            spread = ", ".join(
                f"{style}={count}" for style, count in sorted(self.style_counts.items())
            )
            lines.append(f"Styles:  {spread}")
        for warning in self.warnings:
            lines.append(f"  warn:  {warning}")
        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #


def parse_drape_from_name(path: Path) -> DrapeStyle | None:
    """Read a drape style from a filename prefix.

    ``nivi_001.jpg`` -> :attr:`DrapeStyle.NIVI`. Returns ``None`` when the prefix
    is not a known style, which is fine - the caption then carries the style.

    Parameters
    ----------
    path:
        Image path.

    Returns
    -------
    DrapeStyle or None
    """
    prefix = path.stem.split("_")[0].lower()
    try:
        return DrapeStyle(prefix)
    except ValueError:
        return None


def _read_caption(path: Path) -> str:
    """Read a caption file, collapsing whitespace. Missing file -> empty string."""
    if not path.exists():
        return ""
    try:
        return " ".join(path.read_text(encoding="utf-8").split())
    except OSError as exc:  # pragma: no cover - unreadable file
        logger.warning("Could not read caption %s: %s", path, exc)
        return ""


def scan_drape_dataset(root: Path) -> DatasetReport:
    """Scan a ``drape`` dataset directory.

    Parameters
    ----------
    root:
        Directory holding images and sibling ``.txt`` captions.

    Returns
    -------
    DatasetReport
    """
    report = DatasetReport(mode=DatasetMode.DRAPE, root=root)

    if not root.is_dir():
        report.errors.append(f"{root} is not a directory")
        return report

    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        report.errors.append(
            f"No images found in {root}. Expected files like nivi_001.jpg "
            "alongside nivi_001.txt captions."
        )
        return report

    uncaptioned: list[str] = []

    for image_path in images:
        caption = _read_caption(image_path.with_suffix(".txt"))
        style = parse_drape_from_name(image_path)

        if not caption:
            uncaptioned.append(image_path.name)

        report.drape_samples.append(
            DrapeSample(image_path=image_path, caption=caption, drape_style=style)
        )
        key = style.value if style else "untagged"
        report.style_counts[key] = report.style_counts.get(key, 0) + 1

    if uncaptioned:
        shown = ", ".join(uncaptioned[:5])
        more = f" (+{len(uncaptioned) - 5} more)" if len(uncaptioned) > 5 else ""
        report.errors.append(
            f"{len(uncaptioned)} image(s) have no caption: {shown}{more}. "
            "Run `python scripts/caption_dataset.py` first."
        )

    _add_size_warnings(report)
    return report


def scan_paired_dataset(root: Path) -> DatasetReport:
    """Scan a ``paired`` dataset directory.

    Parameters
    ----------
    root:
        Directory containing ``condition/``, ``target/`` and optionally
        ``garment/`` and ``prompts/``.

    Returns
    -------
    DatasetReport
    """
    report = DatasetReport(mode=DatasetMode.PAIRED, root=root)

    if not root.is_dir():
        report.errors.append(f"{root} is not a directory")
        return report

    for name, required in PAIRED_DIRS.items():
        if required and not (root / name).is_dir():
            report.errors.append(
                f"Missing required directory {root / name}. The paired layout needs "
                "condition/ and target/ (garment/ and prompts/ are optional)."
            )
    if report.errors:
        return report

    def _index(folder: Path) -> dict[str, Path]:
        if not folder.is_dir():
            return {}
        return {
            path.stem: path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }

    conditions = _index(root / "condition")
    targets = _index(root / "target")
    garments = _index(root / "garment")

    if not conditions:
        report.errors.append(f"No images in {root / 'condition'}")
        return report

    orphans = sorted(set(conditions) - set(targets))
    if orphans:
        shown = ", ".join(orphans[:5])
        report.warnings.append(
            f"{len(orphans)} condition image(s) have no matching target and were "
            f"skipped: {shown}"
        )

    for stem in sorted(set(conditions) & set(targets)):
        instruction = _read_caption(root / "prompts" / f"{stem}.txt")
        report.paired_samples.append(
            PairedSample(
                condition_path=conditions[stem],
                target_path=targets[stem],
                instruction=instruction,
                garment_path=garments.get(stem),
            )
        )

    missing_prompts = sum(1 for s in report.paired_samples if not s.instruction)
    if missing_prompts:
        report.warnings.append(
            f"{missing_prompts} triplet(s) have no prompt file; a generic "
            "instruction will be generated at training time."
        )

    _add_size_warnings(report)
    return report


def _add_size_warnings(report: DatasetReport) -> None:
    """Append dataset-size and class-balance warnings in place."""
    count = report.count

    if 0 < count < MIN_RECOMMENDED_SAMPLES:
        report.warnings.append(
            f"Only {count} sample(s). Below ~{MIN_RECOMMENDED_SAMPLES} a LoRA tends "
            "to memorise individual photos rather than learn the drape. Aim for "
            f"{GOOD_SAMPLE_COUNT}."
        )
    elif count < GOOD_SAMPLE_COUNT:
        report.warnings.append(
            f"{count} samples is workable; {GOOD_SAMPLE_COUNT}+ gives noticeably "
            "better generalisation."
        )

    tagged = {k: v for k, v in report.style_counts.items() if k != "untagged"}
    if len(tagged) > 1:
        smallest, largest = min(tagged.values()), max(tagged.values())
        if largest > smallest * 4:
            report.warnings.append(
                "Drape styles are badly imbalanced "
                f"({', '.join(f'{k}={v}' for k, v in sorted(tagged.items()))}). "
                "The LoRA will favour the over-represented style."
            )


def scan_dataset(root: Path, mode: DatasetMode) -> DatasetReport:
    """Scan ``root`` using the layout for ``mode``."""
    if mode is DatasetMode.DRAPE:
        return scan_drape_dataset(root)
    return scan_paired_dataset(root)


def detect_mode(root: Path) -> DatasetMode:
    """Guess the layout from the directory contents.

    A directory containing both ``condition/`` and ``target/`` is a paired
    dataset; anything else is treated as a drape dataset.
    """
    if (root / "condition").is_dir() and (root / "target").is_dir():
        return DatasetMode.PAIRED
    return DatasetMode.DRAPE


# --------------------------------------------------------------------------- #
# Torch datasets
# --------------------------------------------------------------------------- #


def _to_tensor(image: Image.Image) -> Any:
    """Convert a PIL image to a ``[-1, 1]`` CHW float tensor.

    The ``[-1, 1]`` range is what diffusion VAEs expect.
    """
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class DrapeLoraDataset:
    """Torch-style dataset over a ``drape`` sample list.

    Implements ``__len__``/``__getitem__`` only, so it works with
    :class:`torch.utils.data.DataLoader` without importing torch at module
    import time.

    Parameters
    ----------
    samples:
        Samples from :func:`scan_drape_dataset`.
    resolution:
        Target ``(width, height)``. Images are letterboxed, never cropped, so a
        hem is never cut off.
    caption_dropout:
        Probability of replacing a caption with an empty string, which trains
        the unconditional path and improves classifier-free guidance at
        inference. 0.05-0.10 is typical.
    seed:
        Seed for the caption-dropout RNG, so a run is reproducible.
    """

    def __init__(
        self,
        samples: Sequence[DrapeSample],
        *,
        resolution: tuple[int, int] = (768, 1024),
        caption_dropout: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise InvalidInputError(
                "The dataset is empty.", detail="no samples passed to DrapeLoraDataset"
            )
        self.samples = list(samples)
        self.resolution = resolution
        self.caption_dropout = caption_dropout
        self._seed = seed

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return ``{"pixel_values": tensor, "caption": str, "style": str}``."""
        import random

        sample = self.samples[index]
        image = load_image(sample.image_path, label="training image")
        padded, _ = letterbox(image, self.resolution, fill=(255, 255, 255))

        rng = random.Random(self._seed * 1_000_003 + index)
        caption = "" if rng.random() < self.caption_dropout else sample.caption

        return {
            "pixel_values": _to_tensor(padded),
            "caption": caption,
            "style": sample.drape_style.value if sample.drape_style else "",
        }


class PairedLoraDataset:
    """Torch-style dataset over a ``paired`` sample list.

    Yields the condition image, an optional garment reference, the target image
    and the instruction - the four things an edit LoRA needs.
    """

    #: Instruction used when a triplet has no prompt file.
    DEFAULT_INSTRUCTION: Final[str] = (
        "Dress the person in the first image in the garment shown in the second "
        "image, keeping their face, body shape, pose and background unchanged."
    )

    def __init__(
        self,
        samples: Sequence[PairedSample],
        *,
        resolution: tuple[int, int] = (768, 1024),
        caption_dropout: float = 0.05,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise InvalidInputError(
                "The dataset is empty.", detail="no samples passed to PairedLoraDataset"
            )
        self.samples = list(samples)
        self.resolution = resolution
        self.caption_dropout = caption_dropout
        self._seed = seed

    def __len__(self) -> int:
        """Number of triplets."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return condition/target tensors, an optional garment tensor and text."""
        import random

        sample = self.samples[index]

        def _prepare(path: Path, label: str) -> Any:
            image = load_image(path, label=label)
            padded, _ = letterbox(image, self.resolution, fill=(255, 255, 255))
            return _to_tensor(padded)

        rng = random.Random(self._seed * 1_000_003 + index)
        instruction = sample.instruction or self.DEFAULT_INSTRUCTION
        if rng.random() < self.caption_dropout:
            instruction = ""

        item: dict[str, Any] = {
            "condition_values": _prepare(sample.condition_path, "condition image"),
            "target_values": _prepare(sample.target_path, "target image"),
            "caption": instruction,
        }
        if sample.garment_path is not None:
            item["garment_values"] = _prepare(sample.garment_path, "garment image")
        return item


def build_dataset(
    report: DatasetReport,
    *,
    resolution: tuple[int, int] = (768, 1024),
    caption_dropout: float = 0.05,
    seed: int = 0,
) -> DrapeLoraDataset | PairedLoraDataset:
    """Construct the right dataset class for a scan report.

    Raises
    ------
    InvalidInputError
        When the report contains no usable samples.
    """
    if not report.ok:
        detail = "; ".join(report.errors) or "no samples"
        raise InvalidInputError(
            f"Dataset at {report.root} is not usable. {detail}",
            detail=detail,
        )

    kwargs = {
        "resolution": resolution,
        "caption_dropout": caption_dropout,
        "seed": seed,
    }
    if report.mode is DatasetMode.DRAPE:
        return DrapeLoraDataset(report.drape_samples, **kwargs)  # type: ignore[arg-type]
    return PairedLoraDataset(report.paired_samples, **kwargs)  # type: ignore[arg-type]


def iter_image_files(root: Path) -> Iterator[Path]:
    """Yield every image under ``root``, sorted, recursing into subdirectories."""
    yield from (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
