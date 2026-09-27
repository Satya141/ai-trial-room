"""Write caption files for a saree LoRA dataset.

Usage
-----
Template captions from filename drape styles (free, instant)::

    python scripts/caption_dataset.py --dataset datasets/saree_drapes

Richer per-image captions with a vision-language model::

    python scripts/caption_dataset.py --dataset datasets/saree_drapes --strategy vlm

Preview without writing anything::

    python scripts/caption_dataset.py --dataset datasets/saree_drapes --dry-run

Assume a drape for images whose filename does not name one::

    python scripts/caption_dataset.py --dataset datasets/saree_drapes --default-style nivi

Naming
------
Name files ``<drape>_<number>.<ext>`` so the caption can name the drape::

    nivi_001.jpg  bengali_002.jpg  gujarati_003.jpg  nauvari_004.jpg

Anything else is captioned generically and reported, because a drape LoRA
trained on captions that never say "Nauvari" cannot be triggered by asking for
one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.config import Category, DrapeStyle  # noqa: E402
from ai_trial_room.training.caption import (  # noqa: E402
    DEFAULT_TRIGGER,
    DEFAULT_VLM_REPO,
    CaptionStrategy,
    caption_dataset,
    caption_stats,
)
from ai_trial_room.training.dataset import scan_drape_dataset  # noqa: E402
from ai_trial_room.utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Write .txt captions beside every image in a LoRA dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory (recursive)."
    )
    parser.add_argument(
        "--strategy",
        default=CaptionStrategy.TEMPLATE.value,
        choices=[s.value for s in CaptionStrategy],
        help="; ".join(f"{s.value}: {s.description}" for s in CaptionStrategy),
    )
    parser.add_argument(
        "--category",
        default=Category.SAREE.value,
        choices=[c.value for c in Category],
        help="Garment category supplying the caption vocabulary.",
    )
    parser.add_argument(
        "--default-style",
        default=None,
        choices=[d.value for d in DrapeStyle],
        help="Drape assumed when the filename does not name one.",
    )
    parser.add_argument(
        "--trigger", default=DEFAULT_TRIGGER,
        help="Rare token prepended to every caption (the LoRA trigger).",
    )
    parser.add_argument("--vlm-repo", default=DEFAULT_VLM_REPO)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing caption files."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show captions without writing."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    if not args.dataset.is_dir():
        logger.error("Dataset directory does not exist: %s", args.dataset)
        return 2

    result = caption_dataset(
        args.dataset,
        strategy=CaptionStrategy(args.strategy),
        category=Category(args.category),
        trigger=args.trigger,
        default_style=DrapeStyle(args.default_style) if args.default_style else None,
        overwrite=args.overwrite,
        vlm_repo=args.vlm_repo,
        dry_run=args.dry_run,
    )
    print()
    print(result.render())

    if args.dry_run:
        return 0

    # Re-scan so the caller sees exactly what training will see.
    report = scan_drape_dataset(args.dataset)
    print()
    print(report.render())

    stats = caption_stats(report.drape_samples)
    if stats:
        print()
        print(
            f"Caption length: mean {stats['mean_words']:.0f} words "
            f"(min {stats['min_words']:.0f}, max {stats['max_words']:.0f})"
        )
        print(f"Unique captions: {stats['unique_ratio'] * 100:.0f}% of samples")
        print(f"Trigger coverage: {stats['trigger_coverage'] * 100:.0f}%")

        if stats["unique_ratio"] < 0.5:
            print(
                "\n  Note: more than half the captions are identical. A LoRA needs "
                "variation to learn from - consider --strategy vlm for per-image "
                "detail."
            )

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
