"""Run the same inputs through several backends and save a labelled grid.

Purpose: an honest, reproducible A/B that answers "which model should this shop
pay for?" - and doubles as the image you post on LinkedIn.

Usage
-----
Compare the two commercially licensed backends::

    python scripts/compare_backends.py \\
        --person assets/examples/person_01.jpg \\
        --garment assets/examples/saree_01.jpg \\
        --category saree

Add the research-licensed warping baseline (output must not be sold)::

    ALLOW_NONCOMMERCIAL=1 python scripts/compare_backends.py \\
        --person assets/examples/person_01.jpg \\
        --garment assets/examples/kurti_01.jpg \\
        --category kurti --backends qwen_edit flux_klein catvton

Sweep all four saree drapes through one backend::

    python scripts/compare_backends.py \\
        --person p.jpg --garment s.jpg --category saree --sweep-drapes

The same seed is used for every cell unless ``--seed -1``, so differences you
see are the model, not the noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.backends.base import TryOnOptions  # noqa: E402
from ai_trial_room.backends.registry import available_backends, unload_all  # noqa: E402
from ai_trial_room.config import (  # noqa: E402
    CONFIG,
    MODEL_SPECS,
    OUTPUT_DIR,
    BackendId,
    Category,
    DrapeStyle,
)
from ai_trial_room.router import TryOnReport, run_try_on  # noqa: E402
from ai_trial_room.utils.errors import TrialRoomError  # noqa: E402
from ai_trial_room.utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Height each cell is scaled to in the grid.
CELL_HEIGHT = 640
#: Vertical space reserved for the caption strip under each cell.
CAPTION_HEIGHT = 74
PADDING = 14
BACKGROUND = (250, 249, 247)
TEXT_COLOR = (28, 25, 23)
MUTED_COLOR = (120, 113, 108)
WARN_COLOR = (180, 50, 40)


@dataclass
class Cell:
    """One panel of the comparison grid."""

    image: Image.Image
    title: str
    subtitle: str
    warning: str = ""


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font, falling back to PIL's bitmap default.

    Kaggle images ship DejaVu; Windows ships Arial. Either is fine, and the
    bitmap fallback keeps the script working on a bare container.
    """
    for name in ("DejaVuSans.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scale_to_height(image: Image.Image, height: int) -> Image.Image:
    """Resize preserving aspect ratio to an exact pixel height."""
    width = max(1, round(image.width * height / image.height))
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def build_grid(cells: list[Cell], *, header: str) -> Image.Image:
    """Compose labelled cells into a single horizontal grid image.

    Parameters
    ----------
    cells:
        Panels, left to right. The first is usually the original photo.
    header:
        Title drawn across the top.

    Returns
    -------
    Image.Image
    """
    scaled = [_scale_to_height(cell.image, CELL_HEIGHT) for cell in cells]
    title_font = _load_font(24)
    label_font = _load_font(17)
    small_font = _load_font(14)

    header_height = 52
    total_width = sum(image.width for image in scaled) + PADDING * (len(scaled) + 1)
    total_height = header_height + CELL_HEIGHT + CAPTION_HEIGHT + PADDING * 2

    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((PADDING, 14), header, fill=TEXT_COLOR, font=title_font)

    x = PADDING
    for cell, image in zip(cells, scaled):
        y = header_height + PADDING
        canvas.paste(image, (x, y))

        caption_y = y + CELL_HEIGHT + 8
        draw.text((x, caption_y), cell.title, fill=TEXT_COLOR, font=label_font)
        draw.text((x, caption_y + 22), cell.subtitle, fill=MUTED_COLOR, font=small_font)
        if cell.warning:
            draw.text((x, caption_y + 40), cell.warning, fill=WARN_COLOR, font=small_font)

        x += image.width + PADDING

    return canvas


def _run_one(
    person: Path,
    garment: Path,
    category: Category,
    backend_id: BackendId,
    options: TryOnOptions,
) -> TryOnReport | None:
    """Run a single backend, returning ``None`` on a handled failure.

    A missing optional backend must not abort the whole comparison, so
    :class:`TrialRoomError` is caught and reported rather than raised.
    """
    spec = MODEL_SPECS[backend_id]
    logger.info("--- %s (%s) ---", backend_id.value, spec.repo_id)
    try:
        return run_try_on(
            person,
            garment,
            category,
            options=options,
            backend_id=backend_id,
            consent=True,  # CLI operator asserts they own the inputs
            progress=lambda fraction, message: logger.debug("%.0f%% %s", fraction * 100, message),
        )
    except TrialRoomError as exc:
        logger.error("%s failed: %s (%s)", backend_id.value, exc.user_message, exc.detail)
        return None
    finally:
        # Never let two heavy models co-reside between cells.
        unload_all()


def compare_backends(
    person: Path,
    garment: Path,
    category: Category,
    backend_ids: list[BackendId],
    options: TryOnOptions,
) -> tuple[list[Cell], list[dict]]:
    """Run each backend on the same inputs.

    Returns
    -------
    tuple
        ``(cells, records)`` where ``records`` is JSON-serialisable metadata.
    """
    cells: list[Cell] = []
    records: list[dict] = []
    original: Image.Image | None = None

    for backend_id in backend_ids:
        report = _run_one(person, garment, category, backend_id, options)
        if report is None:
            continue

        original = original or report.before
        spec = MODEL_SPECS[backend_id]
        cells.append(
            Cell(
                image=report.after,
                title=spec.repo_id.split("/")[-1],
                subtitle=(
                    f"{report.result.steps} steps · seed {report.result.seed} · "
                    f"{sum(report.timings.values()):.1f}s · {spec.license_name}"
                ),
                warning="" if spec.is_commercial else "NON-COMMERCIAL - not for resale",
            )
        )
        records.append(
            {
                "backend": backend_id.value,
                "repo_id": spec.repo_id,
                "license": spec.license_name,
                "commercial_use": spec.is_commercial,
                "seed": report.result.seed,
                "steps": report.result.steps,
                "timings_s": {k: round(v, 2) for k, v in report.timings.items()},
                "total_s": round(sum(report.timings.values()), 2),
                "prompt": report.result.prompt,
            }
        )

    if original is not None:
        cells.insert(0, Cell(image=original, title="Original", subtitle="uploaded photo"))
    return cells, records


def compare_drapes(
    person: Path,
    garment: Path,
    backend_id: BackendId,
    options: TryOnOptions,
) -> tuple[list[Cell], list[dict]]:
    """Run all four saree drape styles through one backend.

    This is the comparison that actually shows the Indian-wear specialisation:
    same model, same seed, four different regional drapes.
    """
    cells: list[Cell] = []
    records: list[dict] = []
    original: Image.Image | None = None

    for style in DrapeStyle:
        style_options = TryOnOptions(**{**options.__dict__, "drape_style": style})
        report = _run_one(person, garment, Category.SAREE, backend_id, style_options)
        if report is None:
            continue

        original = original or report.before
        cells.append(
            Cell(
                image=report.after,
                title=style.label.split(" (")[0],
                subtitle=(
                    f"{style.label.split('(')[-1].rstrip(')')} · "
                    f"{sum(report.timings.values()):.1f}s"
                ),
            )
        )
        records.append(
            {
                "drape_style": style.value,
                "backend": backend_id.value,
                "seed": report.result.seed,
                "total_s": round(sum(report.timings.values()), 2),
            }
        )

    if original is not None:
        cells.insert(0, Cell(image=original, title="Original", subtitle="uploaded photo"))
    return cells, records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--person", type=Path, required=True, help="Person photo.")
    parser.add_argument("--garment", type=Path, required=True, help="Garment photo.")
    parser.add_argument(
        "--category",
        type=str,
        default="saree",
        choices=[c.value for c in Category],
        help="Garment category.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=[b.value for b in BackendId],
        help="Backends to compare. Defaults to every permitted backend.",
    )
    parser.add_argument(
        "--sweep-drapes",
        action="store_true",
        help="Instead of comparing backends, compare all four saree drapes.",
    )
    parser.add_argument("--steps", type=int, default=CONFIG.runtime.default_steps)
    parser.add_argument("--cfg", type=float, default=CONFIG.runtime.default_true_cfg)
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Fixed seed so cells are comparable. -1 for random.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    for path, label in ((args.person, "person"), (args.garment, "garment")):
        if not path.exists():
            logger.error("Missing %s image: %s", label, path)
            return 2

    category = Category(args.category)
    options = TryOnOptions(steps=args.steps, true_cfg_scale=args.cfg, seed=args.seed)

    if args.sweep_drapes:
        if category is not Category.SAREE:
            logger.error("--sweep-drapes only applies to --category saree")
            return 2
        backend_id = (args.backends and BackendId(args.backends[0])) or CONFIG.default_backend
        header = f"AI Trial Room - saree drape styles - {MODEL_SPECS[backend_id].repo_id}"
        cells, records = compare_drapes(args.person, args.garment, backend_id, options)
    else:
        backend_ids = (
            [BackendId(name) for name in args.backends]
            if args.backends
            else available_backends()
        )
        logger.info("Comparing: %s", ", ".join(b.value for b in backend_ids))
        header = f"AI Trial Room - {category.label} - backend comparison"
        cells, records = compare_backends(
            args.person, args.garment, category, backend_ids, options
        )

    if not cells:
        logger.error("Every backend failed; nothing to write. See the errors above.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "drapes" if args.sweep_drapes else "backends"
    out_path = args.out or OUTPUT_DIR / f"compare-{category.value}-{suffix}-{stamp}.png"

    grid = build_grid(cells, header=header)
    grid.save(out_path, format="PNG", optimize=True)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "category": category.value,
                "person": str(args.person),
                "garment": str(args.garment),
                "resolution": list(CONFIG.runtime.size),
                "runs": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Grid written to %s (%dx%d)", out_path, *grid.size)
    logger.info("Metadata written to %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
