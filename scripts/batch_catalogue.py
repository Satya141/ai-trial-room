"""Batch-generate try-on images for a whole shop catalogue.

This is the feature a shop actually buys. One model photo, a folder of garment
photos, and every combination rendered overnight into a folder they can drop
straight onto their website or WhatsApp catalogue.

Usage
-----
Every saree in a folder, on one model::

    python scripts/batch_catalogue.py \\
        --person models/priya.jpg \\
        --garments catalogue/sarees \\
        --category saree

Several models against several garments (the full cross product)::

    python scripts/batch_catalogue.py \\
        --persons models/ \\
        --garments catalogue/sarees \\
        --category saree --preset best

All four drapes for every saree, for a lookbook::

    python scripts/batch_catalogue.py \\
        --person models/priya.jpg \\
        --garments catalogue/sarees \\
        --category saree --all-drapes

Resume an interrupted run (skips outputs that already exist)::

    python scripts/batch_catalogue.py ... --resume

Outputs
-------
``<out>/images/<person>__<garment>__<drape>.jpg``
    One render per combination.
``<out>/manifest.csv``
    A row per job: inputs, settings, seed, timings, validation findings. This is
    what lets the shop sort by "needs a human look" rather than reviewing all
    400 images.
``<out>/contact_sheet_NN.jpg``
    Thumbnail grids for quick review.

Privacy note
------------
Unlike the interactive app, batch mode writes results to disk *by design* - that
is the deliverable. Inputs are never copied, and the manifest records filenames
only, no image data. Point ``--out`` somewhere you control.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.backends.base import TryOnOptions  # noqa: E402
from ai_trial_room.backends.registry import unload_all  # noqa: E402
from ai_trial_room.config import (  # noqa: E402
    OUTPUT_DIR,
    BackendId,
    Category,
    DrapeStyle,
    DupattaStyle,
    QualityPreset,
)
from ai_trial_room.router import run_try_on  # noqa: E402
from ai_trial_room.utils.errors import TrialRoomError  # noqa: E402
from ai_trial_room.utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Image extensions considered when scanning an input folder.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

#: Thumbnails per contact sheet, and the grid width.
SHEET_CAPACITY: int = 24
SHEET_COLUMNS: int = 6
THUMB_HEIGHT: int = 280

MANIFEST_FIELDS: Sequence[str] = (
    "status",
    "person",
    "garment",
    "category",
    "drape_style",
    "dupatta_style",
    "output",
    "backend",
    "license",
    "seed",
    "steps",
    "refined",
    "total_s",
    "validation_ok",
    "findings",
    "error",
)


@dataclass(frozen=True)
class Job:
    """One person x garment x drape combination to render."""

    person: Path
    garment: Path
    category: Category
    drape_style: DrapeStyle
    dupatta_style: DupattaStyle

    def output_name(self, suffix: str = ".jpg") -> str:
        """Build a deterministic, filesystem-safe output filename.

        Deterministic matters for ``--resume``: the same job must always map to
        the same filename so an interrupted run can skip what it already did.
        """
        parts = [self.person.stem, self.garment.stem]
        if self.category is Category.SAREE:
            parts.append(self.drape_style.value)
        elif self.category is Category.LEHENGA:
            parts.append(self.dupatta_style.value)
        safe = ["".join(c if c.isalnum() or c in "-_" else "-" for c in p) for p in parts]
        return "__".join(safe) + suffix


def collect_images(target: Path, *, label: str) -> list[Path]:
    """Return image files from a file or a directory.

    Parameters
    ----------
    target:
        A single image, or a directory to scan (non-recursively, sorted).
    label:
        Used in error messages.

    Returns
    -------
    list[Path]

    Raises
    ------
    SystemExit
        When nothing usable was found - a batch run should fail loudly and
        immediately rather than after loading 20 GB of weights.
    """
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"{label} {target} is not a supported image type.")
        return [target]

    if not target.is_dir():
        raise SystemExit(f"{label} path does not exist: {target}")

    found = sorted(
        path
        for path in target.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise SystemExit(f"No images found in {label} directory {target}")
    return found


def build_jobs(
    persons: Sequence[Path],
    garments: Sequence[Path],
    category: Category,
    *,
    all_drapes: bool,
    drape_style: DrapeStyle,
    dupatta_style: DupattaStyle,
) -> list[Job]:
    """Expand inputs into the full list of jobs.

    Parameters
    ----------
    persons, garments:
        Input image paths.
    category:
        Garment category for every job.
    all_drapes:
        For sarees, render every drape style rather than just ``drape_style``.
    drape_style, dupatta_style:
        Styles used when not sweeping.

    Returns
    -------
    list[Job]
        Ordered person-major, so a reviewer sees one model at a time.
    """
    if all_drapes and category is Category.SAREE:
        styles: Iterable[DrapeStyle] = list(DrapeStyle)
    else:
        styles = [drape_style]

    return [
        Job(
            person=person,
            garment=garment,
            category=category,
            drape_style=style,
            dupatta_style=dupatta_style,
        )
        for person in persons
        for garment in garments
        for style in styles
    ]


def run_jobs(
    jobs: Sequence[Job],
    out_dir: Path,
    options_template: TryOnOptions,
    *,
    backend_id: BackendId | None,
    resume: bool,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    """Render every job, writing images and collecting manifest rows.

    A failure on one job is recorded and the batch continues - a single bad photo
    in a 400-image catalogue must not cost the whole overnight run.

    Parameters
    ----------
    jobs:
        Jobs to render.
    out_dir:
        Root output directory. Images go in ``out_dir / "images"``.
    options_template:
        Base generation options; the drape and dupatta are set per job.
    backend_id:
        Force a backend, or ``None`` to let the router choose.
    resume:
        Skip jobs whose output file already exists.
    jpeg_quality:
        JPEG quality for written images.

    Returns
    -------
    list[dict]
        Manifest rows, one per job.
    """
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total = len(jobs)
    started_all = time.perf_counter()

    for index, job in enumerate(jobs, start=1):
        out_path = images_dir / job.output_name()
        prefix = f"[{index}/{total}]"

        if resume and out_path.exists():
            logger.info("%s skip (exists) %s", prefix, out_path.name)
            rows.append(
                {
                    "status": "skipped",
                    "person": job.person.name,
                    "garment": job.garment.name,
                    "category": job.category.value,
                    "drape_style": job.drape_style.value,
                    "dupatta_style": job.dupatta_style.value,
                    "output": out_path.name,
                    "error": "",
                }
            )
            continue

        logger.info(
            "%s %s + %s (%s/%s)",
            prefix,
            job.person.name,
            job.garment.name,
            job.category.value,
            job.drape_style.value,
        )

        options = TryOnOptions(**{**options_template.__dict__})
        options.drape_style = job.drape_style
        options.dupatta_style = job.dupatta_style

        try:
            report = run_try_on(
                job.person,
                job.garment,
                job.category,
                options=options,
                backend_id=backend_id,
                consent=True,  # batch operator asserts rights to these photos
                progress=lambda fraction, message: None,
            )
        except TrialRoomError as exc:
            logger.error("%s FAILED: %s", prefix, exc.user_message)
            rows.append(
                {
                    "status": "failed",
                    "person": job.person.name,
                    "garment": job.garment.name,
                    "category": job.category.value,
                    "drape_style": job.drape_style.value,
                    "dupatta_style": job.dupatta_style.value,
                    "output": "",
                    "error": exc.user_message,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the batch
            logger.exception("%s CRASHED", prefix)
            rows.append(
                {
                    "status": "error",
                    "person": job.person.name,
                    "garment": job.garment.name,
                    "category": job.category.value,
                    "drape_style": job.drape_style.value,
                    "dupatta_style": job.dupatta_style.value,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            continue

        report.after.convert("RGB").save(
            out_path, format="JPEG", quality=jpeg_quality, optimize=True
        )

        findings = report.validation.codes()
        rows.append(
            {
                "status": "ok",
                "person": job.person.name,
                "garment": job.garment.name,
                "category": job.category.value,
                "drape_style": job.drape_style.value,
                "dupatta_style": job.dupatta_style.value,
                "output": out_path.name,
                "backend": report.result.spec.repo_id,
                "license": report.result.spec.license_name,
                "seed": report.result.seed,
                "steps": report.result.steps,
                "refined": "|".join(report.result.refined_regions),
                "total_s": round(sum(report.timings.values()), 2),
                "validation_ok": report.validation.ok,
                "findings": "|".join(findings),
                "error": "",
            }
        )

        elapsed = time.perf_counter() - started_all
        done = index
        eta = (elapsed / done) * (total - done)
        logger.info(
            "%s done in %.1fs%s | ETA %s",
            prefix,
            sum(report.timings.values()),
            f" [{','.join(findings)}]" if findings else "",
            _format_duration(eta),
        )

    unload_all()
    return rows


def _format_duration(seconds: float) -> str:
    """Render a duration as ``1h 04m`` / ``12m 30s`` / ``45s``."""
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds)}s"


def write_manifest(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write the manifest CSV, filling missing keys with blanks."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    logger.info("Manifest written to %s (%d rows)", path, len(rows))


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font with a bitmap fallback."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _chunk(items: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    """Yield successive ``size``-length slices of ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_contact_sheets(images_dir: Path, out_dir: Path) -> list[Path]:
    """Build thumbnail grids so a human can review the batch quickly.

    Parameters
    ----------
    images_dir:
        Directory of rendered images.
    out_dir:
        Where to write the sheets.

    Returns
    -------
    list[Path]
        Paths of the sheets written.
    """
    rendered = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not rendered:
        return []

    font = _load_font(14)
    caption_h = 22
    sheets: list[Path] = []

    for sheet_index, batch in enumerate(_chunk(rendered, SHEET_CAPACITY), start=1):
        thumbs: list[tuple[Image.Image, str]] = []
        for path in batch:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                width = max(1, round(image.width * THUMB_HEIGHT / image.height))
                thumbs.append(
                    (image.resize((width, THUMB_HEIGHT), Image.Resampling.LANCZOS), path.stem)
                )

        columns = min(SHEET_COLUMNS, len(thumbs))
        rows_count = (len(thumbs) + columns - 1) // columns
        cell_w = max(thumb.width for thumb, _ in thumbs) + 8
        cell_h = THUMB_HEIGHT + caption_h + 8

        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_count), (250, 249, 247))
        draw = ImageDraw.Draw(sheet)

        for position, (thumb, caption) in enumerate(thumbs):
            column, row = position % columns, position // columns
            x = column * cell_w + 4
            y = row * cell_h + 4
            sheet.paste(thumb, (x, y))
            label = caption if len(caption) <= 34 else caption[:31] + "..."
            draw.text((x, y + THUMB_HEIGHT + 4), label, fill=(70, 64, 60), font=font)

        sheet_path = out_dir / f"contact_sheet_{sheet_index:02d}.jpg"
        sheet.save(sheet_path, format="JPEG", quality=88, optimize=True)
        sheets.append(sheet_path)
        logger.info("Contact sheet %s (%d images)", sheet_path.name, len(thumbs))

    return sheets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Batch-generate try-on images for a shop catalogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    people = parser.add_mutually_exclusive_group(required=True)
    people.add_argument("--person", type=Path, help="A single model photo.")
    people.add_argument("--persons", type=Path, help="A folder of model photos.")

    parser.add_argument(
        "--garments", type=Path, required=True,
        help="A garment image, or a folder of them.",
    )
    parser.add_argument(
        "--category", required=True, choices=[c.value for c in Category],
        help="Garment category for every job in this run.",
    )
    parser.add_argument(
        "--drape", default=DrapeStyle.NIVI.value, choices=[d.value for d in DrapeStyle],
        help="Saree drape style (ignored for other categories).",
    )
    parser.add_argument(
        "--all-drapes", action="store_true",
        help="For sarees, render every drape style for each garment.",
    )
    parser.add_argument(
        "--dupatta", default=DupattaStyle.SINGLE_SHOULDER.value,
        choices=[d.value for d in DupattaStyle],
        help="Lehenga dupatta style (ignored for other categories).",
    )
    parser.add_argument(
        "--preset", default=QualityPreset.BALANCED.value,
        choices=[p.value for p in QualityPreset],
        help="Quality preset.",
    )
    parser.add_argument(
        "--backend", default=None, choices=[b.value for b in BackendId],
        help="Force a backend. Default lets the router choose.",
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Fixed seed for reproducibility, or -1 for random per job.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip jobs whose output image already exists.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--no-contact-sheet", action="store_true",
        help="Skip building review thumbnail grids.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the jobs that would run, then exit without loading any model.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    persons = collect_images(args.person or args.persons, label="person")
    garments = collect_images(args.garments, label="garment")
    category = Category(args.category)

    jobs = build_jobs(
        persons,
        garments,
        category,
        all_drapes=args.all_drapes,
        drape_style=DrapeStyle(args.drape),
        dupatta_style=DupattaStyle(args.dupatta),
    )

    preset = QualityPreset(args.preset)
    logger.info(
        "%d job(s): %d person(s) x %d garment(s)%s | category=%s preset=%s",
        len(jobs),
        len(persons),
        len(garments),
        " x 4 drapes" if args.all_drapes and category is Category.SAREE else "",
        category.value,
        preset.value,
    )

    if args.dry_run:
        for job in jobs:
            print(f"  {job.output_name()}")
        logger.info("Dry run - nothing rendered.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(
        args.out or OUTPUT_DIR / f"catalogue-{category.value}-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing to %s", out_dir.resolve())

    options = TryOnOptions.from_preset(preset, seed=args.seed)
    rows = run_jobs(
        jobs,
        out_dir,
        options,
        backend_id=BackendId(args.backend) if args.backend else None,
        resume=args.resume,
        jpeg_quality=args.jpeg_quality,
    )

    write_manifest(rows, out_dir / "manifest.csv")

    if not args.no_contact_sheet:
        build_contact_sheets(out_dir / "images", out_dir)

    ok = sum(1 for row in rows if row.get("status") == "ok")
    flagged = sum(1 for row in rows if row.get("status") == "ok" and not row.get("validation_ok"))
    failed = sum(1 for row in rows if row.get("status") in {"failed", "error"})
    skipped = sum(1 for row in rows if row.get("status") == "skipped")

    logger.info(
        "Batch complete: %d rendered (%d flagged for review), %d failed, %d skipped -> %s",
        ok,
        flagged,
        failed,
        skipped,
        out_dir.resolve(),
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
