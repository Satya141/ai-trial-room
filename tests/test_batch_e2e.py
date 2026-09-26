"""End-to-end test of catalogue batch mode with a stubbed backend.

Exercises the real :func:`run_jobs` loop - image writing, manifest rows, resume
behaviour, per-job failure isolation and contact sheets - by substituting a fake
``run_try_on``. That covers the orchestration logic without a GPU, which is where
batch bugs actually live (a crashed job killing the run, ``--resume`` skipping the
wrong files, a manifest that loses the failure reason).

Run standalone::

    python tests/test_batch_e2e.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.backends.base import TryOnOptions  # noqa: E402
from ai_trial_room.config import (  # noqa: E402
    Category,
    DrapeStyle,
    DupattaStyle,
    QualityPreset,
)
from ai_trial_room.utils.errors import NoPersonDetectedError  # noqa: E402


def _load_batch() -> Any:
    """Import ``scripts/batch_catalogue.py`` as a module."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "batch_catalogue.py"
    spec = importlib.util.spec_from_file_location("batch_catalogue_e2e", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _fake_report(*, findings: list[str], seed: int = 7) -> SimpleNamespace:
    """Build the minimal object shape ``run_jobs`` consumes from a report."""
    return SimpleNamespace(
        after=Image.new("RGB", (240, 320), (180, 60, 80)),
        timings={"person": 0.4, "garment": 0.2, "generate": 3.1, "postprocess": 0.3},
        validation=SimpleNamespace(
            ok=not findings,
            codes=lambda: list(findings),
        ),
        result=SimpleNamespace(
            seed=seed,
            steps=30,
            refined_regions=["pallu / dupatta"] if findings == [] else [],
            spec=SimpleNamespace(
                repo_id="Qwen/Qwen-Image-Edit-2511",
                license_name="Apache-2.0",
            ),
        ),
    )


def _fixture_dir(root: Path) -> tuple[list[Path], list[Path]]:
    """Write one model photo and three garment photos into ``root``."""
    models, garments = root / "models", root / "garments"
    models.mkdir()
    garments.mkdir()

    person = models / "priya.jpg"
    Image.new("RGB", (600, 900), (210, 180, 160)).save(person)

    garment_paths = []
    for index, colour in enumerate([(180, 40, 60), (30, 90, 140), (200, 160, 40)], start=1):
        path = garments / f"saree_{index:02d}.jpg"
        Image.new("RGB", (400, 600), colour).save(path)
        garment_paths.append(path)

    return [person], garment_paths


def _jobs(batch: Any, persons: list[Path], garments: list[Path]) -> list[Any]:
    """Build a saree job list with the default drape."""
    return batch.build_jobs(
        persons,
        garments,
        Category.SAREE,
        all_drapes=False,
        drape_style=DrapeStyle.NIVI,
        dupatta_style=DupattaStyle.SINGLE_SHOULDER,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_batch_writes_images_and_a_complete_manifest() -> None:
    """Every job produces an image and a manifest row with provenance."""
    batch = _load_batch()

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)
        out_dir = root / "out"

        batch.run_try_on = lambda *a, **k: _fake_report(findings=[])

        rows = batch.run_jobs(
            _jobs(batch, persons, garments),
            out_dir,
            TryOnOptions.from_preset(QualityPreset.BALANCED, seed=7),
            backend_id=None,
            resume=False,
            jpeg_quality=90,
        )

        assert len(rows) == 3
        assert all(row["status"] == "ok" for row in rows)

        images = sorted((out_dir / "images").glob("*.jpg"))
        assert len(images) == 3
        assert all(image.stat().st_size > 0 for image in images)

        row = rows[0]
        assert row["backend"] == "Qwen/Qwen-Image-Edit-2511"
        assert row["license"] == "Apache-2.0"
        assert row["seed"] == 7
        assert row["validation_ok"] is True
        assert row["total_s"] == 4.0
        assert row["refined"] == "pallu / dupatta"

        batch.write_manifest(rows, out_dir / "manifest.csv")
        with (out_dir / "manifest.csv").open(encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        assert len(parsed) == 3
        assert parsed[0]["output"].endswith(".jpg")


def test_batch_records_validation_findings_for_review() -> None:
    """A flagged result is still written, but marked for a human to check."""
    batch = _load_batch()

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)

        batch.run_try_on = lambda *a, **k: _fake_report(
            findings=["background_drift", "exposure_drift"]
        )

        rows = batch.run_jobs(
            _jobs(batch, persons, garments),
            root / "out",
            TryOnOptions.from_preset(QualityPreset.FAST),
            backend_id=None,
            resume=False,
            jpeg_quality=85,
        )

        assert all(row["status"] == "ok" for row in rows)
        assert all(row["validation_ok"] is False for row in rows)
        assert rows[0]["findings"] == "background_drift|exposure_drift"
        # The image is still produced - a flagged result is the operator's call.
        assert len(list((root / "out" / "images").glob("*.jpg"))) == 3


def test_one_bad_photo_does_not_kill_the_batch() -> None:
    """A rejected input is recorded and the run continues.

    This is the property that matters for an overnight catalogue run.
    """
    batch = _load_batch()
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        if calls["n"] == 2:
            raise NoPersonDetectedError(detail="stub")
        return _fake_report(findings=[])

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)

        batch.run_try_on = flaky
        rows = batch.run_jobs(
            _jobs(batch, persons, garments),
            root / "out",
            TryOnOptions.from_preset(QualityPreset.FAST),
            backend_id=None,
            resume=False,
            jpeg_quality=85,
        )

    assert calls["n"] == 3, "all three jobs must be attempted"
    statuses = [row["status"] for row in rows]
    assert statuses.count("ok") == 2
    assert statuses.count("failed") == 1

    failed = next(row for row in rows if row["status"] == "failed")
    assert "No person detected" in str(failed["error"])
    assert failed["output"] == ""


def test_unexpected_crash_is_isolated_too() -> None:
    """A non-TrialRoomError exception must also be contained."""
    batch = _load_batch()
    calls = {"n": 0}

    def crashy(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("something exploded in the pipeline")
        return _fake_report(findings=[])

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)

        batch.run_try_on = crashy
        rows = batch.run_jobs(
            _jobs(batch, persons, garments),
            root / "out",
            TryOnOptions.from_preset(QualityPreset.FAST),
            backend_id=None,
            resume=False,
            jpeg_quality=85,
        )

    assert calls["n"] == 3
    assert [row["status"] for row in rows].count("error") == 1
    crashed = next(row for row in rows if row["status"] == "error")
    assert "RuntimeError" in str(crashed["error"])


def test_resume_skips_only_what_already_exists() -> None:
    """A second run must re-render nothing, and a deleted file must re-render."""
    batch = _load_batch()
    calls = {"n": 0}

    def counting(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _fake_report(findings=[])

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)
        out_dir = root / "out"
        jobs = _jobs(batch, persons, garments)
        options = TryOnOptions.from_preset(QualityPreset.FAST)

        batch.run_try_on = counting

        batch.run_jobs(jobs, out_dir, options, backend_id=None, resume=False, jpeg_quality=85)
        assert calls["n"] == 3

        # Second pass with --resume: everything is already there.
        rows = batch.run_jobs(
            jobs, out_dir, options, backend_id=None, resume=True, jpeg_quality=85
        )
        assert calls["n"] == 3, "resume must not re-render existing outputs"
        assert all(row["status"] == "skipped" for row in rows)

        # Delete one and resume again: exactly one job should run.
        images = sorted((out_dir / "images").glob("*.jpg"))
        images[1].unlink()

        rows = batch.run_jobs(
            jobs, out_dir, options, backend_id=None, resume=True, jpeg_quality=85
        )
        assert calls["n"] == 4, "the deleted output must be regenerated"
        assert [row["status"] for row in rows].count("skipped") == 2
        assert [row["status"] for row in rows].count("ok") == 1


def test_per_job_drape_reaches_the_options() -> None:
    """A drape sweep must pass a different drape into each generation."""
    batch = _load_batch()
    seen: list[str] = []

    def capture(person: Any, garment: Any, category: Any, **kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs["options"].drape_style.value)
        return _fake_report(findings=[])

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)

        jobs = batch.build_jobs(
            persons,
            garments[:1],
            Category.SAREE,
            all_drapes=True,
            drape_style=DrapeStyle.NIVI,
            dupatta_style=DupattaStyle.SINGLE_SHOULDER,
        )

        batch.run_try_on = capture
        batch.run_jobs(
            jobs,
            root / "out",
            TryOnOptions.from_preset(QualityPreset.FAST),
            backend_id=None,
            resume=False,
            jpeg_quality=85,
        )

    assert seen == [style.value for style in DrapeStyle], seen


def test_contact_sheet_covers_every_rendered_image() -> None:
    """The review grid is produced from whatever was written."""
    batch = _load_batch()

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        persons, garments = _fixture_dir(root)
        out_dir = root / "out"

        batch.run_try_on = lambda *a, **k: _fake_report(findings=[])
        batch.run_jobs(
            _jobs(batch, persons, garments),
            out_dir,
            TryOnOptions.from_preset(QualityPreset.FAST),
            backend_id=None,
            resume=False,
            jpeg_quality=85,
        )

        sheets = batch.build_contact_sheets(out_dir / "images", out_dir)
        assert len(sheets) == 1
        with Image.open(sheets[0]) as sheet:
            assert sheet.width > 400
            assert sheet.height > 280


def _main() -> int:
    """Run every ``test_*`` in this module."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[tuple[str, BaseException]] = []

    for name, func in tests:
        try:
            func()
        except BaseException as exc:  # noqa: BLE001 - this *is* the runner
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from ai_trial_room.utils.logging_setup import setup_logging

    setup_logging("ERROR")
    raise SystemExit(_main())
