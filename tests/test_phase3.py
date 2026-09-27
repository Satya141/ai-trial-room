"""Tests for the Phase 3 LoRA training layer.

Covers dataset scanning and validation, caption generation, flow-matching maths,
training-config hardware guards, and LoRA discovery and loading - all without a
GPU or any model weights.

The flow-matching tests matter most: training a rectified-flow model with a DDPM
objective is a silent, expensive mistake, so the interpolation and velocity
target are pinned numerically.

Run standalone::

    python tests/test_phase3.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.config import BackendId, Category, DrapeStyle  # noqa: E402
from ai_trial_room.lora import (  # noqa: E402
    NO_LORA_LABEL,
    LoraSpec,
    apply_trigger,
    discover_loras,
    find_lora,
)
from ai_trial_room.training.caption import (  # noqa: E402
    DEFAULT_TRIGGER,
    CaptionStrategy,
    caption_dataset,
    caption_stats,
    template_caption,
)
from ai_trial_room.training.dataset import (  # noqa: E402
    GOOD_SAMPLE_COUNT,
    MIN_RECOMMENDED_SAMPLES,
    DatasetMode,
    DrapeLoraDataset,
    PairedLoraDataset,
    build_dataset,
    detect_mode,
    parse_drape_from_name,
    scan_dataset,
    scan_drape_dataset,
    scan_paired_dataset,
)
from ai_trial_room.training.train_lora import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    QWEN_MIN_VRAM_GB,
    TrainingConfig,
    TrainingState,
    default_config,
    flow_match_inputs,
    lr_at_step,
    sample_timesteps,
)
from ai_trial_room.utils.errors import InvalidInputError  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_drape_dataset(
    root: Path,
    counts: dict[str, int] | None = None,
    *,
    captions: bool = True,
) -> Path:
    """Write a drape-layout dataset with the given per-style image counts."""
    counts = counts or {"nivi": 3, "bengali": 2}
    folder = root / "saree_drapes"
    folder.mkdir(parents=True, exist_ok=True)

    for style, count in counts.items():
        for index in range(1, count + 1):
            path = folder / f"{style}_{index:03d}.jpg"
            Image.new("RGB", (640, 900), (80, 50, 120)).save(path)
            if captions:
                path.with_suffix(".txt").write_text(
                    template_caption(parse_drape_from_name(path)), encoding="utf-8"
                )
    return folder


def _make_paired_dataset(root: Path, count: int = 3, *, orphans: int = 0) -> Path:
    """Write a paired-layout dataset with optional unmatched condition images."""
    folder = root / "saree_paired"
    for name in ("condition", "target", "garment", "prompts"):
        (folder / name).mkdir(parents=True, exist_ok=True)

    for index in range(1, count + 1):
        stem = f"{index:04d}"
        Image.new("RGB", (600, 900), (200, 180, 170)).save(folder / "condition" / f"{stem}.jpg")
        Image.new("RGB", (600, 900), (180, 40, 60)).save(folder / "target" / f"{stem}.png")
        Image.new("RGB", (400, 600), (180, 40, 60)).save(folder / "garment" / f"{stem}.jpg")

    (folder / "prompts" / "0001.txt").write_text("Dress her in this saree.", encoding="utf-8")

    for index in range(orphans):
        Image.new("RGB", (600, 900), (1, 1, 1)).save(
            folder / "condition" / f"9{index:03d}.jpg"
        )
    return folder


# --------------------------------------------------------------------------- #
# Dataset scanning
# --------------------------------------------------------------------------- #


def test_drape_style_parsed_from_filename() -> None:
    """The filename prefix drives the caption vocabulary."""
    assert parse_drape_from_name(Path("nivi_001.jpg")) is DrapeStyle.NIVI
    assert parse_drape_from_name(Path("GUJARATI_042.png")) is DrapeStyle.GUJARATI
    assert parse_drape_from_name(Path("mystery_001.jpg")) is None
    assert parse_drape_from_name(Path("nounderscore.jpg")) is None


def test_scan_drape_dataset_finds_images_and_styles() -> None:
    """Scanning reports usable samples and a per-style breakdown."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 3, "bengali": 2, "nauvari": 1})
        report = scan_drape_dataset(folder)

    assert report.ok
    assert report.count == 6
    assert report.style_counts == {"nivi": 3, "bengali": 2, "nauvari": 1}
    assert all(sample.caption for sample in report.drape_samples)


def test_uncaptioned_images_are_a_hard_error() -> None:
    """Training on captionless images would silently learn nothing useful."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 2}, captions=False)
        report = scan_drape_dataset(folder)

    assert not report.ok
    assert any("no caption" in error for error in report.errors)
    assert any("caption_dataset.py" in error for error in report.errors)


def test_empty_and_missing_directories_are_reported() -> None:
    """A missing or empty dataset fails with an actionable message."""
    with tempfile.TemporaryDirectory() as raw:
        empty = Path(raw) / "empty"
        empty.mkdir()
        report = scan_drape_dataset(empty)
        assert not report.ok
        assert any("No images found" in error for error in report.errors)

        missing = scan_drape_dataset(Path(raw) / "nope")
        assert not missing.ok


def test_small_dataset_warns_about_overfitting() -> None:
    """Below the recommended count, the scanner says so."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 3})
        report = scan_drape_dataset(folder)

    assert report.ok, "small is still usable"
    assert any(str(MIN_RECOMMENDED_SAMPLES) in warning for warning in report.warnings)
    assert any(str(GOOD_SAMPLE_COUNT) in warning for warning in report.warnings)


def test_style_imbalance_is_flagged() -> None:
    """A 10:1 style split will skew the LoRA, so it is warned about."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 20, "nauvari": 2})
        report = scan_drape_dataset(folder)

    assert any("imbalanced" in warning for warning in report.warnings)


def test_balanced_styles_are_not_flagged() -> None:
    """An even split must not produce an imbalance warning."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 5, "bengali": 5, "nauvari": 4})
        report = scan_drape_dataset(folder)

    assert not any("imbalanced" in warning for warning in report.warnings)


def test_paired_dataset_matches_on_stem_across_extensions() -> None:
    """condition/0001.jpg must pair with target/0001.png."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_paired_dataset(Path(raw), count=3)
        report = scan_paired_dataset(folder)

    assert report.ok
    assert report.count == 3
    assert all(sample.garment_path is not None for sample in report.paired_samples)
    assert report.paired_samples[0].instruction == "Dress her in this saree."
    # Two triplets have no prompt file.
    assert any("no prompt file" in warning for warning in report.warnings)


def test_paired_orphans_are_skipped_and_reported() -> None:
    """A condition image with no target cannot be trained on."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_paired_dataset(Path(raw), count=2, orphans=2)
        report = scan_paired_dataset(folder)

    assert report.count == 2
    assert any("no matching target" in warning for warning in report.warnings)


def test_paired_requires_condition_and_target_directories() -> None:
    """Missing required subdirectories fail with a message naming the layout."""
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw) / "broken"
        (folder / "condition").mkdir(parents=True)
        report = scan_paired_dataset(folder)

    assert not report.ok
    assert any("condition/ and target/" in error for error in report.errors)


def test_mode_auto_detection() -> None:
    """condition/ plus target/ means paired; anything else means drape."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        assert detect_mode(_make_paired_dataset(root)) is DatasetMode.PAIRED
        assert detect_mode(_make_drape_dataset(root)) is DatasetMode.DRAPE


def test_scan_dataset_dispatches_on_mode() -> None:
    """The generic entry point honours the requested mode."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw))
        assert scan_dataset(folder, DatasetMode.DRAPE).mode is DatasetMode.DRAPE


# --------------------------------------------------------------------------- #
# Torch datasets
# --------------------------------------------------------------------------- #


def test_drape_dataset_yields_normalised_tensors() -> None:
    """Images arrive as [-1, 1] CHW tensors at the requested resolution."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 2})
        report = scan_drape_dataset(folder)
        dataset = build_dataset(report, resolution=(256, 384), caption_dropout=0.0)

        assert isinstance(dataset, DrapeLoraDataset)
        assert len(dataset) == 2

        # Must stay inside the temp dir: the dataset reads lazily from disk.
        item = dataset[0]

    tensor = item["pixel_values"]
    assert tuple(tensor.shape) == (3, 384, 256), tensor.shape
    assert float(tensor.min()) >= -1.0001
    assert float(tensor.max()) <= 1.0001
    assert item["caption"]
    assert item["style"] == "nivi"


def test_drape_dataset_letterboxes_rather_than_crops() -> None:
    """A tall image keeps its full height, so a hem is never cut off."""
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw) / "d"
        folder.mkdir()
        path = folder / "nivi_001.jpg"
        Image.new("RGB", (300, 1200), (90, 40, 40)).save(path)
        path.with_suffix(".txt").write_text("aitrsaree, test.", encoding="utf-8")

        dataset = build_dataset(scan_drape_dataset(folder), resolution=(256, 384))
        tensor = dataset[0]["pixel_values"]

    assert tuple(tensor.shape) == (3, 384, 256)
    # White letterbox padding means columns at the far left are near +1.
    assert float(tensor[:, 192, 0].mean()) > 0.9


def test_caption_dropout_is_deterministic_for_a_seed() -> None:
    """Dropout must be reproducible, or a run cannot be repeated."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 8})
        report = scan_drape_dataset(folder)

        a = build_dataset(report, resolution=(128, 192), caption_dropout=0.5, seed=1)
        b = build_dataset(report, resolution=(128, 192), caption_dropout=0.5, seed=1)
        c = build_dataset(report, resolution=(128, 192), caption_dropout=0.5, seed=2)

        first = [a[i]["caption"] for i in range(8)]
        same = [b[i]["caption"] for i in range(8)]
        other = [c[i]["caption"] for i in range(8)]

    assert first == same, "same seed must give the same dropout pattern"
    assert first != other, "a different seed must change it"
    assert any(caption == "" for caption in first), "dropout should fire at 0.5"


def test_paired_dataset_yields_condition_target_and_garment() -> None:
    """An edit LoRA needs all three images plus the instruction."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_paired_dataset(Path(raw), count=2)
        dataset = build_dataset(
            scan_paired_dataset(folder), resolution=(128, 192), caption_dropout=0.0
        )
        assert isinstance(dataset, PairedLoraDataset)
        # Must stay inside the temp dir: the dataset reads lazily from disk.
        item = dataset[0]

    for key in ("condition_values", "target_values", "garment_values"):
        assert tuple(item[key].shape) == (3, 192, 128), key
    assert item["caption"]


def test_paired_missing_prompt_gets_a_default_instruction() -> None:
    """A triplet with no prompt file still trains, with a generic instruction."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_paired_dataset(Path(raw), count=3)
        dataset = build_dataset(
            scan_paired_dataset(folder), resolution=(64, 96), caption_dropout=0.0
        )
        captions = [dataset[i]["caption"] for i in range(len(dataset))]

    assert PairedLoraDataset.DEFAULT_INSTRUCTION in captions


def test_building_from_a_failed_report_raises() -> None:
    """An unusable dataset must fail loudly before any weights load."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 1}, captions=False)
        report = scan_drape_dataset(folder)
        try:
            build_dataset(report)
        except InvalidInputError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected InvalidInputError")


def test_empty_dataset_class_raises() -> None:
    """Constructing a dataset with no samples is a programming error."""
    for cls in (DrapeLoraDataset, PairedLoraDataset):
        try:
            cls([])
        except InvalidInputError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{cls.__name__} accepted an empty sample list")


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #


def test_template_caption_uses_the_inference_vocabulary() -> None:
    """Captions must share tokens with the generation prompts, or the LoRA
    learns a trigger that is never pulled."""
    caption = template_caption(DrapeStyle.GUJARATI)
    assert caption.startswith(DEFAULT_TRIGGER)
    for term in ("pallu", "blouse", "Gujarat", "right shoulder"):
        assert term in caption, term


def test_template_captions_differ_per_drape() -> None:
    """Four drapes, four captions."""
    captions = {style: template_caption(style) for style in DrapeStyle}
    assert len(set(captions.values())) == len(DrapeStyle)


def test_template_caption_handles_unknown_style_and_other_categories() -> None:
    """A missing style still yields a usable generic caption."""
    generic = template_caption(None)
    assert DEFAULT_TRIGGER in generic and "saree" in generic

    lehenga = template_caption(None, category=Category.LEHENGA)
    assert "ghagra" in lehenga and "choli" in lehenga

    kurti = template_caption(None, category=Category.KURTI)
    assert "kurti" in kurti


def test_template_caption_inserts_vlm_detail_and_custom_trigger() -> None:
    """A VLM description is spliced in, and the trigger is configurable."""
    caption = template_caption(
        DrapeStyle.NIVI, trigger="myshop", detail="deep red silk with a gold zari border"
    )
    assert caption.startswith("myshop")
    assert "gold zari border" in caption


def test_caption_dataset_writes_skips_and_overwrites() -> None:
    """Captioning is idempotent unless --overwrite is passed."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 3}, captions=False)

        first = caption_dataset(folder, strategy=CaptionStrategy.TEMPLATE)
        assert len(first.written) == 3
        assert not first.skipped

        second = caption_dataset(folder, strategy=CaptionStrategy.TEMPLATE)
        assert not second.written
        assert len(second.skipped) == 3

        third = caption_dataset(folder, strategy=CaptionStrategy.TEMPLATE, overwrite=True)
        assert len(third.written) == 3


def test_caption_dataset_dry_run_writes_nothing() -> None:
    """--dry-run must not touch disk."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 2}, captions=False)
        result = caption_dataset(folder, dry_run=True)

        assert len(result.written) == 2
        assert not any(path.exists() for path in result.written)


def test_caption_dataset_reports_untagged_filenames() -> None:
    """An unrecognised prefix is reported so it can be renamed."""
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw) / "d"
        folder.mkdir()
        Image.new("RGB", (400, 600)).save(folder / "mystery_001.jpg")
        Image.new("RGB", (400, 600)).save(folder / "nivi_001.jpg")

        result = caption_dataset(folder)

    assert len(result.untagged) == 1
    assert result.untagged[0].name == "mystery_001.jpg"
    assert "Rename them" in result.render()


def test_default_style_covers_untagged_images() -> None:
    """--default-style lets an untagged image still get drape vocabulary."""
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw) / "d"
        folder.mkdir()
        path = folder / "photo_001.jpg"
        Image.new("RGB", (400, 600)).save(path)

        caption_dataset(folder, default_style=DrapeStyle.NAUVARI)
        caption = path.with_suffix(".txt").read_text(encoding="utf-8")

    assert "Nauvari" in caption


def test_caption_stats_detect_low_variety() -> None:
    """Identical captions give the LoRA nothing to discriminate on."""
    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 6})
        report = scan_drape_dataset(folder)

    stats = caption_stats(report.drape_samples)
    assert stats["trigger_coverage"] == 1.0
    assert stats["unique_ratio"] < 0.5, "identical template captions expected"
    assert stats["mean_words"] > 20

    assert caption_stats([]) == {}


# --------------------------------------------------------------------------- #
# Flow matching - the maths most expensive to get wrong
# --------------------------------------------------------------------------- #


def test_timesteps_stay_strictly_inside_the_unit_interval() -> None:
    """t=0 or t=1 gives a degenerate target, so both are clamped away."""
    for strategy in ("uniform", "logit_normal"):
        t = sample_timesteps(512, strategy=strategy, logit_mean=0.0, logit_std=1.0)
        assert float(t.min()) > 0.0
        assert float(t.max()) < 1.0
        assert tuple(t.shape) == (512,)


def test_logit_normal_concentrates_in_the_middle() -> None:
    """Logit-normal should put more mass mid-schedule than uniform does."""
    import torch

    torch.manual_seed(0)
    uniform = sample_timesteps(4096, strategy="uniform", logit_mean=0.0, logit_std=1.0)
    logit = sample_timesteps(4096, strategy="logit_normal", logit_mean=0.0, logit_std=1.0)

    def mid_fraction(t: Any) -> float:
        return float(((t > 0.3) & (t < 0.7)).float().mean())

    assert mid_fraction(logit) > mid_fraction(uniform)


def test_flow_match_interpolation_is_exact() -> None:
    """xt = (1-t)*x0 + t*x1 and target = x1 - x0, verified against the noise."""
    import torch

    torch.manual_seed(0)
    latents = torch.randn(2, 4, 8, 8)

    # t=1 must return the clean latent exactly and target = x1 - x0.
    ones = torch.ones(2)
    noisy, target = flow_match_inputs(latents, ones)
    assert torch.allclose(noisy, latents, atol=1e-5)

    # Recover x0 from the target, then confirm the interpolation at t=1.
    noise = latents - target
    assert torch.allclose(noisy, (1 - 1.0) * noise + 1.0 * latents, atol=1e-5)

    # t=0 must return pure noise, which is x1 - target.
    zeros = torch.zeros(2)
    noisy0, target0 = flow_match_inputs(latents, zeros)
    assert torch.allclose(noisy0, latents - target0, atol=1e-5)


def test_flow_match_target_is_velocity_not_noise() -> None:
    """The target must be ``x1 - x0``, not ``x0``.

    Training a rectified-flow model on an epsilon target is a silent, expensive
    mistake, so this is pinned explicitly.
    """
    import torch

    torch.manual_seed(1)
    latents = torch.randn(1, 4, 4, 4)
    t = torch.full((1,), 0.5)

    noisy, target = flow_match_inputs(latents, t)
    noise = latents - target

    assert torch.allclose(noisy, 0.5 * noise + 0.5 * latents, atol=1e-5)
    assert not torch.allclose(target, noise, atol=1e-2), "target must not be the noise"


def test_flow_match_broadcasts_over_any_rank() -> None:
    """Timesteps must broadcast for both 4-D and 3-D latents."""
    import torch

    for shape in ((3, 4, 8, 8), (3, 16, 64)):
        latents = torch.randn(*shape)
        noisy, target = flow_match_inputs(latents, torch.rand(3).clamp(0.1, 0.9))
        assert noisy.shape == latents.shape
        assert target.shape == latents.shape


def test_lr_schedule_warms_up_then_decays() -> None:
    """Linear warmup into a cosine decay, ending near zero."""
    config = TrainingConfig(
        dataset_dir=Path("."), output_dir=Path("."),
        learning_rate=1e-4, warmup_steps=100, max_steps=1000, lr_schedule="cosine",
    )

    assert lr_at_step(0, config) < config.learning_rate
    assert lr_at_step(99, config) == config.learning_rate
    assert lr_at_step(550, config) < config.learning_rate
    assert lr_at_step(999, config) < lr_at_step(550, config)
    assert lr_at_step(999, config) >= 0.0

    # Warmup must increase monotonically.
    warmup = [lr_at_step(step, config) for step in range(0, 100, 10)]
    assert warmup == sorted(warmup)


def test_lr_schedule_alternatives() -> None:
    """constant and linear schedules behave as named."""
    base = {"dataset_dir": Path("."), "output_dir": Path("."), "warmup_steps": 0, "max_steps": 100}

    constant = TrainingConfig(**base, learning_rate=1e-4, lr_schedule="constant")
    assert lr_at_step(50, constant) == 1e-4

    linear = TrainingConfig(**base, learning_rate=1e-4, lr_schedule="linear")
    assert abs(lr_at_step(50, linear) - 5e-5) < 1e-6


# --------------------------------------------------------------------------- #
# Training config guards
# --------------------------------------------------------------------------- #


def test_config_rejects_impossible_settings() -> None:
    """Obviously wrong hyperparameters fail before any weights load."""
    for kwargs in (
        {"rank": 0},
        {"rank": 1000},
        {"batch_size": 0},
        {"gradient_accumulation": 0},
        {"max_steps": 0},
    ):
        config = TrainingConfig(dataset_dir=Path("."), output_dir=Path("."), **kwargs)
        try:
            config.validate()
        except InvalidInputError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{kwargs} should have been rejected")


def test_config_warns_about_the_12b_model_on_a_small_card() -> None:
    """Pointing the 12 B target at a 16 GB card must warn, not silently fail."""
    import ai_trial_room.training.train_lora as module
    from ai_trial_room.utils.device import Hardware

    original = module.detect_hardware
    try:
        module.detect_hardware = lambda: Hardware(  # type: ignore[assignment]
            device="cuda", name="Tesla T4", total_vram_gb=15.8,
            supports_bf16=False, compute_capability=(7, 5),
        )
        config = TrainingConfig(
            dataset_dir=Path("."), output_dir=Path("."), backend=BackendId.QWEN_EDIT
        )
        warnings = config.validate()
    finally:
        module.detect_hardware = original

    joined = " ".join(warnings)
    assert "12 B" in joined or "12 B" in joined.replace("12B", "12 B")
    assert "flux_klein" in joined, "should point at the tractable alternative"
    assert "bf16" in joined, "Turing has no usable bf16 - worth saying"


def test_config_is_quiet_on_adequate_hardware() -> None:
    """A 24 GB Ampere card training the 12 B model needs no warnings."""
    import ai_trial_room.training.train_lora as module
    from ai_trial_room.utils.device import Hardware

    original = module.detect_hardware
    try:
        module.detect_hardware = lambda: Hardware(  # type: ignore[assignment]
            device="cuda", name="RTX 4090", total_vram_gb=24.0,
            supports_bf16=True, compute_capability=(8, 9),
        )
        config = TrainingConfig(
            dataset_dir=Path("."), output_dir=Path("."), backend=BackendId.QWEN_EDIT
        )
        warnings = config.validate()
    finally:
        module.detect_hardware = original

    assert not warnings, warnings


def test_default_config_picks_the_tractable_backend_for_small_cards() -> None:
    """Hardware detection should choose 4 B on a T4 and 12 B on a 24 GB card."""
    import ai_trial_room.training.train_lora as module
    from ai_trial_room.utils.device import Hardware

    original = module.detect_hardware
    try:
        module.detect_hardware = lambda: Hardware(  # type: ignore[assignment]
            device="cuda", name="Tesla T4", total_vram_gb=15.8,
            supports_bf16=False, compute_capability=(7, 5),
        )
        t4 = default_config(Path("d"), Path("o"))

        module.detect_hardware = lambda: Hardware(  # type: ignore[assignment]
            device="cuda", name="RTX 4090", total_vram_gb=24.0,
            supports_bf16=True, compute_capability=(8, 9),
        )
        big = default_config(Path("d"), Path("o"))

        module.detect_hardware = lambda: Hardware(  # type: ignore[assignment]
            device="cuda", name="RTX 5050 Laptop", total_vram_gb=8.0,
            supports_bf16=True, compute_capability=(12, 0),
        )
        small = default_config(Path("d"), Path("o"))
    finally:
        module.detect_hardware = original

    assert t4.backend is BackendId.FLUX_KLEIN
    assert big.backend is BackendId.QWEN_EDIT
    assert small.backend is BackendId.FLUX_KLEIN
    assert small.quantize_base == "4bit"
    assert max(small.resolution) < max(t4.resolution), "8 GB should drop resolution"
    assert QWEN_MIN_VRAM_GB > 15.8, "the guard must exclude a T4"


def test_config_serialises_for_the_sidecar() -> None:
    """A LoRA must carry the recipe that produced it."""
    config = TrainingConfig(
        dataset_dir=Path("datasets/x"), output_dir=Path("loras/y"), rank=32
    )
    payload = json.loads(config.to_json())

    assert payload["rank"] == 32
    assert payload["mode"] == DatasetMode.DRAPE.value
    assert payload["backend"] == BackendId.FLUX_KLEIN.value
    assert payload["resolution"] == [768, 1024]
    assert isinstance(payload["target_modules"], list)
    assert "to_q" in payload["target_modules"]


def test_default_target_modules_are_attention_only() -> None:
    """Feed-forward layers triple adapter size for little gain on a style LoRA."""
    assert "to_q" in DEFAULT_TARGET_MODULES
    assert "to_out.0" in DEFAULT_TARGET_MODULES
    assert not any("ff" in name or "mlp" in name for name in DEFAULT_TARGET_MODULES)


def test_training_state_tracks_recent_loss() -> None:
    """The log shows a rolling mean, which needs to handle an empty history."""
    state = TrainingState()
    assert state.recent_loss() != state.recent_loss()  # nan

    state.loss_history.extend([1.0, 0.5, 0.25, 0.25])
    assert abs(state.recent_loss(window=2) - 0.25) < 1e-9
    assert abs(state.recent_loss(window=4) - 0.5) < 1e-9
    assert state.elapsed_s >= 0.0


# --------------------------------------------------------------------------- #
# LoRA discovery and loading
# --------------------------------------------------------------------------- #


def _write_lora(root: Path, name: str, **sidecar: Any) -> Path:
    """Create a fake adapter directory with weights and a sidecar."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

    payload = {
        "name": name,
        "backend": BackendId.QWEN_EDIT.value,
        "trigger": DEFAULT_TRIGGER,
        "rank": 16,
        "steps_trained": 1200,
        "base_repo": "Qwen/Qwen-Image-Edit-2511",
        "base_license": "Apache-2.0",
    }
    payload.update(sidecar)
    (folder / "aitr_lora.json").write_text(json.dumps(payload), encoding="utf-8")
    return folder


def test_discovery_finds_adapters_and_reads_metadata() -> None:
    """A trained adapter is self-describing via its sidecar."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_lora(root, "saree-drape-v1")
        _write_lora(root, "lehenga-v2", backend=BackendId.FLUX_KLEIN.value, rank=32)

        found = discover_loras(root)

    assert [spec.name for spec in found] == ["lehenga-v2", "saree-drape-v1"]
    by_name = {spec.name: spec for spec in found}
    assert by_name["saree-drape-v1"].backend is BackendId.QWEN_EDIT
    assert by_name["lehenga-v2"].backend is BackendId.FLUX_KLEIN
    assert by_name["lehenga-v2"].rank == 32
    assert not by_name["saree-drape-v1"].unverified
    assert "1200 steps" in by_name["saree-drape-v1"].describe()


def test_discovery_skips_checkpoints_and_weightless_directories() -> None:
    """Only final adapters are offered, not mid-run checkpoints."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_lora(root, "good")
        _write_lora(root, "step-000400")
        (root / "notes").mkdir()

        found = discover_loras(root)

    assert [spec.name for spec in found] == ["good"]


def test_adapter_without_a_sidecar_is_unverified_but_usable() -> None:
    """A third-party LoRA should still load, with a warning."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        folder = root / "mystery"
        folder.mkdir()
        (folder / "adapter_model.safetensors").write_bytes(b"\x00" * 16)

        found = discover_loras(root)

    assert len(found) == 1
    assert found[0].unverified
    assert "[unverified]" in found[0].label
    # Unverified adapters are permitted on any backend.
    assert found[0].is_compatible(BackendId.FLUX_KLEIN)
    assert found[0].is_compatible(BackendId.QWEN_EDIT)


def test_malformed_sidecar_degrades_to_unverified() -> None:
    """Broken JSON must not crash discovery."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        folder = root / "broken"
        folder.mkdir()
        (folder / "adapter_model.safetensors").write_bytes(b"\x00")
        (folder / "aitr_lora.json").write_text("{not json", encoding="utf-8")

        found = discover_loras(root)

    assert len(found) == 1
    assert found[0].unverified


def test_unknown_backend_in_sidecar_degrades_to_unverified() -> None:
    """A sidecar naming a backend we do not have must not crash."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_lora(root, "future", backend="some_future_model")
        found = discover_loras(root)

    assert len(found) == 1
    assert found[0].unverified


def test_backend_mismatch_is_incompatible() -> None:
    """A FLUX adapter must not be offered for Qwen - it would produce noise."""
    spec = LoraSpec(
        name="flux-only", path=Path("."), backend=BackendId.FLUX_KLEIN, rank=16
    )
    assert spec.is_compatible(BackendId.FLUX_KLEIN)
    assert not spec.is_compatible(BackendId.QWEN_EDIT)


def test_find_lora_by_name_and_label() -> None:
    """Lookup works from the bare name and from the dropdown label."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        _write_lora(root, "saree-v1")

        assert find_lora("saree-v1", root) is not None
        assert find_lora("saree-v1  [unverified]", root) is not None
        assert find_lora("nope", root) is None
        assert find_lora(NO_LORA_LABEL, root) is None
        assert find_lora("", root) is None


def test_discovery_of_a_missing_directory_is_empty() -> None:
    """No loras/ directory simply means no adapters, not an error."""
    assert discover_loras(Path("definitely/not/here")) == []


def test_apply_trigger_inserts_once_and_only_when_needed() -> None:
    """The trigger must be present, but never duplicated."""
    spec = LoraSpec(
        name="x", path=Path("."), backend=BackendId.QWEN_EDIT, trigger="aitrsaree"
    )

    prompt = "Dress the person in the saree."
    triggered = apply_trigger(prompt, spec)
    assert triggered.startswith("aitrsaree, ")

    # Idempotent, and case-insensitive about what counts as already present.
    assert apply_trigger(triggered, spec) == triggered
    assert apply_trigger("AITRSAREE, something", spec) == "AITRSAREE, something"

    # No spec or no trigger means no change.
    assert apply_trigger(prompt, None) == prompt
    bare = LoraSpec(name="y", path=Path("."), backend=BackendId.QWEN_EDIT, trigger="")
    assert apply_trigger(prompt, bare) == prompt


def test_lora_choices_always_offers_the_base_model_first() -> None:
    """The safe default must be first in the dropdown."""
    from ai_trial_room.lora import lora_choices

    choices = lora_choices()
    assert choices[0] == NO_LORA_LABEL


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def _load_script(name: str) -> Any:
    """Import a file in ``scripts/`` as a module."""
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"script_{name.replace('.', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_caption_cli_parses() -> None:
    """The captioning CLI exposes the documented flags."""
    cli = _load_script("caption_dataset.py")
    args = cli.parse_args(
        ["--dataset", "d", "--strategy", "vlm", "--default-style", "nauvari", "--overwrite"]
    )
    assert args.strategy == "vlm"
    assert args.default_style == "nauvari"
    assert args.overwrite


def test_train_cli_builds_a_config_from_flags() -> None:
    """Explicit flags must override the hardware-derived defaults."""
    cli = _load_script("train_saree_lora.py")
    args = cli.parse_args(
        [
            "--dataset", "datasets/x", "--name", "my-lora",
            "--backend", "qwen_edit", "--rank", "8", "--steps", "500",
            "--lr", "5e-5", "--grad-accum", "16", "--trigger", "myshop",
            "--no-gradient-checkpointing",
        ]
    )
    config = cli.build_config(args)

    assert config.backend is BackendId.QWEN_EDIT
    assert config.rank == 8
    assert config.alpha == 8, "alpha should follow rank when not set explicitly"
    assert config.max_steps == 500
    assert config.learning_rate == 5e-5
    assert config.gradient_accumulation == 16
    assert config.trigger == "myshop"
    assert not config.gradient_checkpointing
    assert config.output_dir.name == "my-lora"


def test_train_cli_dry_run_on_a_real_dataset() -> None:
    """--dry-run validates and reports without loading any weights."""
    cli = _load_script("train_saree_lora.py")

    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 3, "bengali": 3})
        code = cli.main(["--dataset", str(folder), "--name", "t", "--dry-run"])

    assert code == 0


def test_train_cli_fails_on_an_uncaptioned_dataset() -> None:
    """A dataset that cannot train must exit non-zero, even in a dry run."""
    cli = _load_script("train_saree_lora.py")

    with tempfile.TemporaryDirectory() as raw:
        folder = _make_drape_dataset(Path(raw), {"nivi": 2}, captions=False)
        code = cli.main(["--dataset", str(folder), "--dry-run"])

    assert code == 1


def test_train_cli_rejects_a_missing_dataset() -> None:
    """A bad path exits 2 rather than raising."""
    cli = _load_script("train_saree_lora.py")
    assert cli.main(["--dataset", "definitely/not/here", "--dry-run"]) == 2


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


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
