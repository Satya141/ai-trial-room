"""Tests for the Phase 2 quality layer.

Covers the four things Phase 2 added, all without a GPU or model weights:

* drape-specific negative prompts and dupatta styles
* garment sub-region geometry (pallu, pleats, blouse)
* Laplacian identity blending and region tone matching
* output validation and catalogue batch mode

Run standalone::

    python tests/test_phase2.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.backends.base import TryOnOptions  # noqa: E402
from ai_trial_room.backends.prompts import (  # noqa: E402
    BASE_NEGATIVE,
    DRAPE_SPECS,
    build_negative_prompt,
    build_prompt,
    build_refine_prompt,
)
from ai_trial_room.config import (  # noqa: E402
    Category,
    DrapeStyle,
    DupattaStyle,
    QualityPreset,
    ValidationConfig,
)
from ai_trial_room.postprocessing.blend import (  # noqa: E402
    _blur5,
    _downsample,
    _upsample,
    laplacian_blend,
    match_region_tone,
)
from ai_trial_room.postprocessing.face_preserve import (  # noqa: E402
    FaceRegion,
    build_face_mask,
)
from ai_trial_room.postprocessing.validate import (  # noqa: E402
    Severity,
    check_background_preserved,
    check_exposure,
    check_framing_note,
    check_garment_changed,
    validate_result,
)
from ai_trial_room.preprocessing.person import (  # noqa: E402
    AtrLabel,
    ParseResult,
    PoseLandmark,
    PoseResult,
)
from ai_trial_room.preprocessing.regions import (  # noqa: E402
    GarmentRegion,
    derive_regions,
    restrict_to_mask,
)

WIDTH, HEIGHT = 768, 1024


def _pose() -> PoseResult:
    """Standing front-facing full-length pose, same fixture as test_core."""
    landmarks = {
        PoseLandmark.NOSE: (384.0, 120.0, 0.99),
        PoseLandmark.LEFT_SHOULDER: (300.0, 260.0, 0.97),
        PoseLandmark.RIGHT_SHOULDER: (468.0, 260.0, 0.93),
        PoseLandmark.LEFT_HIP: (320.0, 520.0, 0.90),
        PoseLandmark.RIGHT_HIP: (450.0, 520.0, 0.88),
        PoseLandmark.LEFT_KNEE: (322.0, 740.0, 0.80),
        PoseLandmark.LEFT_ANKLE: (324.0, 950.0, 0.70),
    }
    return PoseResult(landmarks=landmarks, image_size=(WIDTH, HEIGHT), visible_ratio=0.8)


def _parse() -> ParseResult:
    """Person in a top and jeans, face and hair labelled."""
    labels = np.zeros((HEIGHT, WIDTH), np.uint8)
    labels[240:520, 290:480] = int(AtrLabel.UPPER_CLOTHES)
    labels[520:900, 300:470] = int(AtrLabel.PANTS)
    labels[90:200, 340:430] = int(AtrLabel.FACE)
    labels[60:120, 330:440] = int(AtrLabel.HAIR)
    return ParseResult(label_map=labels, available=True)


# --------------------------------------------------------------------------- #
# Negative prompts
# --------------------------------------------------------------------------- #


def test_negative_prompt_layers_base_category_and_drape() -> None:
    """All three layers must appear, most specific last."""
    negative = build_negative_prompt(Category.SAREE, drape_style=DrapeStyle.NAUVARI)
    assert "different face" in negative, "base layer missing"
    assert "jeans or trousers visible under the saree" in negative, "category layer missing"
    assert "straight skirt" in negative, "drape layer missing"
    assert len(negative) > len(BASE_NEGATIVE)


def test_each_drape_forbids_the_wrong_shoulder() -> None:
    """The negative must name the *opposite* shoulder to the one it wants."""
    gujarati = build_negative_prompt(Category.SAREE, drape_style=DrapeStyle.GUJARATI)
    assert DRAPE_SPECS[DrapeStyle.GUJARATI].shoulder == "right"
    assert "pallu over the left shoulder" in gujarati

    nivi = build_negative_prompt(Category.SAREE, drape_style=DrapeStyle.NIVI)
    assert DRAPE_SPECS[DrapeStyle.NIVI].shoulder == "left"
    assert "pallu over the right shoulder" in nivi


def test_drape_negatives_are_all_distinct() -> None:
    """Four drapes, four different negative prompts."""
    negatives = {
        style: build_negative_prompt(Category.SAREE, drape_style=style)
        for style in DrapeStyle
    }
    assert len(set(negatives.values())) == len(DrapeStyle)


def test_nauvari_forbids_collapsing_into_a_skirt() -> None:
    """Nauvari's failure mode is a joined-leg skirt; the negative must say so."""
    negative = build_negative_prompt(Category.SAREE, drape_style=DrapeStyle.NAUVARI)
    for term in ("straight skirt", "legs joined together", "lehenga"):
        assert term in negative, term


def test_every_category_has_its_own_confusions_listed() -> None:
    """Each category names the garments it tends to be confused with."""
    for category in Category:
        negative = build_negative_prompt(category)
        assert len(negative) > len(BASE_NEGATIVE) + 40, category


def test_non_saree_categories_ignore_drape_style() -> None:
    """Passing a drape for a kurti must not leak saree terms into the negative."""
    a = build_negative_prompt(Category.KURTI, drape_style=DrapeStyle.NAUVARI)
    b = build_negative_prompt(Category.KURTI, drape_style=DrapeStyle.NIVI)
    assert a == b


# --------------------------------------------------------------------------- #
# Dupatta styles and refinement prompts
# --------------------------------------------------------------------------- #


def test_dupatta_styles_produce_distinct_lehenga_prompts() -> None:
    """Each dupatta style must change the prompt text."""
    prompts = {
        style: build_prompt(Category.LEHENGA, dupatta_style=style, dominant_shoulder="left")
        for style in DupattaStyle
    }
    assert len(set(prompts.values())) == len(DupattaStyle)
    assert "bridal veil" in prompts[DupattaStyle.OVER_HEAD]
    assert "both shoulders" in prompts[DupattaStyle.BOTH_SHOULDERS]
    assert "forearms" in prompts[DupattaStyle.ARM_DRAPE]


def test_dupatta_label_round_trip() -> None:
    """UI labels resolve back to enum members."""
    for style in DupattaStyle:
        assert DupattaStyle.from_label(style.label) is style


def test_refine_prompt_is_detail_only_and_drape_aware() -> None:
    """The refine instruction must forbid changing the drape, only the detail."""
    prompt = build_refine_prompt(
        Category.SAREE, "pallu / dupatta", drape_style=DrapeStyle.GUJARATI
    )
    assert "change nothing except the fidelity of the fabric detail" in prompt
    assert "spread wide and flat across the chest" in prompt

    other = build_refine_prompt(
        Category.SAREE, "pallu / dupatta", drape_style=DrapeStyle.NIVI
    )
    assert other != prompt, "refine focus must follow the drape"


# --------------------------------------------------------------------------- #
# Quality presets
# --------------------------------------------------------------------------- #


def test_presets_increase_in_cost() -> None:
    """Fast < Balanced < Best in steps, and only Best refines."""
    fast = QualityPreset.FAST.settings()
    balanced = QualityPreset.BALANCED.settings()
    best = QualityPreset.BEST.settings()

    assert fast.steps < balanced.steps < best.steps
    assert not fast.refine and not balanced.refine
    assert best.refine and best.refine_steps > 0


def test_options_from_preset_and_override_semantics() -> None:
    """Explicit values win; ``None`` leaves the preset value in place."""
    options = TryOnOptions.from_preset(QualityPreset.BEST)
    assert options.steps == 40
    assert options.refine

    overridden = TryOnOptions.from_preset(
        QualityPreset.BEST, steps=None, seed=99, refine_strength=0.2
    )
    assert overridden.steps == 40, "None must not clobber the preset"
    assert overridden.seed == 99
    assert overridden.refine_strength == 0.2


def test_preset_label_round_trip() -> None:
    """Radio labels resolve back to enum members."""
    for preset in QualityPreset:
        assert QualityPreset.from_label(preset.label) is preset


# --------------------------------------------------------------------------- #
# Region geometry
# --------------------------------------------------------------------------- #


def test_saree_regions_include_pallu_and_pleats() -> None:
    """A saree yields pallu, pleats and blouse regions with real coverage."""
    regions = derive_regions(_pose(), _parse(), Category.SAREE, drape_style=DrapeStyle.NIVI)

    for region in (GarmentRegion.PALLU, GarmentRegion.PLEATS, GarmentRegion.BLOUSE):
        assert regions.get(region) is not None, region
        assert regions.coverage(region) > 0.005, (region, regions.coverage(region))


def test_kurti_has_no_pallu_or_pleats() -> None:
    """Non-draped categories must not produce drape-only regions."""
    regions = derive_regions(_pose(), _parse(), Category.KURTI)
    assert regions.get(GarmentRegion.PALLU) is None
    assert regions.get(GarmentRegion.PLEATS) is None
    assert regions.get(GarmentRegion.BLOUSE) is not None


def test_pallu_sits_on_the_requested_shoulder() -> None:
    """Left-shoulder and right-shoulder pallus must land on opposite sides."""
    left = derive_regions(
        _pose(), _parse(), Category.SAREE, drape_style=DrapeStyle.NIVI, pallu_shoulder="left"
    ).get(GarmentRegion.PALLU)
    right = derive_regions(
        _pose(), _parse(), Category.SAREE, drape_style=DrapeStyle.BENGALI, pallu_shoulder="right"
    ).get(GarmentRegion.PALLU)

    assert left is not None and right is not None

    # Compare mass either side of the body midline at shoulder height.
    # int64 is required: numpy sums uint8 into an *unsigned* accumulator, so a
    # negative difference would wrap to a huge positive number.
    midline = 384
    band = slice(250, 330)
    left_arr = np.asarray(left, dtype=np.int64)[band]
    right_arr = np.asarray(right, dtype=np.int64)[band]

    left_bias = left_arr[:, :midline].sum() - left_arr[:, midline:].sum()
    right_bias = right_arr[:, :midline].sum() - right_arr[:, midline:].sum()

    assert left_bias > 0, "left-shoulder pallu should weight the left of frame"
    assert right_bias < 0, "right-shoulder pallu should weight the right of frame"


def test_gujarati_pallu_is_a_wide_chest_panel() -> None:
    """Seedha pallu is spread across the chest, so it is wider than a Nivi band."""
    nivi = derive_regions(
        _pose(), _parse(), Category.SAREE, drape_style=DrapeStyle.NIVI
    )
    gujarati = derive_regions(
        _pose(), _parse(), Category.SAREE, drape_style=DrapeStyle.GUJARATI
    )

    chest = slice(270, 400)
    nivi_width = (np.asarray(nivi.get(GarmentRegion.PALLU))[chest] > 128).sum(axis=1).mean()
    guj_width = (np.asarray(gujarati.get(GarmentRegion.PALLU))[chest] > 128).sum(axis=1).mean()

    assert guj_width > nivi_width * 1.2, (nivi_width, guj_width)


def test_pleats_flare_downward() -> None:
    """The pleat fan must be wider at the hem than at the waist."""
    regions = derive_regions(_pose(), _parse(), Category.SAREE)
    pleats = np.asarray(regions.get(GarmentRegion.PLEATS))

    waist = (pleats[540] > 128).sum()
    lower = (pleats[700] > 128).sum()
    assert lower > waist, (waist, lower)


def test_blouse_region_stops_above_the_waist() -> None:
    """A choli must not extend down to the hips."""
    regions = derive_regions(_pose(), _parse(), Category.SAREE)
    blouse = np.asarray(regions.get(GarmentRegion.BLOUSE))

    assert blouse[300, 384] > 128, "chest should be inside the blouse region"
    assert blouse[515, 384] < 128, "hip line should be outside the blouse region"


def test_region_usability_threshold() -> None:
    """A tiny region is reported as not worth a refinement pass."""
    regions = derive_regions(_pose(), _parse(), Category.SAREE)
    assert regions.is_usable(GarmentRegion.PALLU)

    sliver = Image.new("L", (WIDTH, HEIGHT), 0)
    sliver.putpixel((10, 10), 255)
    tiny = type(regions)(masks={GarmentRegion.PALLU: sliver}, image_size=(WIDTH, HEIGHT))
    assert not tiny.is_usable(GarmentRegion.PALLU)


def test_restrict_to_mask_clips_region_to_allowed_area() -> None:
    """A region cannot extend outside the inpaint mask."""
    region = Image.new("L", (64, 64), 255)
    allowed = Image.new("L", (64, 64), 0)
    allowed.paste(Image.new("L", (32, 64), 255), (0, 0))

    clipped = np.asarray(restrict_to_mask(region, allowed))
    assert clipped[32, 10] == 255
    assert clipped[32, 50] == 0


def test_regions_degrade_without_landmarks() -> None:
    """Missing hips must not raise - the region is simply absent."""
    bare = PoseResult(
        landmarks={PoseLandmark.NOSE: (384.0, 120.0, 0.99)},
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.2,
    )
    regions = derive_regions(bare, _parse(), Category.SAREE)
    assert regions.get(GarmentRegion.PALLU) is None
    assert regions.get(GarmentRegion.PLEATS) is None


# --------------------------------------------------------------------------- #
# Laplacian blending
# --------------------------------------------------------------------------- #


def test_binomial_blur_preserves_a_constant_field() -> None:
    """The 5-tap kernel must sum to 1, or the pyramid changes brightness."""
    const = np.full((32, 32, 3), 100.0, np.float32)
    assert abs(float(_blur5(const).mean()) - 100.0) < 1e-3


def test_pyramid_helpers_have_expected_shapes() -> None:
    """Downsample halves, upsample restores."""
    array = np.random.default_rng(0).random((64, 48, 3)).astype(np.float32)
    assert _blur5(array).shape == array.shape
    assert _downsample(array).shape == (32, 24, 3)
    assert _upsample(_downsample(array), (64, 48)).shape == (64, 48, 3)


def test_laplacian_blend_respects_the_mask() -> None:
    """Each side of a split mask takes its own source."""
    size = (128, 128)
    white = Image.new("RGB", size, (255, 255, 255))
    black = Image.new("RGB", size, (0, 0, 0))

    split = np.zeros(size[::-1], np.uint8)
    split[:, : size[0] // 2] = 255

    out = np.asarray(laplacian_blend(white, black, Image.fromarray(split, "L")), float)
    assert out[64, 10].mean() > 230
    assert out[64, -10].mean() < 25


def test_laplacian_blend_preserves_high_frequency_detail() -> None:
    """This is the whole point: texture must survive the blend, not cross-fade.

    A full mask should reconstruct the foreground almost exactly, which a naive
    blur-based blend would not.
    """
    size = (128, 128)
    texture = (np.random.default_rng(1).random((128, 128, 3)) * 255).astype(np.uint8)
    full = Image.fromarray(np.full(size[::-1], 255, np.uint8), "L")

    out = np.asarray(
        laplacian_blend(Image.fromarray(texture), Image.new("RGB", size), full), float
    )
    error = float(np.abs(out - texture.astype(float)).mean())
    assert error < 6.0, error


def test_laplacian_levels_are_clamped_for_small_images() -> None:
    """Asking for more levels than the image can support must not crash."""
    small = Image.new("RGB", (16, 16), (128, 128, 128))
    mask = Image.new("L", (16, 16), 255)
    out = laplacian_blend(small, Image.new("RGB", (16, 16)), mask, levels=12)
    assert out.size == (16, 16)


def test_match_region_tone_moves_toward_the_target() -> None:
    """The original face gets the generated frame's colour cast."""
    cool = Image.new("RGB", (64, 64), (150, 160, 200))
    warm = Image.new("RGB", (64, 64), (200, 150, 110))
    mask = Image.new("L", (64, 64), 255)

    fixed = np.asarray(match_region_tone(cool, warm, mask, strength=1.0), float)[32, 32]
    assert fixed[0] > 150, "red should rise toward the warm target"
    assert fixed[2] < 200, "blue should fall toward the warm target"


def test_match_region_tone_is_a_no_op_for_a_tiny_mask() -> None:
    """Too few pixels means no reliable statistic, so leave the image alone."""
    source = Image.new("RGB", (64, 64), (150, 160, 200))
    mask = Image.new("L", (64, 64), 0)
    mask.putpixel((1, 1), 255)
    assert match_region_tone(source, Image.new("RGB", (64, 64)), mask) is source


def test_identity_strength_scales_the_face_mask_peak() -> None:
    """Lower identity strength must lower the blend opacity."""
    region = FaceRegion(
        polygon=[(340, 100), (430, 100), (430, 200), (340, 200)],
        image_size=(WIDTH, HEIGHT),
    )
    full = np.asarray(build_face_mask(region, strength=1.0)).max()
    half = np.asarray(build_face_mask(region, strength=0.5)).max()
    assert full > 240
    assert 110 < half < 145


def test_face_mask_expands_up_more_than_down() -> None:
    """Downward growth is limited so the mask stays off the blouse neckline."""
    region = FaceRegion(
        polygon=[(340, 100), (430, 100), (430, 200), (340, 200)],
        image_size=(WIDTH, HEIGHT),
    )
    column = np.asarray(build_face_mask(region))[:, 385]
    lit = np.flatnonzero(column > 20)
    up = 100 - int(lit[0])
    down = int(lit[-1]) - 200
    assert up > down, (up, down)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _mask(coverage: slice) -> Image.Image:
    """Build a simple 8-bit mask white over ``coverage`` rows."""
    array = np.zeros((HEIGHT, WIDTH), np.uint8)
    array[coverage] = 255
    return Image.fromarray(array, "L")


def test_background_drift_is_flagged() -> None:
    """A completely repainted background must raise a warning."""
    config = ValidationConfig()
    original = Image.new("RGB", (WIDTH, HEIGHT), (200, 200, 200))
    generated = Image.new("RGB", (WIDTH, HEIGHT), (40, 40, 40))

    finding = check_background_preserved(generated, original, _mask(slice(0, 10)), config)
    assert finding is not None
    assert finding.code == "background_drift"
    assert finding.severity is Severity.WARNING


def test_unchanged_background_passes() -> None:
    """An identical background must produce no finding."""
    config = ValidationConfig()
    image = Image.new("RGB", (WIDTH, HEIGHT), (200, 200, 200))
    assert check_background_preserved(image, image, _mask(slice(0, 10)), config) is None


def test_unchanged_garment_is_flagged() -> None:
    """The worst failure - returning the original outfit - must be caught."""
    config = ValidationConfig()
    image = Image.new("RGB", (WIDTH, HEIGHT), (120, 120, 120))

    finding = check_garment_changed(image, image, _mask(slice(200, 800)), config)
    assert finding is not None
    assert finding.code == "garment_unchanged"


def test_changed_garment_passes() -> None:
    """A genuinely different garment region produces no finding."""
    config = ValidationConfig()
    original = Image.new("RGB", (WIDTH, HEIGHT), (120, 120, 120))
    generated = Image.new("RGB", (WIDTH, HEIGHT), (220, 30, 60))
    assert check_garment_changed(generated, original, _mask(slice(200, 800)), config) is None


def test_exposure_drift_is_flagged_with_a_direction() -> None:
    """The message must say whether the result got brighter or darker."""
    config = ValidationConfig()
    dark = Image.new("RGB", (WIDTH, HEIGHT), (40, 40, 40))
    bright = Image.new("RGB", (WIDTH, HEIGHT), (220, 220, 220))

    brighter = check_exposure(bright, dark, config)
    assert brighter is not None and "brighter" in brighter.message

    darker = check_exposure(dark, bright, config)
    assert darker is not None and "darker" in darker.message


def test_framing_note_is_info_not_warning() -> None:
    """Partial framing is worth mentioning, not worth flagging as a problem."""
    note = check_framing_note(Category.SAREE, "half")
    assert note is not None
    assert note.severity is Severity.INFO

    assert check_framing_note(Category.SAREE, "full") is None
    assert check_framing_note(Category.KURTI, "half") is None


def test_validation_can_be_disabled() -> None:
    """A disabled config returns an empty, clean report."""
    config = ValidationConfig(enabled=False)
    report = validate_result(
        Image.new("RGB", (64, 64)),
        Image.new("RGB", (64, 64), (255, 255, 255)),
        Image.new("L", (64, 64), 255),
        Category.SAREE,
        "full",
        config=config,
    )
    assert report.findings == []
    assert report.ok


def test_report_separates_warnings_from_info() -> None:
    """``ok`` must ignore INFO findings but respect WARNINGs."""
    config = ValidationConfig(require_face=False)
    original = Image.new("RGB", (WIDTH, HEIGHT), (200, 200, 200))
    generated = Image.new("RGB", (WIDTH, HEIGHT), (30, 30, 30))

    report = validate_result(
        generated, original, _mask(slice(0, 20)), Category.SAREE, "half", config=config
    )
    assert not report.ok, "background + exposure drift should fail"
    assert report.warnings
    assert any(f.severity is Severity.INFO for f in report.findings), "framing note expected"
    assert report.render().startswith("- ")
    assert "partial_framing" in report.codes()


def test_validation_never_raises_on_odd_input() -> None:
    """Mismatched sizes must be handled, not thrown."""
    config = ValidationConfig(require_face=False)
    report = validate_result(
        Image.new("RGB", (100, 120)),
        Image.new("RGB", (WIDTH, HEIGHT)),
        Image.new("L", (WIDTH, HEIGHT), 255),
        Category.KURTI,
        "full",
        config=config,
    )
    assert isinstance(report.codes(), list)


# --------------------------------------------------------------------------- #
# Batch catalogue mode
# --------------------------------------------------------------------------- #


def _load_batch_module() -> Any:
    """Import ``scripts/batch_catalogue.py`` as a module."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "batch_catalogue.py"
    spec = importlib.util.spec_from_file_location("batch_catalogue", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs this registered
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_batch_job_expansion_is_a_cross_product() -> None:
    """2 people x 3 garments x 4 drapes = 24 jobs."""
    batch = _load_batch_module()
    persons = [Path(f"m{i}.jpg") for i in range(2)]
    garments = [Path(f"g{i}.jpg") for i in range(3)]

    single = batch.build_jobs(
        persons, garments, Category.SAREE,
        all_drapes=False, drape_style=DrapeStyle.NIVI,
        dupatta_style=DupattaStyle.SINGLE_SHOULDER,
    )
    assert len(single) == 6

    swept = batch.build_jobs(
        persons, garments, Category.SAREE,
        all_drapes=True, drape_style=DrapeStyle.NIVI,
        dupatta_style=DupattaStyle.SINGLE_SHOULDER,
    )
    assert len(swept) == 24
    assert {job.drape_style for job in swept} == set(DrapeStyle)


def test_all_drapes_only_applies_to_sarees() -> None:
    """A lehenga has no drape sweep, so the count must not multiply."""
    batch = _load_batch_module()
    jobs = batch.build_jobs(
        [Path("m.jpg")], [Path("g.jpg")], Category.LEHENGA,
        all_drapes=True, drape_style=DrapeStyle.NIVI,
        dupatta_style=DupattaStyle.OVER_HEAD,
    )
    assert len(jobs) == 1


def test_batch_output_names_are_deterministic_and_safe() -> None:
    """``--resume`` depends on the same job always producing the same name."""
    batch = _load_batch_module()
    job = batch.Job(
        person=Path("My Model (1).jpg"),
        garment=Path("saree #7.jpg"),
        category=Category.SAREE,
        drape_style=DrapeStyle.GUJARATI,
        dupatta_style=DupattaStyle.SINGLE_SHOULDER,
    )
    name = job.output_name()
    assert name == job.output_name(), "must be deterministic"
    assert "gujarati" in name
    assert not any(c in name for c in ' ()#/\\'), name

    lehenga = batch.Job(
        person=Path("m.jpg"), garment=Path("g.jpg"), category=Category.LEHENGA,
        drape_style=DrapeStyle.NIVI, dupatta_style=DupattaStyle.OVER_HEAD,
    )
    assert "over_head" in lehenga.output_name()


def test_batch_collect_images_filters_and_sorts(tmp_path: Path | None = None) -> None:
    """Only image files are picked up, in sorted order."""
    import tempfile

    batch = _load_batch_module()
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw)
        for name in ("b.jpg", "a.png", "notes.txt", "c.webp"):
            (folder / name).write_bytes(b"x")

        found = batch.collect_images(folder, label="garment")
        assert [p.name for p in found] == ["a.png", "b.jpg", "c.webp"]


def test_batch_collect_images_fails_loudly_on_empty_folder() -> None:
    """A batch run must abort before loading 20 GB of weights."""
    import tempfile

    batch = _load_batch_module()
    with tempfile.TemporaryDirectory() as raw:
        try:
            batch.collect_images(Path(raw), label="garment")
        except SystemExit:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit")


def test_batch_manifest_round_trips_every_column() -> None:
    """Rows with missing keys must still write, with blanks."""
    import csv as csv_module
    import tempfile

    batch = _load_batch_module()
    rows = [
        {"status": "ok", "person": "a.jpg", "garment": "b.jpg", "seed": 7,
         "validation_ok": True, "findings": "background_drift"},
        {"status": "failed", "person": "a.jpg", "error": "No person detected"},
    ]
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "manifest.csv"
        batch.write_manifest(rows, path)

        with path.open(encoding="utf-8") as handle:
            parsed = list(csv_module.DictReader(handle))

    assert len(parsed) == 2
    assert set(parsed[0]) == set(batch.MANIFEST_FIELDS)
    assert parsed[0]["seed"] == "7"
    assert parsed[1]["error"] == "No person detected"
    assert parsed[1]["seed"] == ""


def test_batch_contact_sheet_is_built_from_rendered_images() -> None:
    """A review grid is produced, and is larger than a single thumbnail."""
    import tempfile

    batch = _load_batch_module()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        images = root / "images"
        images.mkdir()
        for index in range(7):
            Image.new("RGB", (300, 450), (30 * index, 90, 140)).save(
                images / f"m__g{index}__nivi.jpg"
            )

        sheets = batch.build_contact_sheets(images, root)
        assert len(sheets) == 1
        with Image.open(sheets[0]) as sheet:
            assert sheet.width > 300 * 2
            assert sheet.height > 450


def test_batch_contact_sheet_handles_empty_directory() -> None:
    """No images means no sheets, and no exception."""
    import tempfile

    batch = _load_batch_module()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "images").mkdir()
        assert batch.build_contact_sheets(root / "images", root) == []


def test_batch_duration_formatting() -> None:
    """ETA rendering covers seconds, minutes and hours."""
    batch = _load_batch_module()
    assert batch._format_duration(45) == "45s"
    assert batch._format_duration(150) == "2m 30s"
    assert batch._format_duration(3900) == "1h 05m"
    assert batch._format_duration(-5) == "0s"


def test_batch_cli_parses_and_rejects_conflicts() -> None:
    """``--person`` and ``--persons`` are mutually exclusive."""
    batch = _load_batch_module()

    args = batch.parse_args(
        ["--person", "m.jpg", "--garments", "g/", "--category", "saree",
         "--all-drapes", "--preset", "best"]
    )
    assert args.category == "saree"
    assert args.all_drapes
    assert args.preset == "best"

    try:
        batch.parse_args(
            ["--person", "m.jpg", "--persons", "m/", "--garments", "g/",
             "--category", "saree"]
        )
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected mutually-exclusive SystemExit")


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
