"""Dependency-light tests for the layers that need no GPU or model weights.

Runnable two ways::

    pytest tests/test_core.py -v
    python tests/test_core.py          # plain-python fallback, no pytest needed

These cover the logic most likely to break silently: the mask geometry that
decides which pixels get repainted, the letterbox round-trip that must return
the user's original resolution, the license gate, and the prompt builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.backends.prompts import (  # noqa: E402
    DRAPE_SPECS,
    PRESERVE_CLAUSE,
    build_prompt,
)
from ai_trial_room.config import (  # noqa: E402
    CONFIG,
    MODEL_SPECS,
    BackendId,
    Category,
    DrapeStyle,
)
from ai_trial_room.postprocessing.blend import (  # noqa: E402
    feather_composite,
    harmonize,
)
from ai_trial_room.postprocessing.face_preserve import (  # noqa: E402
    FaceRegion,
    _convex_hull,
    build_face_mask,
)
from ai_trial_room.preprocessing.person import (  # noqa: E402
    AtrLabel,
    ParseResult,
    PoseLandmark,
    PoseResult,
    _draped_extension_mask,
    build_inpaint_mask,
)
from ai_trial_room.utils.device import (  # noqa: E402
    detect_hardware,
    is_oom_error,
    resolve_dtype,
    resolve_quantization,
)
from ai_trial_room.utils.errors import (  # noqa: E402
    ConsentNotGivenError,
    ImageTooSmallError,
    LicenseRestrictedError,
    NoPersonDetectedError,
    TrialRoomError,
)
from ai_trial_room.utils.image_io import (  # noqa: E402
    assert_min_size,
    ephemeral_dir,
    flatten_on_color,
    letterbox,
    make_side_by_side,
    round_to_multiple,
    trim_alpha,
    unletterbox,
)

WIDTH, HEIGHT = 768, 1024


# --------------------------------------------------------------------------- #
# Fixtures built by hand so the tests need no assets on disk
# --------------------------------------------------------------------------- #


def _fake_pose() -> PoseResult:
    """A standing, front-facing, full-length pose at 768x1024."""
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


def _fake_parse() -> ParseResult:
    """A person in a top and jeans, with face and hair labelled."""
    labels = np.zeros((HEIGHT, WIDTH), np.uint8)
    labels[240:520, 290:480] = int(AtrLabel.UPPER_CLOTHES)
    labels[520:900, 300:470] = int(AtrLabel.PANTS)
    labels[90:200, 340:430] = int(AtrLabel.FACE)
    labels[60:120, 330:440] = int(AtrLabel.HAIR)
    return ParseResult(label_map=labels, available=True)


# --------------------------------------------------------------------------- #
# Config and licensing
# --------------------------------------------------------------------------- #


def test_category_parsing_and_draping() -> None:
    """Labels round-trip, and only sarees/lehengas count as draped."""
    assert Category.from_label("Saree") is Category.SAREE
    assert Category.from_label("  kurti ") is Category.KURTI
    assert Category.SAREE.is_draped
    assert Category.LEHENGA.is_draped
    assert not Category.KURTI.is_draped
    assert not Category.DRESS.is_draped


def test_drape_style_parsing() -> None:
    """Drape labels round-trip from their full UI text."""
    assert DrapeStyle.from_label("Gujarati (Seedha pallu)") is DrapeStyle.GUJARATI
    assert DrapeStyle.from_label("Nauvari (Maharashtrian)") is DrapeStyle.NAUVARI
    assert len(DRAPE_SPECS) == len(DrapeStyle)


def test_commercial_default_and_license_split() -> None:
    """The shipped default must be a commercially licensed backend."""
    assert not CONFIG.allow_noncommercial, "non-commercial must be opt-in"
    assert MODEL_SPECS[CONFIG.default_backend].is_commercial

    commercial = [s.repo_id for s in MODEL_SPECS.values() if s.is_commercial]
    research = [s.repo_id for s in MODEL_SPECS.values() if not s.is_commercial]
    assert len(commercial) == 2, commercial
    assert len(research) == 1, research
    assert "CatVTON" in research[0]


def test_draped_categories_always_route_to_editing() -> None:
    """A warping backend can never be chosen for a saree or lehenga."""
    for category in (Category.SAREE, Category.LEHENGA):
        assert CONFIG.resolved_backend(category) is BackendId.QWEN_EDIT


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_every_category_gets_a_preservation_clause() -> None:
    """Identity preservation must appear verbatim in every prompt."""
    for category in Category:
        prompt = build_prompt(category)
        assert PRESERVE_CLAUSE in prompt, category
        assert len(prompt) > 300, (category, len(prompt))


def test_drape_styles_produce_distinct_prompts() -> None:
    """Each drape names its own shoulder and region, so outputs differ."""
    prompts = {
        style: build_prompt(Category.SAREE, drape_style=style) for style in DrapeStyle
    }
    assert len(set(prompts.values())) == len(DrapeStyle), "drapes must not collide"

    gujarati = prompts[DrapeStyle.GUJARATI]
    assert "RIGHT shoulder" in gujarati
    assert "Gujarat" in gujarati
    assert "pallu" in gujarati and "choli" in gujarati

    assert "Atpoure" in prompts[DrapeStyle.BENGALI]
    assert "dhoti-style" in prompts[DrapeStyle.NAUVARI]


def test_lehenga_prompt_names_all_three_pieces() -> None:
    """A lehenga is ghagra + choli + dupatta; naming each is what renders it."""
    prompt = build_prompt(Category.LEHENGA, dominant_shoulder="right")
    for component in ("ghagra", "choli", "dupatta", "right shoulder"):
        assert component in prompt, component


def test_framing_hint_prevents_invented_lower_body() -> None:
    """A half-body photo must instruct the model not to extend the frame."""
    assert "do not extend the frame" in build_prompt(Category.KURTI, framing="half")
    assert "full-length" in build_prompt(Category.SAREE, framing="full")


def test_unstitched_hint_switches_on() -> None:
    """A flat fabric length gets a drape-it-rather-than-copy-it instruction."""
    assert "unstitched length laid flat" in build_prompt(
        Category.SAREE, looks_unstitched=True
    )
    assert "different person or mannequin" in build_prompt(
        Category.SAREE, looks_unstitched=False
    )


def test_extra_prompt_is_appended() -> None:
    """Free-text from the UI reaches the prompt verbatim."""
    assert "gold potli bag" in build_prompt(Category.SAREE, extra="gold potli bag")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_letterbox_round_trip_restores_original_size() -> None:
    """The user must get their result back at the resolution they uploaded."""
    source = Image.new("RGB", (900, 1500), (30, 140, 90))
    padded, info = letterbox(source, (WIDTH, HEIGHT))

    assert padded.size == (WIDTH, HEIGHT)
    assert info.original_size == (900, 1500)

    restored = unletterbox(padded, info)
    assert restored.size == (900, 1500)
    # Content survives, not merely the dimensions.
    assert abs(restored.getpixel((450, 750))[1] - 140) < 6


def test_letterbox_pads_rather_than_crops() -> None:
    """A very tall photo keeps its full height as padding, losing nothing.

    This is what stops a saree hem being cropped off before generation.
    """
    tall = Image.new("RGB", (400, 1600))
    padded, info = letterbox(tall, (WIDTH, HEIGHT))
    assert info.scaled_size[1] == HEIGHT, "full height retained"
    assert info.scaled_size[0] < WIDTH, "width padded"
    assert padded.size == (WIDTH, HEIGHT)


def test_trim_alpha_crops_to_content_with_margin() -> None:
    """Background removal output is cropped tight, plus the configured margin."""
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    canvas.paste(Image.new("RGBA", (100, 200), (200, 50, 50, 255)), (150, 100))

    trimmed = trim_alpha(canvas, margin=10)
    assert trimmed.size == (120, 220)

    flat = flatten_on_color(trimmed)
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)


def test_round_to_multiple_and_side_by_side() -> None:
    """VAE-friendly rounding, and equal-height before/after composition."""
    assert round_to_multiple(1000) == 1008
    assert round_to_multiple(768) == 768
    assert round_to_multiple(0) == 16

    left = Image.new("RGB", (300, 400))
    right = Image.new("RGB", (200, 500))
    strip = make_side_by_side(left, right, gap=12)
    assert strip.height == 500


def test_pose_geometry() -> None:
    """Derived body geometry matches the landmarks it came from."""
    pose = _fake_pose()
    assert pose.framing == "full"
    assert pose.dominant_shoulder == "left"
    assert pose.torso_box() == (300, 260, 468, 520)
    assert pose.shoulder_width_px == 168.0


def test_framing_degrades_with_visibility() -> None:
    """Dropping ankle/knee landmarks steps framing down, not off a cliff."""
    full = _fake_pose()
    three_quarter = PoseResult(
        landmarks={k: v for k, v in full.landmarks.items() if k is not PoseLandmark.LEFT_ANKLE},
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.7,
    )
    half = PoseResult(
        landmarks={
            k: v
            for k, v in full.landmarks.items()
            if k not in {PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_KNEE}
        },
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.6,
    )
    assert (full.framing, three_quarter.framing, half.framing) == (
        "full",
        "three_quarter",
        "half",
    )


# --------------------------------------------------------------------------- #
# Masks - the highest-value logic in the project
# --------------------------------------------------------------------------- #


def test_saree_mask_covers_legs_but_kurti_mask_does_not() -> None:
    """The core Indian-wear correctness property.

    A kurti replaces only the top, so the subject's jeans must survive. A saree
    replaces everything below the shoulders, so the jeans must be repainted -
    otherwise denim shows through under the drape.
    """
    parse, pose = _fake_parse(), _fake_pose()
    kurti = np.asarray(build_inpaint_mask(parse, pose, Category.KURTI))
    saree = np.asarray(build_inpaint_mask(parse, pose, Category.SAREE))

    jeans_y, jeans_x = 700, 385
    assert kurti[jeans_y, jeans_x] < 40, "kurti must keep the existing trousers"
    assert saree[jeans_y, jeans_x] > 200, "saree must repaint the trousers"
    assert saree.mean() > kurti.mean() * 1.8


def test_identity_region_is_never_repainted() -> None:
    """Face and hair are excluded from every category's mask."""
    parse, pose = _fake_parse(), _fake_pose()
    for category in Category:
        mask = np.asarray(build_inpaint_mask(parse, pose, category))
        assert mask[150, 385] < 30, f"{category} repaints the face"
        assert mask[90, 385] < 60, f"{category} repaints the hair"


def test_draped_mask_reaches_the_hem() -> None:
    """A saree mask must extend to the bottom of the frame for the hem."""
    mask = np.asarray(build_inpaint_mask(_fake_parse(), _fake_pose(), Category.SAREE))
    assert mask[1000, 384] > 150


def test_drape_extension_flares_outward() -> None:
    """The added lower-body region widens from waist to hem.

    A straight rectangle would give a pencil-skirt silhouette; a saree and
    especially a lehenga need visible volume.
    """
    extension = _draped_extension_mask(_fake_pose(), (WIDTH, HEIGHT))
    waist = extension[540].sum()
    hem = extension[1000].sum()
    assert hem > waist, (waist, hem)
    assert hem >= waist * 1.4


def test_mask_falls_back_when_parsing_unavailable() -> None:
    """With no parsing model we still produce a usable pose-derived mask."""
    empty = ParseResult(label_map=np.zeros((HEIGHT, WIDTH), np.uint8), available=False)
    mask = np.asarray(build_inpaint_mask(empty, _fake_pose(), Category.SAREE))
    assert mask.mean() > 5, "fallback mask must not be empty"


# --------------------------------------------------------------------------- #
# Postprocessing
# --------------------------------------------------------------------------- #


def test_harmonize_pulls_exposure_toward_the_source_photo() -> None:
    """A bright generated image is darkened toward a dim source photo."""
    bright = Image.new("RGB", (256, 256), (240, 230, 220))
    dim = Image.new("RGB", (256, 256), (120, 110, 100))

    corrected = harmonize(bright, dim, strength=1.0)
    before = np.asarray(bright, float).mean()
    after = np.asarray(corrected, float).mean()
    target = np.asarray(dim, float).mean()

    assert after < before
    assert abs(after - target) < abs(before - target)


def test_harmonize_at_zero_strength_is_a_no_op() -> None:
    """Strength 0 must return the input untouched."""
    image = Image.new("RGB", (64, 64), (200, 100, 50))
    assert harmonize(image, Image.new("RGB", (64, 64)), strength=0.0) is image


def test_feather_composite_blends_both_sources() -> None:
    """A mid-grey mask yields a mix, not either extreme."""
    white = Image.new("RGB", (64, 64), (255, 255, 255))
    black = Image.new("RGB", (64, 64), (0, 0, 0))
    mixed = np.asarray(
        feather_composite(white, black, Image.new("L", (64, 64), 128)), float
    ).mean()
    assert 60 < mixed < 200


def test_convex_hull_drops_interior_points() -> None:
    """The hull of a square plus its centre is just the square."""
    hull = _convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    assert len(hull) == 4
    assert (5, 5) not in hull


def test_face_mask_is_opaque_inside_and_clear_outside() -> None:
    """The blend mask selects the face and nothing else."""
    region = FaceRegion(
        polygon=[(340, 100), (430, 100), (430, 200), (340, 200)],
        image_size=(WIDTH, HEIGHT),
    )
    assert region.width == 90
    assert region.centroid == (385.0, 150.0)

    mask = np.asarray(build_face_mask(region))
    assert mask[150, 385] > 200, "face centre must be selected"
    assert mask[600, 385] < 10, "torso must not be selected"


# --------------------------------------------------------------------------- #
# Errors and device
# --------------------------------------------------------------------------- #


def test_errors_carry_actionable_user_messages() -> None:
    """Every error the UI can show must tell the user what to do."""
    try:
        assert_min_size(Image.new("RGB", (100, 100)), 384, label="photo of the person")
    except ImageTooSmallError as exc:
        assert "100x100" in exc.user_message
        assert "384px" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("expected ImageTooSmallError")

    for error_cls in (NoPersonDetectedError, LicenseRestrictedError, ConsentNotGivenError):
        error = error_cls()
        assert isinstance(error, TrialRoomError)
        assert len(error.user_message) > 30, error_cls


def test_detail_stays_out_of_the_user_message() -> None:
    """Technical detail is logged, never shown."""
    error = NoPersonDetectedError(detail="mediapipe returned no pose_landmarks")
    assert "mediapipe" not in error.user_message
    assert error.detail is not None


def test_device_resolution_is_consistent() -> None:
    """dtype and quantization choices agree with the detected hardware."""
    import torch

    hardware = detect_hardware()
    dtype = resolve_dtype("auto")
    quantization = resolve_quantization("auto")

    assert quantization in {"none", "fp8", "int4"}
    if not hardware.is_cuda:
        assert dtype is torch.float32
        assert quantization == "none"
    else:
        assert dtype in {torch.bfloat16, torch.float16}
        if hardware.total_vram_gb < 15.0:
            assert quantization == "int4"

    assert resolve_dtype("float16") is torch.float16
    assert resolve_quantization("none") == "none"


def test_oom_detection() -> None:
    """OOM is recognised from the message so the UI can suggest a fix."""
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_oom_error(ValueError("unrelated"))


def test_ephemeral_dir_is_purged_even_on_error() -> None:
    """Uploaded photos must not survive a failed request."""
    captured: list[Path] = []
    try:
        with ephemeral_dir() as tmp:
            captured.append(tmp)
            (tmp / "photo.png").write_bytes(b"x")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not captured[0].exists()


def test_download_cache_survives_the_request_but_is_swept_by_ttl() -> None:
    """A downloadable result must outlive the handler, then be purged.

    Gradio fetches the file *after* the handler returns, so it cannot live in an
    ephemeral_dir. It must still not accumulate - hence the TTL sweep.
    """
    import os
    import time

    from ai_trial_room.utils.image_io import download_cache_dir, purge_download_cache

    cache = download_cache_dir()
    assert cache.exists()
    assert cache.is_dir()
    # Must be outside the project tree: nothing lands next to the source.
    assert Path.cwd() not in cache.parents

    fresh = cache / "fresh.png"
    fresh.write_bytes(b"fresh")

    stale = cache / "stale.png"
    stale.write_bytes(b"stale")
    old = time.time() - 4000
    os.utime(stale, (old, old))

    removed = purge_download_cache(max_age_s=900.0)
    assert removed >= 1
    assert not stale.exists(), "expired result must be deleted"
    assert fresh.exists(), "a just-written result must survive for the download"

    fresh.unlink()


def test_download_cache_dir_is_stable_within_a_process() -> None:
    """Repeated calls return the same directory, so paths stay valid."""
    from ai_trial_room.utils.image_io import download_cache_dir

    assert download_cache_dir() == download_cache_dir()


# --------------------------------------------------------------------------- #
# Plain-python runner, so the suite works without pytest installed
# --------------------------------------------------------------------------- #


def _main() -> int:
    """Run every ``test_*`` function in this module and report results."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: list[tuple[str, BaseException]] = []

    for name, func in tests:
        try:
            func()
        except BaseException as exc:  # noqa: BLE001 - this *is* the test runner
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    for name, exc in failures:
        print(f"  - {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
