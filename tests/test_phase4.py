"""Tests for Phase 4: commercially-safe parsing and Spaces deployment.

Two things matter most here and both are tested directly:

* **The licence gate.** The default human-parsing provider must be Apache-2.0,
  and the research-licensed one must be unreachable without an explicit opt-in.
  This is the difference between a project you can invoice for and one you cannot.
* **The deploy manifest.** A Space is public infrastructure. Training photos,
  trained adapters and ``.env`` must never reach it, and the check must not depend
  on the exclusion lists happening to be right.

Run standalone::

    python tests/test_phase4.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.config import CONFIG, Category, LicenseClass  # noqa: E402
from ai_trial_room.preprocessing.parsing import (  # noqa: E402
    MEDIAPIPE_MODEL_URL,
    ParsingProvider,
    SelfieClass,
    _split_by_pose,
    parse_human,
    provider_report,
    resolve_provider,
)
from ai_trial_room.preprocessing.person import (  # noqa: E402
    AtrLabel,
    ParseResult,
    PoseLandmark,
    PoseResult,
    build_inpaint_mask,
)
from ai_trial_room.spaces_support import (  # noqa: E402
    ZEROGPU_MAX_DURATION,
    SpaceInfo,
    apply_space_overrides,
    detect_space,
    estimate_duration,
    gpu,
    space_footer,
)

WIDTH, HEIGHT = 768, 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pose() -> PoseResult:
    """Standing front-facing full-length pose."""
    return PoseResult(
        landmarks={
            PoseLandmark.NOSE: (384.0, 120.0, 0.99),
            PoseLandmark.LEFT_SHOULDER: (300.0, 260.0, 0.97),
            PoseLandmark.RIGHT_SHOULDER: (468.0, 260.0, 0.93),
            PoseLandmark.LEFT_HIP: (320.0, 520.0, 0.90),
            PoseLandmark.RIGHT_HIP: (450.0, 520.0, 0.88),
            PoseLandmark.LEFT_KNEE: (322.0, 740.0, 0.80),
            PoseLandmark.LEFT_ANKLE: (324.0, 950.0, 0.70),
        },
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.8,
    )


def _mediapipe_categories() -> np.ndarray:
    """A realistic MediaPipe output: one `clothes` class over top AND trousers."""
    categories = np.zeros((HEIGHT, WIDTH), np.uint8)
    categories[240:900, 290:480] = SelfieClass.CLOTHES
    categories[260:520, 250:300] = SelfieClass.BODY_SKIN  # arm
    categories[900:990, 300:470] = SelfieClass.BODY_SKIN  # ankles
    categories[90:200, 340:430] = SelfieClass.FACE_SKIN
    categories[60:120, 330:440] = SelfieClass.HAIR
    return categories


def _label_map_from(categories: np.ndarray) -> np.ndarray:
    """Apply the same class mapping :func:`parse_with_mediapipe` uses."""
    label_map = np.zeros(categories.shape, np.uint8)
    label_map[categories == SelfieClass.HAIR] = int(AtrLabel.HAIR)
    label_map[categories == SelfieClass.FACE_SKIN] = int(AtrLabel.FACE)
    label_map[categories == SelfieClass.ACCESSORIES] = int(AtrLabel.BAG)
    label_map[categories == SelfieClass.CLOTHES] = int(AtrLabel.UPPER_CLOTHES)
    label_map[categories == SelfieClass.BODY_SKIN] = int(AtrLabel.LEFT_ARM)
    return label_map


# --------------------------------------------------------------------------- #
# The licence fix
# --------------------------------------------------------------------------- #


def test_default_parsing_provider_is_commercially_licensed() -> None:
    """This is the Phase 4 headline: the default must be Apache-2.0."""
    provider = resolve_provider()
    assert provider is ParsingProvider.MEDIAPIPE
    assert provider.is_commercial
    assert provider.license_class is LicenseClass.COMMERCIAL
    assert provider.license_name == "Apache-2.0"


def test_segformer_is_blocked_without_an_explicit_opt_in() -> None:
    """The research-licensed parser must be unreachable by default."""
    assert not CONFIG.allow_noncommercial, "the shipped default must be commercial-only"
    assert not ParsingProvider.SEGFORMER.is_commercial
    # Asking for it by name silently downgrades rather than serving it.
    assert resolve_provider("segformer") is ParsingProvider.MEDIAPIPE


def test_segformer_reachable_only_with_the_opt_in() -> None:
    """With ALLOW_NONCOMMERCIAL=1 the research parser becomes selectable."""
    original = CONFIG.allow_noncommercial
    try:
        CONFIG.allow_noncommercial = True
        assert resolve_provider("segformer") is ParsingProvider.SEGFORMER
    finally:
        CONFIG.allow_noncommercial = original


def test_unknown_provider_falls_back_safely() -> None:
    """A typo must not disable parsing or reach a gated provider."""
    assert resolve_provider("nonsense") is ParsingProvider.MEDIAPIPE
    assert resolve_provider("") is ParsingProvider.MEDIAPIPE
    assert resolve_provider("pose_only") is ParsingProvider.POSE_ONLY


def test_pose_only_provider_needs_no_model() -> None:
    """The no-model path is commercially safe and reports itself unavailable."""
    assert ParsingProvider.POSE_ONLY.is_commercial
    outcome = parse_human(Image.new("RGB", (64, 96)), provider="pose_only")
    assert outcome.provider is ParsingProvider.POSE_ONLY
    assert not outcome.available
    assert outcome.label_map.shape == (96, 64)


def test_provider_report_is_renderable() -> None:
    """The About panel needs a complete, honest provider summary."""
    report = provider_report()
    assert set(report) == {"Provider", "Model", "Licence", "Commercial use"}
    assert report["Licence"] == "Apache-2.0"
    assert report["Commercial use"] == "yes"


def test_mediapipe_model_url_is_the_official_google_one() -> None:
    """The weights must come from Google's own bucket, not a mirror."""
    assert MEDIAPIPE_MODEL_URL.startswith("https://storage.googleapis.com/mediapipe-models/")
    assert MEDIAPIPE_MODEL_URL.endswith(".tflite")


# --------------------------------------------------------------------------- #
# Recovering garment distinctions from coarse classes
# --------------------------------------------------------------------------- #


def test_pose_split_separates_upper_from_lower_clothing() -> None:
    """MediaPipe has one `clothes` class; the hip line splits it.

    Without this, a kurti mask would repaint the customer's trousers.
    """
    categories = _mediapipe_categories()
    label_map = _label_map_from(categories)

    assert label_map[700, 385] == int(AtrLabel.UPPER_CLOTHES), "unsplit baseline"

    _split_by_pose(label_map, categories, _pose())

    assert label_map[300, 385] == int(AtrLabel.UPPER_CLOTHES), "top stays upper"
    assert label_map[700, 385] == int(AtrLabel.PANTS), "trousers become lower"
    assert label_map[300, 270] == int(AtrLabel.LEFT_ARM), "skin above hips is arm"
    assert label_map[950, 385] == int(AtrLabel.LEFT_LEG), "skin below hips is leg"
    assert label_map[150, 385] == int(AtrLabel.FACE), "face untouched"
    assert label_map[100, 385] == int(AtrLabel.HAIR), "hair untouched"


def test_split_restores_the_kurti_versus_saree_mask_difference() -> None:
    """The payoff: an Apache-2.0 model yields the same masks as the gated one.

    A kurti must leave the trousers alone; a saree must repaint them.
    """
    categories = _mediapipe_categories()
    label_map = _label_map_from(categories)
    pose = _pose()
    _split_by_pose(label_map, categories, pose)

    parse = ParseResult(label_map=label_map, available=True, provider="mediapipe")
    kurti = np.asarray(build_inpaint_mask(parse, pose, Category.KURTI))
    saree = np.asarray(build_inpaint_mask(parse, pose, Category.SAREE))

    assert kurti[700, 385] < 40, "kurti must keep the existing trousers"
    assert saree[700, 385] > 200, "saree must repaint the trousers"
    assert saree.mean() > kurti.mean() * 1.8

    # Identity is protected under both.
    assert kurti[150, 385] < 30
    assert saree[150, 385] < 30


def test_split_is_a_no_op_without_hip_landmarks() -> None:
    """Missing hips must leave the map unchanged rather than raising."""
    categories = _mediapipe_categories()
    label_map = _label_map_from(categories)
    before = label_map.copy()

    bare = PoseResult(
        landmarks={PoseLandmark.NOSE: (384.0, 120.0, 0.99)},
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.2,
    )
    _split_by_pose(label_map, categories, bare)

    assert np.array_equal(label_map, before)


def test_split_clamps_an_out_of_frame_hip_line() -> None:
    """A hip landmark below the image must not index out of bounds."""
    categories = _mediapipe_categories()
    label_map = _label_map_from(categories)

    pose = PoseResult(
        landmarks={
            PoseLandmark.LEFT_HIP: (320.0, 99999.0, 0.9),
            PoseLandmark.RIGHT_HIP: (450.0, 99999.0, 0.9),
        },
        image_size=(WIDTH, HEIGHT),
        visible_ratio=0.5,
    )
    _split_by_pose(label_map, categories, pose)  # must not raise
    assert label_map.shape == (HEIGHT, WIDTH)


def test_parse_result_records_its_provider() -> None:
    """Provenance must survive into the ParseResult for logs and the UI."""
    result = ParseResult(
        label_map=np.zeros((8, 8), np.uint8), available=True, provider="mediapipe"
    )
    assert result.provider == "mediapipe"
    # Default stays empty so older constructions keep working.
    assert ParseResult(label_map=np.zeros((8, 8), np.uint8), available=False).provider == ""


# --------------------------------------------------------------------------- #
# Spaces / ZeroGPU
# --------------------------------------------------------------------------- #


def test_space_detection_is_false_locally() -> None:
    """Off Spaces everything must be inert."""
    info = detect_space()
    assert isinstance(info, SpaceInfo)
    assert not info.on_space
    assert info.is_local
    assert space_footer() == ""
    assert apply_space_overrides() == []


def test_gpu_decorator_is_a_passthrough_off_spaces() -> None:
    """The decorated function must behave identically with no `spaces` package."""

    @gpu(duration=90)
    def double(value: int) -> int:
        return value * 2

    assert double(21) == 42

    @gpu()
    def triple(value: int) -> int:
        return value * 3

    assert triple(5) == 15


def test_duration_estimate_scales_and_is_capped() -> None:
    """ZeroGPU kills an under-estimated call, so the estimate must track cost."""
    fast = estimate_duration(20, "qwen_edit")
    balanced = estimate_duration(30, "qwen_edit")
    best = estimate_duration(40, "qwen_edit", refine=True)

    assert fast < balanced < best
    assert best <= ZEROGPU_MAX_DURATION

    # The 4B model is cheaper per step.
    assert estimate_duration(30, "flux_klein") < estimate_duration(30, "qwen_edit")

    # Absurd inputs clamp rather than overflow the quota.
    assert estimate_duration(10_000, "qwen_edit", refine=True) == ZEROGPU_MAX_DURATION
    assert estimate_duration(1, "qwen_edit") >= 30

    # An unknown backend still produces a usable number.
    assert 30 <= estimate_duration(30, "unknown_backend") <= ZEROGPU_MAX_DURATION


def test_space_overrides_force_commercial_only_and_disable_offload() -> None:
    """A public Space must never serve a research-licensed model."""
    import ai_trial_room.spaces_support as module

    original_detect = module.detect_space
    saved = (
        CONFIG.allow_noncommercial,
        CONFIG.runtime.model_cpu_offload,
        CONFIG.runtime.sequential_cpu_offload,
        CONFIG.runtime.quantize,
        CONFIG.privacy.delete_outputs_on_exit,
    )
    try:
        module.detect_space = lambda: SpaceInfo(  # type: ignore[assignment]
            on_space=True, zero_gpu=True, space_id="owner/space"
        )
        CONFIG.allow_noncommercial = True
        CONFIG.runtime.model_cpu_offload = True
        CONFIG.runtime.quantize = "int4"

        changes = module.apply_space_overrides()

        assert CONFIG.allow_noncommercial is False, "must force commercial-only"
        assert CONFIG.runtime.model_cpu_offload is False
        assert CONFIG.runtime.sequential_cpu_offload is False
        assert CONFIG.runtime.quantize == "none"
        assert CONFIG.privacy.delete_outputs_on_exit is True
        assert any("ALLOW_NONCOMMERCIAL" in change for change in changes)
        assert any("CPU offload" in change for change in changes)
    finally:
        module.detect_space = original_detect
        (
            CONFIG.allow_noncommercial,
            CONFIG.runtime.model_cpu_offload,
            CONFIG.runtime.sequential_cpu_offload,
            CONFIG.runtime.quantize,
            CONFIG.privacy.delete_outputs_on_exit,
        ) = saved


def test_dedicated_space_keeps_offload_but_still_forces_licence() -> None:
    """Non-ZeroGPU Spaces have a small resident GPU, so offload stays useful."""
    import ai_trial_room.spaces_support as module

    original_detect = module.detect_space
    saved = (CONFIG.allow_noncommercial, CONFIG.runtime.model_cpu_offload)
    try:
        module.detect_space = lambda: SpaceInfo(  # type: ignore[assignment]
            on_space=True, zero_gpu=False, space_id="owner/space"
        )
        CONFIG.allow_noncommercial = True
        CONFIG.runtime.model_cpu_offload = True

        module.apply_space_overrides()

        assert CONFIG.allow_noncommercial is False
        assert CONFIG.runtime.model_cpu_offload is True, "keep offload off ZeroGPU"
    finally:
        module.detect_space = original_detect
        CONFIG.allow_noncommercial, CONFIG.runtime.model_cpu_offload = saved


def test_space_footer_mentions_the_cold_start() -> None:
    """Users need warning that the first request loads ~20 GB."""
    import ai_trial_room.spaces_support as module

    original = module.detect_space
    try:
        module.detect_space = lambda: SpaceInfo(  # type: ignore[assignment]
            on_space=True, zero_gpu=True, space_id="owner/space"
        )
        footer = module.space_footer()
    finally:
        module.detect_space = original

    assert "ZeroGPU" in footer
    assert "cold start" in footer


# --------------------------------------------------------------------------- #
# Space configuration files
# --------------------------------------------------------------------------- #


def _front_matter() -> dict[str, str]:
    """Parse the top-level scalar keys of the Space README front matter."""
    text = (PROJECT_ROOT / "space" / "README.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, "space/README.md has no YAML front matter"

    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line or line.startswith((" ", "-", "#")):
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    return values


def test_space_readme_has_every_required_key() -> None:
    """Spaces refuses to build without these."""
    front_matter = _front_matter()
    for key in ("title", "emoji", "colorFrom", "colorTo", "sdk", "sdk_version", "app_file"):
        assert key in front_matter, f"missing {key}"

    assert front_matter["sdk"] == "gradio", "ZeroGPU only supports the Gradio SDK"
    assert front_matter["app_file"] == "app.py"
    assert front_matter["license"] == "apache-2.0"


def test_space_sdk_version_matches_the_pinned_gradio() -> None:
    """A mismatch here is a silent, confusing build failure."""
    declared = _front_matter()["sdk_version"]

    requirements = (PROJECT_ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    pinned = next(
        line.split("==")[1].strip()
        for line in requirements.splitlines()
        if line.startswith("gradio==")
    )
    assert declared == pinned, f"sdk_version {declared} != gradio {pinned}"


def test_space_requirements_include_spaces_but_not_torch() -> None:
    """`spaces` provides the ZeroGPU decorator; torch ships with the image."""
    requirements = (PROJECT_ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert any(line.split("==")[0] == "spaces" for line in lines)
    assert not any(line.startswith("torch") for line in lines), "do not pin torch on Spaces"
    # Training and dev packages must not bloat the image.
    assert not any(line.startswith(("pytest", "bitsandbytes")) for line in lines)


def test_space_packages_cover_the_opencv_shared_libraries() -> None:
    """Missing libGL is the most common Spaces startup failure for this stack."""
    packages = (PROJECT_ROOT / "space" / "packages.txt").read_text(encoding="utf-8")
    assert "libgl1" in packages
    assert "libglib2.0-0" in packages


def test_space_readme_documents_the_consent_gate_and_privacy() -> None:
    """The public card must state what happens to uploaded photos."""
    text = (PROJECT_ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert "not stored" in text
    assert "consent" in text.lower()
    assert "full-length" in text
    # All four drapes should be documented for users.
    for drape in ("Nivi", "Bengali", "Gujarati", "Nauvari"):
        assert drape in text, drape


# --------------------------------------------------------------------------- #
# Deploy manifest
# --------------------------------------------------------------------------- #


def _load_deploy() -> Any:
    """Import ``scripts/deploy_space.py`` as a module."""
    path = PROJECT_ROOT / "scripts" / "deploy_space.py"
    spec = importlib.util.spec_from_file_location("deploy_space_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_manifest_contains_everything_the_space_needs() -> None:
    """A Space without these three files will not start."""
    deploy = _load_deploy()
    remotes = {upload.remote for upload in deploy.collect_uploads()}

    for required in ("app.py", "README.md", "requirements.txt", "packages.txt"):
        assert required in remotes, f"missing {required}"
    assert any(remote.startswith("ai_trial_room/") for remote in remotes)
    assert deploy.verify_manifest(deploy.collect_uploads()) == []


def test_manifest_excludes_training_data_tests_and_secrets() -> None:
    """A Space is public. None of this may ever reach it."""
    deploy = _load_deploy()
    remotes = [upload.remote for upload in deploy.collect_uploads()]

    for forbidden in ("datasets/", "loras/", "outputs/", "tests/", "notebooks/", "third_party/"):
        assert not [r for r in remotes if r.startswith(forbidden)], forbidden

    assert not [r for r in remotes if ".env" in r]
    assert not [r for r in remotes if r.endswith((".safetensors", ".ckpt", ".pth", ".ipynb"))]
    assert not [r for r in remotes if "__pycache__" in r]


def test_space_readme_is_promoted_over_the_project_readme() -> None:
    """The Space card must be space/README.md, not the developer README."""
    deploy = _load_deploy()
    uploads = {upload.remote: upload for upload in deploy.collect_uploads()}

    assert uploads["README.md"].local.parent.name == "space"
    assert uploads["requirements.txt"].local.parent.name == "space"


def test_verify_manifest_catches_a_leaked_secret() -> None:
    """The leak check must not depend on the exclusion lists being correct."""
    deploy = _load_deploy()

    with tempfile.TemporaryDirectory() as raw:
        planted = Path(raw) / "secret_token.txt"
        planted.write_text("hf_xxx", encoding="utf-8")

        uploads = deploy.collect_uploads() + [
            deploy.Upload(local=planted, remote="config/secret_token.txt")
        ]
        problems = deploy.verify_manifest(uploads)

    assert any("secret" in problem for problem in problems)


def test_verify_manifest_catches_leaked_training_data() -> None:
    """Planting a dataset file must be caught independently of the scan."""
    deploy = _load_deploy()

    with tempfile.TemporaryDirectory() as raw:
        planted = Path(raw) / "nivi_001.jpg"
        Image.new("RGB", (32, 32)).save(planted)

        uploads = deploy.collect_uploads() + [
            deploy.Upload(local=planted, remote="datasets/saree_drapes/nivi_001.jpg")
        ]
        problems = deploy.verify_manifest(uploads)

    assert any("training or output data" in problem for problem in problems)


def test_verify_manifest_reports_missing_essentials() -> None:
    """An incomplete manifest must be refused with a specific reason."""
    deploy = _load_deploy()
    problems = deploy.verify_manifest([])

    joined = " ".join(problems)
    assert "app.py" in joined
    assert "README.md" in joined
    assert "requirements.txt" in joined
    assert "ai_trial_room" in joined


def test_deploy_cli_dry_run_needs_no_token() -> None:
    """--dry-run must work offline and without credentials."""
    deploy = _load_deploy()

    saved = os.environ.pop("HF_TOKEN", None)
    try:
        code = deploy.main(["--repo", "owner/space", "--dry-run"])
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved

    assert code == 0


def test_deploy_cli_rejects_a_malformed_repo_id() -> None:
    """A repo without a slash cannot be a Space."""
    deploy = _load_deploy()
    assert deploy.main(["--repo", "no-slash", "--dry-run"]) == 2


def test_deploy_cli_requires_a_token_for_a_real_push() -> None:
    """Without HF_TOKEN the push must fail before contacting the Hub."""
    deploy = _load_deploy()

    saved = os.environ.pop("HF_TOKEN", None)
    try:
        code = deploy.main(["--repo", "owner/space", "--yes"])
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved

    assert code == 2


def test_deploy_hardware_choices_include_zerogpu() -> None:
    """ZeroGPU must be requestable, since it is the recommended tier."""
    deploy = _load_deploy()
    assert "zero-a10g" in deploy.HARDWARE_CHOICES
    assert "cpu-basic" in deploy.HARDWARE_CHOICES


# --------------------------------------------------------------------------- #
# App integration
# --------------------------------------------------------------------------- #


def test_app_duration_estimator_reads_the_right_arguments() -> None:
    """Indices come from the signature, so adding a UI control cannot break it."""
    if importlib.util.find_spec("gradio") is None:
        print("    (skipped: gradio not installed)")
        return

    import app

    index = app._GENERATE_PARAM_INDEX
    for name in ("preset_label", "steps", "refine"):
        assert name in index, name

    def call(preset: str, steps: int, refine: bool) -> int:
        args: list[Any] = [None] * len(index)
        args[index["preset_label"]] = preset
        args[index["steps"]] = steps
        args[index["refine"]] = refine
        return app._gpu_duration(*args)

    fast = call("Fast (~1 min)", 0, False)
    best = call("Best (~4 min)", 0, True)
    assert best > fast, (fast, best)

    # An explicit step override must raise the estimate.
    assert call("Fast (~1 min)", 50, False) > fast

    # Keyword form and the no-argument fallback must both work.
    assert app._gpu_duration(preset_label="Best (~4 min)", steps=0, refine=True) == best
    assert app._gpu_duration() >= 30


def test_app_about_panel_reports_the_commercial_parser() -> None:
    """The About panel must show the Apache-2.0 provider, not the gated one."""
    if importlib.util.find_spec("gradio") is None:
        print("    (skipped: gradio not installed)")
        return

    import app

    markdown = app._about_markdown()
    assert "Apache-2.0" in markdown
    assert "Human parsing" in markdown
    assert "hip line" in markdown, "should explain how coarse classes are refined"


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
