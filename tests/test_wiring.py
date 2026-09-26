"""Import-and-wiring tests: does the whole graph hold together?

These exercise module imports, the abstract-base contract, the licence gate and
the router's decisions. They stub out heavy dependencies where necessary, so the
suite runs with no GPU and no model weights.

Modules that need ``gradio`` or ``diffusers`` are skipped with a clear message
when those are absent, rather than failing.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has(module: str) -> bool:
    """True when ``module`` is importable in this environment."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_DIFFUSERS = _has("diffusers")
HAS_GRADIO = _has("gradio")


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #


def test_core_modules_import() -> None:
    """Every module that needs only torch/PIL/numpy must import cleanly."""
    for name in (
        "ai_trial_room",
        "ai_trial_room.config",
        "ai_trial_room.utils.errors",
        "ai_trial_room.utils.logging_setup",
        "ai_trial_room.utils.device",
        "ai_trial_room.utils.image_io",
        "ai_trial_room.preprocessing.person",
        "ai_trial_room.preprocessing.garment",
        "ai_trial_room.postprocessing.blend",
        "ai_trial_room.postprocessing.face_preserve",
        "ai_trial_room.backends.prompts",
        "ai_trial_room.backends.base",
        "ai_trial_room.backends.edit_backend",
        "ai_trial_room.backends.vton_backend",
        "ai_trial_room.backends.registry",
        "ai_trial_room.router",
    ):
        importlib.import_module(name)


def test_app_imports_when_gradio_present() -> None:
    """``app.py`` must import without starting a server."""
    if not HAS_GRADIO:
        print("    (skipped: gradio not installed)")
        return
    module = importlib.import_module("app")
    assert hasattr(module, "build_ui")
    assert hasattr(module, "generate")


def test_comparison_script_parses_its_arguments() -> None:
    """The comparison CLI must be importable and its parser well-formed."""
    spec = importlib.util.spec_from_file_location(
        "compare_backends",
        Path(__file__).resolve().parent.parent / "scripts" / "compare_backends.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which raises AttributeError if absent.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    args = module.parse_args(
        ["--person", "a.jpg", "--garment", "b.jpg", "--category", "lehenga", "--seed", "7"]
    )
    assert args.category == "lehenga"
    assert args.seed == 7
    assert not args.sweep_drapes


# --------------------------------------------------------------------------- #
# Backend contract
# --------------------------------------------------------------------------- #


def test_every_backend_implements_the_contract() -> None:
    """Each registered backend is constructible and declares its metadata."""
    from ai_trial_room.backends.registry import _FACTORIES
    from ai_trial_room.config import BackendId

    assert set(_FACTORIES) == set(BackendId), "registry must cover every BackendId"

    for backend_id, factory in _FACTORIES.items():
        backend = factory()
        assert backend.backend_id is backend_id
        assert backend.family in {"edit", "vton"}
        assert not backend.is_loaded, "constructing must not load weights"
        assert backend.spec.repo_id
        assert backend.supported_categories, backend_id


def test_editing_backends_cover_all_categories_and_warping_does_not() -> None:
    """Capability declarations must match physical reality."""
    from ai_trial_room.backends.edit_backend import FluxKleinBackend, QwenEditBackend
    from ai_trial_room.backends.vton_backend import CatVtonBackend
    from ai_trial_room.config import Category

    for cls in (QwenEditBackend, FluxKleinBackend):
        backend = cls()
        assert backend.supported_categories == frozenset(Category)
        assert backend.supports(Category.SAREE)

    warping = CatVtonBackend()
    assert not warping.supports(Category.SAREE)
    assert not warping.supports(Category.LEHENGA)
    assert warping.supports(Category.KURTI)


def test_pipeline_class_names_match_the_model_specs() -> None:
    """The class each backend imports must match what the spec advertises."""
    from ai_trial_room.backends.edit_backend import FluxKleinBackend, QwenEditBackend

    for cls in (QwenEditBackend, FluxKleinBackend):
        backend = cls()
        assert backend.pipeline_class_name == backend.spec.pipeline_class


# --------------------------------------------------------------------------- #
# Licence gate - the commercially important test
# --------------------------------------------------------------------------- #


def test_commercial_backends_pass_the_licence_gate() -> None:
    """Apache-2.0 backends load without an opt-in."""
    from ai_trial_room.backends.edit_backend import FluxKleinBackend, QwenEditBackend

    QwenEditBackend().check_license()
    FluxKleinBackend().check_license()


def test_noncommercial_backend_is_blocked_by_default() -> None:
    """CatVTON must refuse to load before any weights are fetched."""
    from ai_trial_room.backends.vton_backend import CatVtonBackend
    from ai_trial_room.utils.errors import LicenseRestrictedError

    try:
        CatVtonBackend().check_license()
    except LicenseRestrictedError as exc:
        assert "ALLOW_NONCOMMERCIAL" in exc.user_message
        assert "CC BY-NC-SA" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("non-commercial backend was not blocked")


def test_noncommercial_backend_opens_with_explicit_opt_in() -> None:
    """With the opt-in set, the gate allows it through (with a warning)."""
    import ai_trial_room.config as config_module

    original = config_module.CONFIG.allow_noncommercial
    try:
        config_module.CONFIG.allow_noncommercial = True
        from ai_trial_room.backends.vton_backend import CatVtonBackend

        CatVtonBackend().check_license()  # must not raise
    finally:
        config_module.CONFIG.allow_noncommercial = original


def test_available_backends_excludes_research_models_by_default() -> None:
    """The UI dropdown must not offer a model you cannot sell."""
    from ai_trial_room.backends.registry import available_backends
    from ai_trial_room.config import MODEL_SPECS

    default = available_backends()
    assert all(MODEL_SPECS[b].is_commercial for b in default)
    assert len(default) == 2

    with_research = available_backends(include_noncommercial=True)
    assert len(with_research) == 3
    # Commercial options must come first so the default selection is safe.
    assert MODEL_SPECS[with_research[0]].is_commercial


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_router_upgrades_warping_request_for_draped_garments() -> None:
    """Asking for CatVTON with a saree silently routes to the editing backend."""
    import ai_trial_room.config as config_module
    from ai_trial_room.backends.registry import reset_registry
    from ai_trial_room.config import BackendId, Category
    from ai_trial_room.router import select_backend

    original = config_module.CONFIG.allow_noncommercial
    try:
        config_module.CONFIG.allow_noncommercial = True
        reset_registry()
        assert select_backend(Category.SAREE, BackendId.CATVTON) is BackendId.QWEN_EDIT
        assert select_backend(Category.LEHENGA, BackendId.CATVTON) is BackendId.QWEN_EDIT
        # A category it genuinely supports is honoured.
        assert select_backend(Category.KURTI, BackendId.CATVTON) is BackendId.CATVTON
    finally:
        config_module.CONFIG.allow_noncommercial = original
        reset_registry()


def test_router_rejects_a_blocked_backend_request() -> None:
    """Requesting a gated backend without the opt-in raises, not falls back."""
    from ai_trial_room.config import BackendId, Category
    from ai_trial_room.router import select_backend
    from ai_trial_room.utils.errors import LicenseRestrictedError

    try:
        select_backend(Category.KURTI, BackendId.CATVTON)
    except LicenseRestrictedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected LicenseRestrictedError")


def test_router_defaults_to_commercial_for_every_category() -> None:
    """With no preference, every category resolves to a sellable model."""
    from ai_trial_room.config import MODEL_SPECS, Category
    from ai_trial_room.router import select_backend

    for category in Category:
        chosen = select_backend(category, None)
        assert MODEL_SPECS[chosen].is_commercial, category


def test_consent_is_enforced_below_the_ui() -> None:
    """``run_try_on`` must reject a request with consent=False before any work."""
    from ai_trial_room.config import Category
    from ai_trial_room.router import run_try_on
    from ai_trial_room.utils.errors import ConsentNotGivenError

    try:
        run_try_on("nope.jpg", "nope.jpg", Category.SAREE, consent=False)
    except ConsentNotGivenError:
        pass
    else:  # pragma: no cover
        raise AssertionError("consent check did not fire")


def test_routing_table_is_renderable() -> None:
    """The About tab's routing table has a row per category."""
    from ai_trial_room.config import Category
    from ai_trial_room.router import describe_routing

    rows = describe_routing()
    assert len(rows) == len(Category)
    for row in rows:
        assert set(row) == {"Category", "Backend", "Family", "License"}
        assert row["License"] == "Apache-2.0", "default routing must stay commercial"


# --------------------------------------------------------------------------- #
# Options and registry mechanics
# --------------------------------------------------------------------------- #


def test_seed_resolution() -> None:
    """A negative or unset seed becomes a concrete random one; a set seed sticks."""
    from ai_trial_room.backends.base import TryOnOptions

    assert TryOnOptions(seed=4242).resolved_seed() == 4242
    for options in (TryOnOptions(seed=-1), TryOnOptions(seed=None)):
        seed = options.resolved_seed()
        assert isinstance(seed, int) and 0 <= seed < 2**31


def test_registry_reuses_instances_and_reports_loaded_state() -> None:
    """``get_backend`` is a cache, and nothing is loaded until asked."""
    from ai_trial_room.backends.registry import (
        get_backend,
        loaded_backends,
        reset_registry,
    )
    from ai_trial_room.config import BackendId

    reset_registry()
    first = get_backend(BackendId.QWEN_EDIT)
    second = get_backend(BackendId.QWEN_EDIT)
    assert first is second
    assert loaded_backends() == [], "no weights should be resident yet"
    reset_registry()


def test_unsupported_category_is_refused_before_loading() -> None:
    """A capability mismatch raises from generate() without fetching weights."""
    from ai_trial_room.backends.base import TryOnRequest
    from ai_trial_room.backends.vton_backend import CatVtonBackend
    from ai_trial_room.config import Category
    from ai_trial_room.utils.errors import BackendError

    backend = CatVtonBackend()
    request = TryOnRequest(person=None, garment=None, category=Category.SAREE)  # type: ignore[arg-type]

    try:
        backend.generate(request)
    except BackendError as exc:
        assert "does not support" in exc.user_message
        assert not backend.is_loaded
    else:  # pragma: no cover
        raise AssertionError("expected BackendError")


def test_missing_diffusers_class_gives_an_actionable_message() -> None:
    """A too-old diffusers install must say so, not raise AttributeError."""
    if not HAS_DIFFUSERS:
        print("    (skipped: diffusers not installed)")
        return

    from ai_trial_room.backends.edit_backend import QwenEditBackend
    from ai_trial_room.utils.errors import ModelLoadError

    backend = QwenEditBackend()
    backend.pipeline_class_name = "DefinitelyNotARealPipeline"
    try:
        backend._import_pipeline_class()
    except ModelLoadError as exc:
        assert "diffusers" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("expected ModelLoadError")


def test_catvton_reports_missing_installation_clearly() -> None:
    """Without the cloned repo, CatVTON explains itself as optional."""
    import ai_trial_room.config as config_module
    from ai_trial_room.backends.vton_backend import CatVtonBackend
    from ai_trial_room.utils.errors import BackendUnavailableError

    original_flag = config_module.CONFIG.allow_noncommercial
    original_path = os.environ.get("AITR_CATVTON_PATH")
    try:
        config_module.CONFIG.allow_noncommercial = True
        os.environ["AITR_CATVTON_PATH"] = "third_party/definitely-not-here"
        CatVtonBackend()._repo_path()
    except BackendUnavailableError as exc:
        assert "optional" in exc.user_message
    else:  # pragma: no cover
        raise AssertionError("expected BackendUnavailableError")
    finally:
        config_module.CONFIG.allow_noncommercial = original_flag
        if original_path is None:
            os.environ.pop("AITR_CATVTON_PATH", None)
        else:
            os.environ["AITR_CATVTON_PATH"] = original_path


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _main() -> int:
    """Run every ``test_*`` in this module, reporting pass/fail per test."""
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
