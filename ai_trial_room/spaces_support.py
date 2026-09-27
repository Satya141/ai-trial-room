"""Hugging Face Spaces and ZeroGPU integration.

ZeroGPU in one paragraph
------------------------
On ZeroGPU hardware there is **no GPU attached to the process** until a function
decorated with ``@spaces.GPU`` is called; the GPU is allocated for that call and
released afterwards. That inverts two assumptions this project makes everywhere
else:

1. **Nothing may touch CUDA at import time.** Backends already load lazily, so
   this holds - but :func:`ai_trial_room.utils.device.detect_hardware` is cached,
   and on ZeroGPU it would cache "no GPU" during startup and then be wrong inside
   the decorated call. :func:`reset_hardware_cache` clears it.
2. **CPU offload is counterproductive.** ``enable_model_cpu_offload`` exists to
   stream weights onto a small resident GPU. On ZeroGPU the GPU is large (H200
   slice) and only briefly present, so offloading just adds transfer time to
   every request. :func:`apply_space_overrides` turns it off.

Everything here degrades to a no-op off Spaces, so ``app.py`` needs no branching
and local development is unaffected.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Final, TypeVar

from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

#: Environment variables Spaces sets. Presence of the first means "on a Space".
_SPACE_ID_VARS: Final[tuple[str, ...]] = ("SPACE_ID", "SPACE_REPO_ID")

#: Set by ZeroGPU hardware specifically.
_ZEROGPU_VARS: Final[tuple[str, ...]] = ("SPACES_ZERO_GPU", "ZEROGPU")

#: ZeroGPU's hard ceiling on a single function call, in seconds.
ZEROGPU_MAX_DURATION: Final[int] = 300

#: Seconds of headroom added to every duration estimate, covering model load,
#: preprocessing and postprocessing around the denoise loop.
_DURATION_OVERHEAD_S: Final[int] = 55

#: Rough seconds per denoising step on a ZeroGPU slice, per backend family.
#: Deliberately pessimistic: an underestimate gets the call killed mid-generation,
#: while an overestimate only costs queue priority.
_SECONDS_PER_STEP: Final[dict[str, float]] = {"qwen_edit": 1.8, "flux_klein": 0.8}


@dataclass(frozen=True)
class SpaceInfo:
    """What kind of environment this process is running in."""

    on_space: bool
    zero_gpu: bool
    space_id: str = ""
    sdk_version: str = ""

    @property
    def is_local(self) -> bool:
        """True when running outside Spaces entirely."""
        return not self.on_space

    def describe(self) -> str:
        """One-line summary for the startup log."""
        if not self.on_space:
            return "local (not a Hugging Face Space)"
        kind = "ZeroGPU" if self.zero_gpu else "dedicated hardware"
        return f"Hugging Face Space {self.space_id or '?'} on {kind}"


@functools.lru_cache(maxsize=1)
def detect_space() -> SpaceInfo:
    """Detect the Spaces environment. Cached for the process lifetime."""
    space_id = ""
    for name in _SPACE_ID_VARS:
        if os.environ.get(name):
            space_id = os.environ[name]
            break

    on_space = bool(space_id)
    zero_gpu = any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in _ZEROGPU_VARS
    )

    info = SpaceInfo(
        on_space=on_space,
        zero_gpu=zero_gpu,
        space_id=space_id,
        sdk_version=os.environ.get("GRADIO_VERSION", ""),
    )
    logger.info("Environment: %s", info.describe())
    return info


def reset_hardware_cache() -> None:
    """Clear the cached hardware snapshot.

    On ZeroGPU, :func:`~ai_trial_room.utils.device.detect_hardware` runs during
    startup with no GPU attached and caches that. Calling this at the top of a
    ``@spaces.GPU`` function makes the next detection see the real device.
    """
    from ai_trial_room.utils.device import detect_hardware

    detect_hardware.cache_clear()


def estimate_duration(steps: int, backend: str = "qwen_edit", *, refine: bool = False) -> int:
    """Estimate how long one generation needs, for the ZeroGPU quota.

    Parameters
    ----------
    steps:
        Denoising steps for the first pass.
    backend:
        Backend id; picks the per-step cost.
    refine:
        Whether a second region-refinement pass runs, which roughly doubles the
        denoise work.

    Returns
    -------
    int
        Seconds, clamped to :data:`ZEROGPU_MAX_DURATION`.

    Examples
    --------
    >>> estimate_duration(30, "qwen_edit") <= ZEROGPU_MAX_DURATION
    True
    >>> estimate_duration(30, "flux_klein") < estimate_duration(30, "qwen_edit")
    True
    """
    per_step = _SECONDS_PER_STEP.get(backend, 1.8)
    total_steps = steps * (2 if refine else 1)
    estimate = int(total_steps * per_step) + _DURATION_OVERHEAD_S
    return max(30, min(estimate, ZEROGPU_MAX_DURATION))


def gpu(duration: int | Callable[..., int] | None = None) -> Callable[[F], F]:
    """Return ``spaces.GPU`` when on ZeroGPU, otherwise a pass-through decorator.

    Using this instead of importing ``spaces`` directly keeps ``app.py`` free of
    environment branching, and keeps ``spaces`` an optional dependency.

    Parameters
    ----------
    duration:
        Seconds, or a callable receiving the wrapped function's arguments and
        returning seconds. ``None`` lets ZeroGPU use its default.

    Returns
    -------
    Callable
        A decorator.

    Examples
    --------
    >>> @gpu(duration=90)
    ... def generate(x):
    ...     return x * 2
    >>> generate(3)
    6
    """
    info = detect_space()

    if not info.zero_gpu:
        def _passthrough(func: F) -> F:
            return func

        return _passthrough

    try:
        import spaces  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "SPACES_ZERO_GPU is set but the `spaces` package is not installed; "
            "GPU allocation will not work. Add `spaces` to requirements.txt."
        )

        def _missing(func: F) -> F:
            return func

        return _missing

    def _decorate(func: F) -> F:
        decorated = (
            spaces.GPU(func) if duration is None else spaces.GPU(duration=duration)(func)
        )
        logger.info(
            "Wrapped %s with spaces.GPU (duration=%s)",
            getattr(func, "__name__", "<fn>"),
            duration if duration is not None else "default",
        )
        return decorated  # type: ignore[return-value]

    return _decorate


def apply_space_overrides() -> list[str]:
    """Adjust the runtime config for the detected Space environment.

    Returns
    -------
    list[str]
        Human-readable descriptions of what was changed, for the startup log.
    """
    from ai_trial_room.config import CONFIG

    info = detect_space()
    changes: list[str] = []

    if not info.on_space:
        return changes

    # A public Space must never serve a research-licensed model.
    if CONFIG.allow_noncommercial:
        CONFIG.allow_noncommercial = False
        changes.append(
            "forced ALLOW_NONCOMMERCIAL off - a public Space must not serve "
            "research-licensed models"
        )

    # Uploads are other people's photos on a public Space; bin outputs too.
    if not CONFIG.privacy.delete_outputs_on_exit:
        CONFIG.privacy.delete_outputs_on_exit = True
        changes.append("enabled output deletion")

    if info.zero_gpu:
        runtime = CONFIG.runtime
        if runtime.model_cpu_offload or runtime.sequential_cpu_offload:
            runtime.model_cpu_offload = False
            runtime.sequential_cpu_offload = False
            changes.append(
                "disabled CPU offload - ZeroGPU attaches a large GPU per call, so "
                "streaming weights only adds latency"
            )
        if runtime.quantize != "none":
            runtime.quantize = "none"
            changes.append("disabled quantization - there is VRAM headroom on ZeroGPU")

    for change in changes:
        logger.info("Space override: %s", change)
    return changes


def space_footer() -> str:
    """Markdown footer shown in the UI when running on a Space."""
    info = detect_space()
    if not info.on_space:
        return ""

    hardware = "ZeroGPU (shared, allocated per request)" if info.zero_gpu else "dedicated GPU"
    return (
        f"Running on Hugging Face Spaces · {hardware} · "
        "first request after a cold start loads ~20 GB of weights and may take "
        "several minutes."
    )
