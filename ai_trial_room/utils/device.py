"""GPU capability detection and memory hygiene.

The app must run unchanged on three very different machines:

======================  ========  ==========================================
Target                  VRAM      Strategy
======================  ========  ==========================================
Kaggle / Colab T4       16 GB     fp16 (no bf16 on Turing) + model offload
RTX 40xx / 50xx laptop  8-12 GB   bf16 + int4 (nunchaku) + sequential offload
A10G / L4 Space         24 GB     bf16, everything resident
======================  ========  ==========================================

:func:`detect_hardware` inspects the live device once and caches the result;
:func:`resolve_dtype` and :func:`resolve_quantization` turn ``"auto"`` config
values into concrete choices.
"""

from __future__ import annotations

import functools
import gc
from dataclasses import dataclass

from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Hardware:
    """A snapshot of the compute device available to this process."""

    device: str
    """``"cuda"``, ``"mps"`` or ``"cpu"``."""

    name: str
    """Marketing name, e.g. ``"NVIDIA GeForce RTX 5050 Laptop GPU"``."""

    total_vram_gb: float
    """Total device memory in GiB. ``0.0`` on CPU."""

    supports_bf16: bool
    """True on Ampere (sm_80) and newer."""

    compute_capability: tuple[int, int] | None
    """CUDA compute capability, or ``None`` off CUDA."""

    @property
    def is_cuda(self) -> bool:
        """True when a CUDA device is present."""
        return self.device == "cuda"

    @property
    def is_low_vram(self) -> bool:
        """True when the card cannot hold a 12B transformer at 8-bit."""
        return self.is_cuda and self.total_vram_gb < 15.0

    def describe(self) -> str:
        """One-line summary for the log and the UI footer."""
        if not self.is_cuda:
            return f"{self.device.upper()} (no GPU - generation will be very slow)"
        return (
            f"{self.name} | {self.total_vram_gb:.1f} GB VRAM | "
            f"bf16={'yes' if self.supports_bf16 else 'no'}"
        )


@functools.lru_cache(maxsize=1)
def detect_hardware() -> Hardware:
    """Inspect the current machine. Cached for the process lifetime."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
        logger.warning("PyTorch is not installed; assuming CPU.")
        return Hardware("cpu", "cpu", 0.0, False, None)

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        hardware = Hardware(
            device="cuda",
            name=props.name,
            total_vram_gb=props.total_memory / (1024**3),
            supports_bf16=torch.cuda.is_bf16_supported(),
            compute_capability=capability,
        )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        hardware = Hardware("mps", "Apple Silicon (MPS)", 0.0, False, None)
    else:
        hardware = Hardware("cpu", "cpu", 0.0, False, None)

    logger.info("Detected hardware: %s", hardware.describe())
    return hardware


def resolve_dtype(requested: str = "auto"):
    """Turn a config dtype string into a ``torch.dtype``.

    Parameters
    ----------
    requested:
        ``"auto"``, ``"bfloat16"``, ``"float16"`` or ``"float32"``.

    Returns
    -------
    torch.dtype
        ``bfloat16`` on Ampere+, ``float16`` on older CUDA cards (the T4 has no
        usable bf16 path), ``float32`` on CPU.
    """
    import torch

    explicit = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if requested.lower() in explicit:
        return explicit[requested.lower()]

    hardware = detect_hardware()
    if not hardware.is_cuda:
        return torch.float32
    return torch.bfloat16 if hardware.supports_bf16 else torch.float16


def resolve_quantization(requested: str = "auto") -> str:
    """Decide whether to quantize the transformer.

    Parameters
    ----------
    requested:
        ``"auto"``, ``"none"``, ``"fp8"`` or ``"int4"``.

    Returns
    -------
    str
        One of ``"none"``, ``"fp8"``, ``"int4"``. In ``auto`` mode: ``int4``
        under 15 GB VRAM (needs ``nunchaku``), ``fp8`` from 15-20 GB, and
        ``none`` above that.
    """
    choice = requested.lower()
    if choice in {"none", "fp8", "int4"}:
        return choice

    hardware = detect_hardware()
    if not hardware.is_cuda:
        return "none"
    if hardware.total_vram_gb < 15.0:
        return "int4"
    if hardware.total_vram_gb < 20.0:
        return "fp8"
    return "none"


def free_vram() -> None:
    """Release cached allocations back to the driver.

    Called between backend swaps so two heavy models never co-reside. Safe to
    call when no GPU is present.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:  # pragma: no cover
        pass


def vram_report() -> str:
    """Return a short ``allocated / reserved / total`` string for logging."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda unavailable"
        gib = 1024**3
        return (
            f"allocated={torch.cuda.memory_allocated() / gib:.2f} GB "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f} GB "
            f"total={detect_hardware().total_vram_gb:.1f} GB"
        )
    except ImportError:  # pragma: no cover
        return "torch unavailable"


def is_oom_error(exc: BaseException) -> bool:
    """Heuristically classify an exception as a GPU out-of-memory failure."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text
