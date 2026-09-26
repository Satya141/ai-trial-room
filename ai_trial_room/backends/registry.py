"""Backend registry with single-resident GPU memory management.

The invariant this module exists to enforce: **at most
``CONFIG.runtime.max_resident_backends`` heavy models hold VRAM at any moment.**

On a 16 GB T4, Qwen-Image-Edit alone needs most of the card. Loading a second
backend without evicting the first is the single most common way a Gradio demo
dies with CUDA OOM three minutes into a screen recording. :func:`get_backend`
evicts on a least-recently-used basis before handing back a backend.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Final

from ai_trial_room.backends.base import TryOnBackend
from ai_trial_room.backends.edit_backend import FluxKleinBackend, QwenEditBackend
from ai_trial_room.backends.vton_backend import CatVtonBackend
from ai_trial_room.config import CONFIG, MODEL_SPECS, BackendId, LicenseClass
from ai_trial_room.utils.device import vram_report
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Constructors, not instances - nothing is built until first use.
_FACTORIES: Final[dict[BackendId, Callable[[], TryOnBackend]]] = {
    BackendId.QWEN_EDIT: QwenEditBackend,
    BackendId.FLUX_KLEIN: FluxKleinBackend,
    BackendId.CATVTON: CatVtonBackend,
}

#: Instantiated backends, ordered by most recent use (LRU at the front).
_instances: "OrderedDict[BackendId, TryOnBackend]" = OrderedDict()

#: Guards the registry. Gradio serves requests on a thread pool, so two
#: concurrent generations could otherwise both try to load weights.
_lock = threading.RLock()


def available_backends(*, include_noncommercial: bool | None = None) -> list[BackendId]:
    """List backends this deployment is permitted to use.

    Parameters
    ----------
    include_noncommercial:
        Override the ``ALLOW_NONCOMMERCIAL`` setting. ``None`` uses the config.

    Returns
    -------
    list[BackendId]
        Commercial backends first, so the UI's default selection is safe.
    """
    allow = CONFIG.allow_noncommercial if include_noncommercial is None else include_noncommercial

    commercial = [
        bid
        for bid, spec in MODEL_SPECS.items()
        if spec.license_class is LicenseClass.COMMERCIAL
    ]
    if not allow:
        return commercial

    research = [
        bid
        for bid, spec in MODEL_SPECS.items()
        if spec.license_class is LicenseClass.NON_COMMERCIAL
    ]
    return commercial + research


def get_backend(backend_id: BackendId, *, evict: bool = True) -> TryOnBackend:
    """Return a backend instance, evicting others to stay within the VRAM budget.

    Weights are *not* loaded here - that happens lazily inside
    :meth:`~ai_trial_room.backends.base.TryOnBackend.generate`, so the UI can
    show a progress bar during the download.

    Parameters
    ----------
    backend_id:
        Which backend to fetch.
    evict:
        When True (the default), unload other loaded backends so that at most
        ``CONFIG.runtime.max_resident_backends`` remain resident.

    Returns
    -------
    TryOnBackend
    """
    with _lock:
        backend = _instances.get(backend_id)
        if backend is None:
            backend = _FACTORIES[backend_id]()
            _instances[backend_id] = backend
            logger.debug("Instantiated backend %s", backend_id.value)

        # Mark as most recently used.
        _instances.move_to_end(backend_id)

        if evict:
            _evict_locked(keep=backend_id)

        return backend


def _evict_locked(*, keep: BackendId) -> None:
    """Unload least-recently-used backends. Caller must hold ``_lock``."""
    budget = max(1, CONFIG.runtime.max_resident_backends)
    loaded = [bid for bid, backend in _instances.items() if backend.is_loaded]

    if len(loaded) < budget:
        return

    # _instances is ordered LRU-first, so evict from the front, never `keep`.
    for backend_id in list(_instances.keys()):
        if len([b for b in _instances.values() if b.is_loaded]) < budget:
            break
        if backend_id == keep:
            continue
        backend = _instances[backend_id]
        if backend.is_loaded:
            logger.info(
                "Evicting %s to make room for %s (budget=%d)",
                backend_id.value,
                keep.value,
                budget,
            )
            backend.unload()


def unload_all() -> None:
    """Unload every backend and release VRAM.

    Useful between notebook cells, and wired to the UI's "Free GPU memory"
    button so a long demo session can recover from memory pressure.
    """
    with _lock:
        for backend_id, backend in _instances.items():
            if backend.is_loaded:
                logger.info("Unloading %s", backend_id.value)
                backend.unload()
    logger.info("All backends unloaded | %s", vram_report())


def loaded_backends() -> list[BackendId]:
    """Return the ids of backends currently holding weights."""
    with _lock:
        return [bid for bid, backend in _instances.items() if backend.is_loaded]


def reset_registry() -> None:
    """Unload everything and forget all instances. Intended for tests."""
    with _lock:
        unload_all()
        _instances.clear()
