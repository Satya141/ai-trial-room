"""Backend selection and the end-to-end try-on pipeline.

:func:`select_backend` answers "which model should serve this request?" using
three rules, in order:

1. **Capability.** A draped garment (saree, lehenga) can only be served by an
   editing backend. No warping model can drape cloth.
2. **License.** A non-commercial backend is only ever selected when the
   operator has explicitly opted in via ``ALLOW_NONCOMMERCIAL=1``.
3. **Preference.** Otherwise honour the caller's request, falling back to the
   configured default.

:func:`run_try_on` is the single function ``app.py`` and the scripts call. It
owns the whole flow: preprocess, select, generate, postprocess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai_trial_room.backends.base import (
    ProgressCallback,
    TryOnOptions,
    TryOnRequest,
    TryOnResult,
    _noop_progress,
)
from ai_trial_room.backends.registry import available_backends, get_backend
from ai_trial_room.config import CONFIG, MODEL_SPECS, BackendId, Category, LicenseClass
from ai_trial_room.postprocessing.blend import harmonize
from ai_trial_room.postprocessing.face_preserve import restore_face
from ai_trial_room.preprocessing.garment import GarmentAssets, prepare_garment
from ai_trial_room.preprocessing.person import PersonAssets, prepare_person
from ai_trial_room.utils.errors import ConsentNotGivenError, LicenseRestrictedError
from ai_trial_room.utils.image_io import scrub_metadata, unletterbox
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TryOnReport:
    """The full outcome of one try-on, for the UI and for comparison scripts."""

    result: TryOnResult
    person: PersonAssets
    garment: GarmentAssets
    #: Final image at the uploaded photo's original resolution.
    final_image: Image.Image
    #: Wall-clock seconds for each stage, for the "why is it slow" question.
    timings: dict[str, float]

    @property
    def before(self) -> Image.Image:
        """The original photo, for the side-by-side view."""
        return self.person.original

    @property
    def after(self) -> Image.Image:
        """The generated try-on at original resolution."""
        return self.final_image

    def caption(self) -> str:
        """Markdown provenance line shown beneath the output."""
        spec = MODEL_SPECS[self.result.backend_id]
        warning = ""
        if spec.license_class is LicenseClass.NON_COMMERCIAL:
            warning = "  \n⚠️ **Non-commercial model** - this output may not be sold."
        total = sum(self.timings.values())
        return (
            f"{self.result.summary()}  \n"
            f"Stages: "
            + " · ".join(f"{name} {value:.1f}s" for name, value in self.timings.items())
            + f" · **total {total:.1f}s**"
            + warning
        )


def select_backend(
    category: Category,
    requested: BackendId | None = None,
) -> BackendId:
    """Choose the backend for ``category``.

    Parameters
    ----------
    category:
        Target garment category.
    requested:
        Caller's preference. ``None`` means "decide for me".

    Returns
    -------
    BackendId

    Raises
    ------
    LicenseRestrictedError
        When ``requested`` is non-commercial and the opt-in is not set.

    Examples
    --------
    A warping backend is silently upgraded for a draped garment:

    >>> select_backend(Category.SAREE, BackendId.CATVTON) is BackendId.QWEN_EDIT
    True
    """
    permitted = available_backends()

    if requested is not None:
        spec = MODEL_SPECS[requested]
        if requested not in permitted:
            raise LicenseRestrictedError(
                f"{spec.repo_id} is licensed {spec.license_name} and is disabled. "
                "Export ALLOW_NONCOMMERCIAL=1 to enable it for research use.",
                detail=f"requested={requested.value}",
            )

        backend = get_backend(requested, evict=False)
        if backend.supports(category):
            return requested

        fallback = CONFIG.resolved_backend(category)
        logger.info(
            "%s cannot serve %s (draped garment); routing to %s instead.",
            requested.value,
            category.value,
            fallback.value,
        )
        return fallback

    return CONFIG.resolved_backend(category)


def run_try_on(
    person_source: str | Path | Image.Image | np.ndarray,
    garment_source: str | Path | Image.Image | np.ndarray,
    category: Category,
    *,
    options: TryOnOptions | None = None,
    backend_id: BackendId | None = None,
    consent: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> TryOnReport:
    """Run the complete try-on pipeline.

    Parameters
    ----------
    person_source:
        Uploaded photo of the person.
    garment_source:
        Uploaded photo of the garment.
    category:
        Garment category.
    options:
        Generation settings. Defaults are taken from the config.
    backend_id:
        Force a specific backend. ``None`` lets :func:`select_backend` decide.
    consent:
        Must be True. Mirrors the UI checkbox; enforced here too so scripts and
        any future API cannot bypass it.
    progress:
        ``(fraction, message)`` sink for the UI progress bar.

    Returns
    -------
    TryOnReport

    Raises
    ------
    ConsentNotGivenError
        If ``consent`` is False.
    TrialRoomError
        Any preprocessing or backend failure, with a user-safe message.
    """
    if not consent:
        raise ConsentNotGivenError()

    options = options or TryOnOptions()
    timings: dict[str, float] = {}

    # --- 1. Preprocess ---------------------------------------------------- #
    progress(0.02, "Analysing the photo...")
    started = time.perf_counter()
    person = prepare_person(person_source, category)
    timings["person"] = time.perf_counter() - started

    progress(0.10, "Isolating the garment...")
    started = time.perf_counter()
    garment = prepare_garment(garment_source, category)
    timings["garment"] = time.perf_counter() - started

    # --- 2. Route --------------------------------------------------------- #
    chosen = select_backend(category, backend_id)
    backend = get_backend(chosen)
    logger.info(
        "Routing %s -> %s (%s, %s)",
        category.value,
        chosen.value,
        backend.family,
        backend.spec.license_name,
    )

    # --- 3. Generate ------------------------------------------------------ #
    started = time.perf_counter()
    result = backend.generate(
        TryOnRequest(
            person=person,
            garment=garment,
            category=category,
            options=options,
            progress=progress,
        )
    )
    timings["generate"] = time.perf_counter() - started

    # --- 4. Postprocess --------------------------------------------------- #
    progress(0.92, "Restoring face and matching colours...")
    started = time.perf_counter()
    image = result.image

    if options.harmonize_colors:
        image = harmonize(image, person.image)
    if options.preserve_face:
        image = restore_face(image, person)

    # Return at the resolution the customer uploaded, with EXIF stripped so a
    # downloaded file cannot leak GPS coordinates from the original photo.
    final_image = scrub_metadata(unletterbox(image, person.letterbox_info))
    timings["postprocess"] = time.perf_counter() - started

    progress(1.0, "Done")
    logger.info(
        "Try-on complete for %s in %.1fs total",
        category.value,
        sum(timings.values()),
    )

    return TryOnReport(
        result=result,
        person=person,
        garment=garment,
        final_image=final_image,
        timings=timings,
    )


def describe_routing() -> list[dict[str, Any]]:
    """Return a table of category -> backend decisions, for the UI's About tab."""
    rows: list[dict[str, Any]] = []
    for category in Category:
        backend_id = CONFIG.resolved_backend(category)
        spec = MODEL_SPECS[backend_id]
        rows.append(
            {
                "Category": category.label,
                "Backend": spec.repo_id,
                "Family": "editing" if not category.is_draped else "editing (required)",
                "License": spec.license_name,
            }
        )
    return rows
