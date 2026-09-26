"""Typed exceptions with user-facing messages.

Every exception carries a ``user_message`` that is safe to render directly in
the Gradio UI - no stack traces, no file paths, no model internals. The
technical detail stays in the log.
"""

from __future__ import annotations


class TrialRoomError(Exception):
    """Base class for all AI Trial Room errors.

    Parameters
    ----------
    user_message:
        Short, actionable sentence shown to the end user.
    detail:
        Technical context for the log. Never shown in the UI.
    """

    #: Fallback shown when a subclass does not supply a message.
    default_user_message: str = "Something went wrong. Please try again."

    def __init__(self, user_message: str | None = None, detail: str | None = None) -> None:
        self.user_message = user_message or self.default_user_message
        self.detail = detail
        super().__init__(self.user_message if detail is None else f"{self.user_message} ({detail})")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class InvalidInputError(TrialRoomError):
    """The user supplied an image or option the pipeline cannot use."""

    default_user_message = "That input could not be used. Please check your uploads."


class MissingImageError(InvalidInputError):
    """A required upload is absent."""

    default_user_message = "Please upload both a photo of the person and a garment image."


class NoPersonDetectedError(InvalidInputError):
    """Pose detection found no usable human figure."""

    default_user_message = (
        "No person detected in the photo. Use a clear, well-lit, front-facing "
        "full-body or half-body shot."
    )


class PersonPartiallyVisibleError(InvalidInputError):
    """A person was found but too much of the body is out of frame."""

    default_user_message = (
        "Only part of the body is visible. For sarees and lehengas, use a photo "
        "showing the person from head to at least mid-calf."
    )


class ImageTooSmallError(InvalidInputError):
    """An upload is below the minimum usable resolution."""

    default_user_message = "That image is too small. Please upload a larger, sharper photo."


class GarmentNotFoundError(InvalidInputError):
    """Background removal left nothing usable."""

    default_user_message = (
        "Could not isolate the garment. Use a photo of the garment on a plain "
        "background, or a flat-lay product shot."
    )


class ConsentNotGivenError(InvalidInputError):
    """The consent checkbox was not ticked."""

    default_user_message = (
        "Please confirm you own this photo or have permission to use it before "
        "generating."
    )


# --------------------------------------------------------------------------- #
# Backend / runtime
# --------------------------------------------------------------------------- #


class BackendError(TrialRoomError):
    """A generative backend failed."""

    default_user_message = "The try-on model failed to run. Please try again."


class BackendUnavailableError(BackendError):
    """A backend's weights or dependencies are not installed."""

    default_user_message = (
        "This try-on model is not installed in this environment. "
        "See the README for setup steps."
    )


class LicenseRestrictedError(BackendError):
    """A non-commercial backend was requested without an explicit opt-in."""

    default_user_message = (
        "That model is licensed for research use only and is disabled. "
        "Set ALLOW_NONCOMMERCIAL=1 to enable it for non-commercial use."
    )


class OutOfMemoryError(BackendError):
    """The GPU ran out of memory."""

    default_user_message = (
        "Ran out of GPU memory. Try a lower resolution or fewer inference steps "
        "in Advanced settings."
    )


class ModelLoadError(BackendError):
    """Weights could not be downloaded or instantiated."""

    default_user_message = (
        "Could not load the model weights. Check your network connection and "
        "HF_TOKEN, then retry."
    )
