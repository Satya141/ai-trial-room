"""Post-generation quality checks.

Diffusion try-on fails in recognisable ways, and a shop deploying this needs to
know *which* result went wrong without eyeballing every one:

* the model ignored the garment instruction and returned the original outfit;
* it repainted the background as well as the clothes;
* it relit the whole frame so the person no longer matches the shop's lighting;
* it lost the face entirely.

Each check compares the output against the input it should have preserved, and
emits a :class:`Warning` with a sentence the salesperson can act on. Nothing here
ever rejects a result - a flagged image the operator can judge is more useful
than a refusal, and false positives are cheap while a hidden failure is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import numpy as np
from PIL import Image

from ai_trial_room.config import CONFIG, Category, ValidationConfig
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """How much a finding should worry the operator."""

    INFO = "info"
    WARNING = "warning"

    @property
    def icon(self) -> str:
        """Emoji used when rendering into the Gradio status line."""
        return "ℹ️" if self is Severity.INFO else "⚠️"


@dataclass(frozen=True)
class Finding:
    """One quality observation about a generated image."""

    code: str
    """Stable machine-readable identifier, e.g. ``"background_drift"``."""

    severity: Severity
    message: str
    """One sentence the operator can act on."""

    value: float | None = None
    """The measured quantity, for logs and the batch manifest."""

    threshold: float | None = None
    """The limit that was crossed."""

    def render(self) -> str:
        """Format as a Markdown bullet for the UI."""
        return f"{self.severity.icon} {self.message}"


@dataclass
class ValidationReport:
    """All findings for one generated image."""

    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing rose above :attr:`Severity.INFO`."""
        return not any(f.severity is Severity.WARNING for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        """Only the warning-level findings."""
        return [f for f in self.findings if f.severity is Severity.WARNING]

    def render(self) -> str:
        """Render every finding as Markdown bullets, or an empty string."""
        if not self.findings:
            return ""
        return "\n".join(f"- {finding.render()}" for finding in self.findings)

    def codes(self) -> list[str]:
        """Finding codes, for the batch CSV manifest."""
        return [finding.code for finding in self.findings]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _mean_abs_diff(a: Image.Image, b: Image.Image, weights: np.ndarray) -> float:
    """Weighted mean absolute difference between two RGB images, 0-255."""
    total = float(weights.sum())
    if total < 64.0:
        return 0.0
    left = np.asarray(a.convert("RGB"), dtype=np.float32)
    right = np.asarray(b.convert("RGB"), dtype=np.float32)
    diff = np.abs(left - right).mean(axis=2)
    return float((diff * weights).sum() / total)


def check_background_preserved(
    generated: Image.Image,
    original: Image.Image,
    inpaint_mask: Image.Image,
    config: ValidationConfig,
) -> Finding | None:
    """Verify the area outside the garment mask was left alone.

    The prompt tells the model to keep the background unchanged, but it has no
    hard constraint forcing that. A large drift here usually means the whole
    frame was regenerated, which is what makes a result unusable for a catalogue
    where every photo must share a backdrop.
    """
    mask = np.asarray(inpaint_mask.convert("L").resize(generated.size), dtype=np.float32) / 255.0
    outside = 1.0 - mask

    # Ignore a rim around the mask: feathering means the boundary legitimately changes.
    if outside.sum() < 64.0:
        return None

    drift = _mean_abs_diff(generated, original, outside)
    if drift <= config.max_background_drift:
        return None

    return Finding(
        code="background_drift",
        severity=Severity.WARNING,
        message=(
            "The background changed noticeably. If you need a consistent backdrop, "
            "lower the guidance scale or re-run with a different seed."
        ),
        value=round(drift, 2),
        threshold=config.max_background_drift,
    )


def check_garment_changed(
    generated: Image.Image,
    original: Image.Image,
    inpaint_mask: Image.Image,
    config: ValidationConfig,
) -> Finding | None:
    """Verify the garment region actually changed.

    If the masked area is nearly identical to the input, the model declined to
    follow the instruction - a real and fairly common failure with low guidance
    or an ambiguous garment photo. Silently returning the original outfit is the
    worst possible outcome, because it looks like a success.
    """
    mask = np.asarray(inpaint_mask.convert("L").resize(generated.size), dtype=np.float32) / 255.0
    if mask.sum() < 64.0:
        return None

    change = _mean_abs_diff(generated, original, mask)
    # Normalise against a nominal 255 range to get a 0-1 "how much changed".
    normalised = min(1.0, change / 60.0)

    if normalised >= config.min_garment_coverage:
        return None

    return Finding(
        code="garment_unchanged",
        severity=Severity.WARNING,
        message=(
            "The outfit barely changed - the model may have ignored the garment. "
            "Try a clearer garment photo on a plain background, or raise the "
            "guidance scale."
        ),
        value=round(normalised, 3),
        threshold=config.min_garment_coverage,
    )


def check_exposure(
    generated: Image.Image,
    original: Image.Image,
    config: ValidationConfig,
) -> Finding | None:
    """Verify overall brightness did not shift dramatically."""
    left = float(np.asarray(generated.convert("L"), dtype=np.float32).mean())
    right = float(np.asarray(original.convert("L"), dtype=np.float32).mean())
    drift = abs(left - right)

    if drift <= config.max_exposure_drift:
        return None

    direction = "brighter" if left > right else "darker"
    return Finding(
        code="exposure_drift",
        severity=Severity.WARNING,
        message=(
            f"The result is noticeably {direction} than the original photo. "
            "Enable 'Match colour & lighting', or raise its strength."
        ),
        value=round(drift, 1),
        threshold=config.max_exposure_drift,
    )


def check_face_present(generated: Image.Image, config: ValidationConfig) -> Finding | None:
    """Verify a face is still detectable in the output."""
    if not config.require_face:
        return None

    from ai_trial_room.postprocessing.face_preserve import detect_face

    if detect_face(generated) is not None:
        return None

    return Finding(
        code="face_missing",
        severity=Severity.WARNING,
        message=(
            "No face could be found in the result. Re-run with a different seed, "
            "or use a photo where the face is larger in frame."
        ),
    )


def check_framing_note(category: Category, framing: str) -> Finding | None:
    """Note, without warning, when framing limits what the model can do well."""
    if not category.is_draped:
        return None
    if framing == "full":
        return None

    return Finding(
        code="partial_framing",
        severity=Severity.INFO,
        message=(
            f"Only part of the body is in frame, so the lower drape of the "
            f"{category.label.lower()} is inferred rather than shown. A "
            "full-length photo gives a more accurate result."
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

#: Checks that compare against the original, in the order they are reported.
_COMPARATIVE_CHECKS: Final = (
    check_garment_changed,
    check_background_preserved,
)


def validate_result(
    generated: Image.Image,
    original: Image.Image,
    inpaint_mask: Image.Image,
    category: Category,
    framing: str,
    *,
    config: ValidationConfig | None = None,
) -> ValidationReport:
    """Run every quality check and collect the findings.

    Parameters
    ----------
    generated:
        Model output, at working resolution.
    original:
        Letterboxed input photo, same size as ``generated``.
    inpaint_mask:
        The mask the model was asked to repaint.
    category:
        Garment category.
    framing:
        ``"full"``, ``"three_quarter"`` or ``"half"``.
    config:
        Validation thresholds. Defaults to :attr:`AppConfig.validation`.

    Returns
    -------
    ValidationReport
        Empty when validation is disabled or everything passed.
    """
    config = config or CONFIG.validation
    report = ValidationReport()

    if not config.enabled:
        return report

    if generated.size != original.size:
        original = original.resize(generated.size, Image.Resampling.LANCZOS)

    try:
        for check in _COMPARATIVE_CHECKS:
            finding = check(generated, original, inpaint_mask, config)
            if finding is not None:
                report.findings.append(finding)

        for simple in (
            check_exposure(generated, original, config),
            check_face_present(generated, config),
            check_framing_note(category, framing),
        ):
            if simple is not None:
                report.findings.append(simple)

    except Exception as exc:  # noqa: BLE001 - validation must never break delivery
        logger.warning("Validation failed (%s); returning what was collected.", exc)

    report.metrics = {
        finding.code: finding.value
        for finding in report.findings
        if finding.value is not None
    }

    if report.findings:
        logger.info(
            "Validation: %d finding(s): %s",
            len(report.findings),
            ", ".join(report.codes()),
        )
    else:
        logger.info("Validation: clean")

    return report
