"""LoRA discovery and loading for inference.

A trained adapter is a directory under ``loras/`` containing peft weights plus
the ``aitr_lora.json`` sidecar that
:func:`ai_trial_room.training.train_lora.save_adapter` writes. The sidecar is
what makes a LoRA self-describing - which base model it belongs to, what trigger
token it was captioned with, what it was trained on - so the UI can offer it
safely rather than letting someone load a FLUX adapter into Qwen and get noise.

Discovery is filesystem-based and deliberately dumb: drop a folder in, restart,
and it appears in the dropdown.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ai_trial_room.config import MODEL_SPECS, PROJECT_ROOT, BackendId
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Where trained adapters live.
LORA_DIR: Final[Path] = Path(
    __import__("os").environ.get("AITR_LORA_DIR", PROJECT_ROOT / "loras")
)

#: Sidecar filename written next to the adapter weights.
SIDECAR_NAME: Final[str] = "aitr_lora.json"

#: Filenames that indicate a directory actually holds peft weights.
_WEIGHT_NAMES: Final[tuple[str, ...]] = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_lora_weights.safetensors",
)

#: Label shown for "no LoRA" in the UI.
NO_LORA_LABEL: Final[str] = "None (base model)"


@dataclass(frozen=True)
class LoraSpec:
    """A discovered LoRA adapter."""

    name: str
    path: Path
    backend: BackendId
    """Which base model this adapter was trained against."""

    trigger: str = ""
    """Token to insert into the prompt to activate the learned concept."""

    rank: int = 0
    mode: str = ""
    steps_trained: int = 0
    base_repo: str = ""
    base_license: str = ""
    notes: str = ""
    #: True when no sidecar was found and the metadata is guessed.
    unverified: bool = False

    @property
    def label(self) -> str:
        """Dropdown label, flagging an adapter whose base model is unknown."""
        suffix = "  [unverified]" if self.unverified else ""
        return f"{self.name}{suffix}"

    def is_compatible(self, backend: BackendId) -> bool:
        """True when this adapter may be applied to ``backend``.

        An unverified adapter is allowed through with a warning - it may be a
        third-party LoRA without our sidecar - but a sidecar naming a different
        base model is a hard mismatch.
        """
        if self.unverified:
            return True
        return self.backend is backend

    def describe(self) -> str:
        """One-line summary for logs and the UI."""
        parts = [f"{self.name}"]
        if self.rank:
            parts.append(f"rank {self.rank}")
        if self.steps_trained:
            parts.append(f"{self.steps_trained} steps")
        if self.trigger:
            parts.append(f"trigger '{self.trigger}'")
        parts.append(self.base_repo or self.backend.value)
        return " · ".join(parts)


def _read_sidecar(directory: Path) -> dict[str, Any] | None:
    """Read and parse ``aitr_lora.json``, or ``None`` when absent/invalid."""
    path = directory / SIDECAR_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring malformed %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _has_weights(directory: Path) -> bool:
    """True when ``directory`` contains recognisable adapter weights."""
    return any((directory / name).is_file() for name in _WEIGHT_NAMES)


def _spec_from_directory(directory: Path) -> LoraSpec | None:
    """Build a :class:`LoraSpec` from an adapter directory."""
    if not _has_weights(directory):
        return None

    sidecar = _read_sidecar(directory)
    if sidecar is None:
        logger.info(
            "LoRA %s has no %s; assuming it targets %s.",
            directory.name,
            SIDECAR_NAME,
            MODEL_SPECS[BackendId.QWEN_EDIT].repo_id,
        )
        return LoraSpec(
            name=directory.name,
            path=directory,
            backend=BackendId.QWEN_EDIT,
            unverified=True,
        )

    try:
        backend = BackendId(sidecar.get("backend", BackendId.QWEN_EDIT.value))
    except ValueError:
        logger.warning(
            "LoRA %s names an unknown backend %r; treating as unverified.",
            directory.name,
            sidecar.get("backend"),
        )
        return LoraSpec(
            name=directory.name, path=directory, backend=BackendId.QWEN_EDIT, unverified=True
        )

    return LoraSpec(
        name=str(sidecar.get("name") or directory.name),
        path=directory,
        backend=backend,
        trigger=str(sidecar.get("trigger", "")),
        rank=int(sidecar.get("rank", 0) or 0),
        mode=str(sidecar.get("mode", "")),
        steps_trained=int(sidecar.get("steps_trained", 0) or 0),
        base_repo=str(sidecar.get("base_repo", "")),
        base_license=str(sidecar.get("base_license", "")),
        notes=str(sidecar.get("notes", "")),
    )


def discover_loras(root: Path | None = None) -> list[LoraSpec]:
    """Find every adapter under ``root``.

    Scans immediate subdirectories and the root itself, so both
    ``loras/saree_v1/`` and a bare ``loras/`` adapter are found. Checkpoint
    subdirectories (``step-000400``) are skipped - only final adapters are
    offered.

    Parameters
    ----------
    root:
        Directory to scan. Defaults to :data:`LORA_DIR`.

    Returns
    -------
    list[LoraSpec]
        Sorted by name. Empty when the directory does not exist.
    """
    root = root or LORA_DIR
    if not root.is_dir():
        return []

    found: list[LoraSpec] = []

    root_spec = _spec_from_directory(root)
    if root_spec is not None:
        found.append(root_spec)

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("step-", ".")):
            continue
        spec = _spec_from_directory(child)
        if spec is not None:
            found.append(spec)

    if found:
        logger.info(
            "Discovered %d LoRA(s) in %s: %s",
            len(found),
            root,
            ", ".join(spec.name for spec in found),
        )
    return sorted(found, key=lambda spec: spec.name)


@functools.lru_cache(maxsize=1)
def cached_loras() -> tuple[LoraSpec, ...]:
    """Discover adapters once per process, for the UI dropdown."""
    return tuple(discover_loras())


def find_lora(name: str, root: Path | None = None) -> LoraSpec | None:
    """Look up an adapter by name or by label.

    Parameters
    ----------
    name:
        Adapter name, or the dropdown label including an ``[unverified]`` suffix.
    root:
        Directory to scan. Defaults to :data:`LORA_DIR`.

    Returns
    -------
    LoraSpec or None
    """
    if not name or name == NO_LORA_LABEL:
        return None

    wanted = name.split("  [")[0].strip()
    for spec in discover_loras(root):
        if spec.name == wanted:
            return spec
    return None


def apply_trigger(prompt: str, spec: LoraSpec | None) -> str:
    """Prepend a LoRA's trigger token to ``prompt`` if it is not already there.

    The trigger must appear for the adapter to do anything, and forgetting it is
    the most common reason a freshly trained LoRA "does nothing".

    Parameters
    ----------
    prompt:
        Generation prompt.
    spec:
        Adapter in use, or ``None``.

    Returns
    -------
    str

    Examples
    --------
    >>> spec = LoraSpec(name="x", path=Path("."), backend=BackendId.QWEN_EDIT, trigger="aitrsaree")
    >>> apply_trigger("Dress the person...", spec).startswith("aitrsaree,")
    True
    """
    if spec is None or not spec.trigger:
        return prompt
    if spec.trigger.lower() in prompt.lower():
        return prompt
    return f"{spec.trigger}, {prompt}"


def load_into_pipeline(pipeline: Any, spec: LoraSpec, *, weight: float = 1.0) -> bool:
    """Attach ``spec`` to a loaded diffusers pipeline.

    Tries the diffusers LoRA loader first, then peft directly, because adapters
    saved by ``save_pretrained`` and by ``save_lora_weights`` have different
    layouts and both are common.

    Parameters
    ----------
    pipeline:
        A loaded diffusers pipeline.
    spec:
        Adapter to load.
    weight:
        Adapter scale, 0-1.5. Above ~1.2 a drape LoRA starts overriding the
        garment reference.

    Returns
    -------
    bool
        True when the adapter was attached. Never raises - a failed LoRA load
        should fall back to the base model, not lose the user's generation.
    """
    if spec.unverified:
        logger.warning(
            "LoRA %s has no %s, so its base model is unverified. If output is "
            "noise, it was trained for a different base.",
            spec.name,
            SIDECAR_NAME,
        )

    loader = getattr(pipeline, "load_lora_weights", None)
    if loader is not None:
        try:
            loader(str(spec.path), adapter_name=spec.name)
            _set_adapter_scale(pipeline, spec.name, weight)
            logger.info("Loaded LoRA via diffusers: %s (weight %.2f)", spec.describe(), weight)
            return True
        except Exception as exc:  # noqa: BLE001 - layout mismatches are expected
            logger.debug("diffusers load_lora_weights failed for %s: %s", spec.name, exc)

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        logger.warning("Pipeline has no transformer; cannot load LoRA %s.", spec.name)
        return False

    try:
        from peft import PeftModel

        pipeline.transformer = PeftModel.from_pretrained(
            transformer, str(spec.path), adapter_name=spec.name
        )
        logger.info("Loaded LoRA via peft: %s", spec.describe())
        return True
    except ImportError:
        logger.warning("peft is not installed; cannot load LoRA %s.", spec.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load LoRA %s: %s", spec.name, exc)

    return False


def _set_adapter_scale(pipeline: Any, name: str, weight: float) -> None:
    """Set an adapter's scale, ignoring pipelines that do not support it."""
    setter = getattr(pipeline, "set_adapters", None)
    if setter is None:
        return
    try:
        setter([name], adapter_weights=[float(weight)])
    except Exception as exc:  # noqa: BLE001
        logger.debug("set_adapters failed for %s: %s", name, exc)


def unload_from_pipeline(pipeline: Any) -> None:
    """Detach any LoRA from ``pipeline``, restoring base behaviour."""
    for method in ("unload_lora_weights", "disable_lora"):
        func = getattr(pipeline, method, None)
        if func is None:
            continue
        try:
            func()
            logger.info("Unloaded LoRA via %s().", method)
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s() failed: %s", method, exc)


def lora_choices(backend: BackendId | None = None) -> list[str]:
    """Build the UI dropdown labels, optionally filtered to one backend.

    Returns
    -------
    list[str]
        :data:`NO_LORA_LABEL` first, then compatible adapters.
    """
    labels = [NO_LORA_LABEL]
    for spec in cached_loras():
        if backend is None or spec.is_compatible(backend):
            labels.append(spec.label)
    return labels
