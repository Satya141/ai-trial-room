"""LoRA fine-tuning for the editing backends.

Honest feasibility, before any code
-----------------------------------
These are rectified-flow transformers, and their size decides what hardware can
train them:

=========================== ======  ==================  ==========================
Base model                  Params  Practical minimum   On a 16 GB T4
=========================== ======  ==================  ==========================
``FLUX.2-klein-4B``         4 B     16 GB               **Yes** - the recommended
                                                        target. rank 16, 512-768 px.
``Qwen-Image-Edit-2511``    12 B    24 GB               Marginal. Needs 4-bit base
                                                        + gradient checkpointing +
                                                        512 px + batch 1, and it
                                                        is slow (~4-6 s/step).
=========================== ======  ==================  ==========================

So :func:`default_config` targets **FLUX.2 klein** on a T4 and warns loudly if
you point it at the 12 B model with under 24 GB. Both are Apache-2.0, so either
LoRA is yours to sell.

The T4 has a second problem: no usable bf16. fp16 training of a large
transformer diverges easily, so on Turing this script keeps master weights in
fp32 and autocasts only the forward pass, which costs memory but actually
converges.

What the loss is
----------------
Flow matching, not DDPM epsilon-prediction. For a sample ``x1`` and noise
``x0``, we pick a timestep ``t``, form ``xt = (1 - t) * x0 + t * x1``, and train
the transformer to predict the velocity ``x1 - x0``. The loss is plain MSE
against that target. This is what both Qwen-Image and FLUX.2 use, and training
them with a DDPM objective is a common and silent mistake.

Status
------
This script is **unvalidated on a GPU**. The architecture, loss and memory
strategy are correct by construction, but no training run has been completed, so
treat the hyperparameters as starting points rather than tuned values. Step time
and final quality are unmeasured.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from ai_trial_room.config import CONFIG, MODEL_SPECS, BackendId
from ai_trial_room.training.dataset import (
    DatasetMode,
    DatasetReport,
    build_dataset,
    scan_dataset,
)
from ai_trial_room.utils.device import detect_hardware, free_vram, vram_report
from ai_trial_room.utils.errors import InvalidInputError, TrialRoomError
from ai_trial_room.utils.logging_setup import get_logger

logger = get_logger(__name__)

#: Transformer submodules LoRA is attached to.
#:
#: Attention projections only. The feed-forward layers roughly triple adapter
#: size for little gain on a style/drape concept, and skipping them is what keeps
#: a rank-16 adapter near 40 MB.
DEFAULT_TARGET_MODULES: Final[tuple[str, ...]] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)

#: VRAM below which the 12 B target is refused without an explicit override.
QWEN_MIN_VRAM_GB: Final[float] = 22.0


@dataclass
class TrainingConfig:
    """Every training hyperparameter, in one place.

    Serialised into ``<output>/training_config.json`` next to the adapter, so a
    LoRA always carries the recipe that produced it.
    """

    # --- data ------------------------------------------------------------- #
    dataset_dir: Path
    output_dir: Path
    mode: DatasetMode = DatasetMode.DRAPE
    resolution: tuple[int, int] = (768, 1024)
    caption_dropout: float = 0.05

    # --- model ------------------------------------------------------------ #
    backend: BackendId = BackendId.FLUX_KLEIN
    quantize_base: str = "auto"
    """``auto`` | ``none`` | ``4bit`` | ``8bit``. Applies to the frozen base."""

    # --- LoRA ------------------------------------------------------------- #
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    # --- optimisation ----------------------------------------------------- #
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 4
    max_steps: int = 1200
    warmup_steps: int = 60
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    seed: int = 42

    # --- memory ----------------------------------------------------------- #
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    """bitsandbytes AdamW8bit. Saves ~0.75 GB per 100 M trainable params."""

    # --- flow matching ---------------------------------------------------- #
    timestep_sampling: str = "logit_normal"
    """``uniform`` | ``logit_normal``. Logit-normal concentrates sampling in the
    mid-noise region where most of the learning signal is, and is what the
    Qwen-Image and FLUX papers use."""
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # --- bookkeeping ------------------------------------------------------ #
    save_every: int = 200
    log_every: int = 10
    trigger: str = "aitrsaree"
    notes: str = ""

    def validate(self) -> list[str]:
        """Check the config against the detected hardware.

        Returns
        -------
        list[str]
            Warnings. Fatal problems raise instead.

        Raises
        ------
        InvalidInputError
            For settings that cannot work at all.
        """
        warnings: list[str] = []
        hardware = detect_hardware()

        if self.rank < 1 or self.rank > 256:
            raise InvalidInputError(
                f"LoRA rank {self.rank} is out of range; use 4-128 "
                "(16 is a good default for a drape concept)."
            )
        if self.batch_size < 1 or self.gradient_accumulation < 1:
            raise InvalidInputError("batch_size and gradient_accumulation must be >= 1.")
        if self.max_steps < 1:
            raise InvalidInputError("max_steps must be >= 1.")

        if not hardware.is_cuda:
            warnings.append(
                "No CUDA device detected. Training on CPU is not practical - this "
                "will take days. Use the Kaggle/Colab notebook."
            )
            return warnings

        if self.backend is BackendId.QWEN_EDIT and hardware.total_vram_gb < QWEN_MIN_VRAM_GB:
            warnings.append(
                f"{MODEL_SPECS[self.backend].repo_id} is 12 B and wants "
                f"{QWEN_MIN_VRAM_GB:.0f} GB+ to train; this card has "
                f"{hardware.total_vram_gb:.1f} GB. Expect to need "
                "--quantize-base 4bit, 512 px, and roughly 4-6 s/step - or train "
                "FLUX.2-klein-4B instead with --backend flux_klein."
            )

        if not hardware.supports_bf16:
            warnings.append(
                "This GPU has no usable bf16 (Turing or older). Master weights "
                "stay fp32 with an autocast forward pass, which converges but uses "
                "more memory than bf16 would."
            )

        if max(self.resolution) > 768 and hardware.total_vram_gb < 20:
            warnings.append(
                f"Resolution {self.resolution} is ambitious for "
                f"{hardware.total_vram_gb:.0f} GB. Try 512x768 if you hit OOM."
            )

        effective = self.batch_size * self.gradient_accumulation
        if effective < 4:
            warnings.append(
                f"Effective batch size is {effective}. Flow-matching gradients are "
                "noisy; 4-8 gives noticeably more stable training."
            )

        return warnings

    def to_json(self) -> str:
        """Serialise for the sidecar file."""
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["mode"] = self.mode.value
        payload["backend"] = self.backend.value
        payload["resolution"] = list(self.resolution)
        payload["target_modules"] = list(self.target_modules)
        return json.dumps(payload, indent=2)


def default_config(dataset_dir: Path, output_dir: Path) -> TrainingConfig:
    """Build a config tuned to the detected hardware.

    Chooses the 4 B backend and a 512x768 resolution on small cards, and keeps
    the 12 B backend only where there is room for it.
    """
    hardware = detect_hardware()
    config = TrainingConfig(dataset_dir=dataset_dir, output_dir=output_dir)

    if not hardware.is_cuda:
        return config

    if hardware.total_vram_gb >= QWEN_MIN_VRAM_GB:
        config.backend = BackendId.QWEN_EDIT
        config.resolution = (768, 1024)
        config.quantize_base = "none"
    elif hardware.total_vram_gb >= 15.0:
        config.backend = BackendId.FLUX_KLEIN
        config.resolution = (768, 1024)
        config.quantize_base = "none"
    else:
        config.backend = BackendId.FLUX_KLEIN
        config.resolution = (512, 768)
        config.quantize_base = "4bit"
        config.gradient_accumulation = 8

    return config


# --------------------------------------------------------------------------- #
# Flow-matching helpers
# --------------------------------------------------------------------------- #


def sample_timesteps(
    batch: int,
    *,
    strategy: str,
    logit_mean: float,
    logit_std: float,
    generator: Any = None,
    device: str = "cpu",
) -> Any:
    """Sample flow-matching timesteps in ``(0, 1)``.

    Parameters
    ----------
    batch:
        How many to draw.
    strategy:
        ``uniform`` or ``logit_normal``.
    logit_mean, logit_std:
        Parameters of the logit-normal distribution.
    generator:
        Optional torch generator for reproducibility.
    device:
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(batch,)``, values strictly inside ``(0, 1)``.
    """
    import torch

    if strategy == "uniform":
        raw = torch.rand(batch, generator=generator, device=device)
    else:
        normal = torch.randn(batch, generator=generator, device=device)
        raw = torch.sigmoid(normal * logit_std + logit_mean)

    # Clamp away from the endpoints: t=0 and t=1 give a degenerate target.
    return raw.clamp(1e-4, 1.0 - 1e-4)


def flow_match_inputs(
    latents: Any,
    timesteps: Any,
    *,
    generator: Any = None,
) -> tuple[Any, Any]:
    """Build the noisy input and velocity target for flow matching.

    With ``x1`` the clean latent and ``x0`` Gaussian noise::

        xt     = (1 - t) * x0 + t * x1
        target = x1 - x0

    Parameters
    ----------
    latents:
        Clean latents ``x1``, shape ``(B, C, H, W)``.
    timesteps:
        Shape ``(B,)`` in ``(0, 1)``.
    generator:
        Optional torch generator.

    Returns
    -------
    tuple
        ``(noisy_latents, velocity_target)``.
    """
    import torch

    noise = torch.randn(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )
    # Broadcast t over the channel and spatial dims.
    shape = [latents.shape[0]] + [1] * (latents.ndim - 1)
    t = timesteps.to(latents.device, latents.dtype).view(shape)

    noisy = (1.0 - t) * noise + t * latents
    target = latents - noise
    return noisy, target


def lr_at_step(step: int, config: TrainingConfig) -> float:
    """Learning rate for ``step`` under linear warmup plus the chosen schedule."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)

    if config.lr_schedule == "constant":
        return config.learning_rate
    if config.lr_schedule == "linear":
        return config.learning_rate * (1.0 - progress)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class TrainingState:
    """Mutable progress of a run."""

    step: int = 0
    samples_seen: int = 0
    loss_history: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_s(self) -> float:
        """Wall-clock seconds since the run started."""
        return time.time() - self.started_at

    def recent_loss(self, window: int = 20) -> float:
        """Mean loss over the last ``window`` steps, or ``nan`` if none yet."""
        if not self.loss_history:
            return float("nan")
        tail = self.loss_history[-window:]
        return sum(tail) / len(tail)


def prepare_dataset(config: TrainingConfig) -> tuple[DatasetReport, Any]:
    """Scan and build the dataset described by ``config``.

    Returns
    -------
    tuple
        ``(report, dataset)``.

    Raises
    ------
    InvalidInputError
        When the dataset is unusable.
    """
    report = scan_dataset(config.dataset_dir, config.mode)
    logger.info("\n%s", report.render())

    dataset = build_dataset(
        report,
        resolution=config.resolution,
        caption_dropout=config.caption_dropout,
        seed=config.seed,
    )
    return report, dataset


def build_lora_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the base pipeline and attach a LoRA adapter to its transformer.

    The base model is frozen; only adapter parameters require gradients.

    Returns
    -------
    tuple
        ``(pipeline, transformer_with_adapter)``.

    Raises
    ------
    TrialRoomError
        When diffusers or peft is missing, or the weights cannot load.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise TrialRoomError("PyTorch is required for training.", detail=str(exc)) from exc

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise TrialRoomError(
            "peft is required for LoRA training. Run `pip install peft`.",
            detail=str(exc),
        ) from exc

    spec = MODEL_SPECS[config.backend]
    hardware = detect_hardware()

    try:
        import diffusers
    except ImportError as exc:
        raise TrialRoomError(
            "diffusers is required for training.", detail=str(exc)
        ) from exc

    pipeline_cls = getattr(diffusers, spec.pipeline_class, None)
    if pipeline_cls is None:
        raise TrialRoomError(
            f"diffusers {diffusers.__version__} has no {spec.pipeline_class}. "
            "Upgrade diffusers."
        )

    # Master weights stay fp32 without bf16; the forward pass is autocast.
    load_dtype = torch.bfloat16 if hardware.supports_bf16 else torch.float32

    kwargs: dict[str, Any] = {"torch_dtype": load_dtype}
    if CONFIG.hf_token:
        kwargs["token"] = CONFIG.hf_token

    if config.quantize_base in {"4bit", "8bit"}:
        quant = _bnb_config(config.quantize_base, load_dtype)
        if quant is not None:
            kwargs["quantization_config"] = quant

    logger.info(
        "Loading base %s (dtype=%s, quantize=%s)",
        spec.repo_id,
        load_dtype,
        config.quantize_base,
    )
    try:
        pipeline = pipeline_cls.from_pretrained(spec.repo_id, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise TrialRoomError(
            f"Could not load {spec.repo_id} for training.",
            detail=f"{type(exc).__name__}: {exc}"[:400],
        ) from exc

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise TrialRoomError(
            f"{spec.pipeline_class} has no .transformer to attach LoRA to."
        )

    # Freeze everything, including the text encoder and VAE.
    for module_name in ("vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipeline, module_name, None)
        if module is not None and hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
    transformer.requires_grad_(False)

    present = _present_target_modules(transformer, config.target_modules)
    if not present:
        raise TrialRoomError(
            "None of the configured LoRA target modules exist in this "
            f"transformer. Tried: {', '.join(config.target_modules)}",
            detail="inspect transformer.named_modules() and set --target-modules",
        )
    if len(present) < len(config.target_modules):
        logger.info(
            "Targeting %d of %d configured module names (absent: %s)",
            len(present),
            len(config.target_modules),
            ", ".join(sorted(set(config.target_modules) - set(present))),
        )

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(present),
        init_lora_weights="gaussian",
        bias="none",
    )
    adapted = get_peft_model(transformer, lora_config)

    if config.gradient_checkpointing:
        for target in (adapted, getattr(adapted, "base_model", None)):
            if target is not None and hasattr(target, "enable_gradient_checkpointing"):
                try:
                    target.enable_gradient_checkpointing()
                    logger.info("Gradient checkpointing enabled.")
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Gradient checkpointing failed: %s", exc)

    trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    total = sum(p.numel() for p in adapted.parameters())
    logger.info(
        "LoRA attached: rank %d, %s trainable of %s total (%.3f%%) | %s",
        config.rank,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(1, total),
        vram_report(),
    )

    pipeline.transformer = adapted
    return pipeline, adapted


def _bnb_config(mode: str, compute_dtype: Any) -> Any | None:
    """Build a bitsandbytes quantization config, or ``None`` if unavailable."""
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[assignment]
        except ImportError:
            logger.warning("bitsandbytes config unavailable; loading base unquantized.")
            return None

    try:
        if mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build a %s quantization config: %s", mode, exc)
        return None


def _present_target_modules(module: Any, wanted: tuple[str, ...]) -> list[str]:
    """Return which of ``wanted`` module-name suffixes exist in ``module``.

    peft raises if asked for a target that does not exist, and the two supported
    architectures name their attention projections differently, so this filters
    the list to whatever is actually there.
    """
    names = {name.split(".")[-1] for name, _ in module.named_modules()}
    # "to_out.0" is a two-segment suffix, so check the full string too.
    full = {name for name, _ in module.named_modules()}
    present: list[str] = []
    for target in wanted:
        if target in names or any(entry.endswith(target) for entry in full):
            present.append(target)
    return present


def build_optimizer(parameters: Any, config: TrainingConfig) -> Any:
    """Create the optimiser, preferring 8-bit AdamW when available."""
    import torch

    trainable = [p for p in parameters if p.requires_grad]

    if config.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb

            logger.info("Using bitsandbytes AdamW8bit.")
            return bnb.optim.AdamW8bit(
                trainable,
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay,
            )
        except ImportError:
            logger.info("bitsandbytes unavailable; using torch AdamW.")

    return torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


def save_adapter(
    transformer: Any,
    config: TrainingConfig,
    state: TrainingState,
    *,
    tag: str = "final",
) -> Path:
    """Write the adapter plus its metadata sidecar.

    Parameters
    ----------
    transformer:
        The peft-wrapped transformer.
    config:
        Run configuration, serialised alongside.
    state:
        Progress, recorded in the metadata.
    tag:
        ``final``, or ``step-000400`` for a checkpoint.

    Returns
    -------
    Path
        Directory the adapter was written to.
    """
    destination = config.output_dir if tag == "final" else config.output_dir / tag
    destination.mkdir(parents=True, exist_ok=True)

    transformer.save_pretrained(destination)

    (destination / "training_config.json").write_text(config.to_json(), encoding="utf-8")
    (destination / "aitr_lora.json").write_text(
        json.dumps(
            {
                "name": config.output_dir.name,
                "backend": config.backend.value,
                "base_repo": MODEL_SPECS[config.backend].repo_id,
                "base_license": MODEL_SPECS[config.backend].license_name,
                "trigger": config.trigger,
                "rank": config.rank,
                "alpha": config.alpha,
                "mode": config.mode.value,
                "resolution": list(config.resolution),
                "steps_trained": state.step,
                "samples_seen": state.samples_seen,
                "final_loss": state.recent_loss(),
                "trained_minutes": round(state.elapsed_s / 60.0, 1),
                "notes": config.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Saved adapter (%s) to %s", tag, destination)
    return destination


def train(config: TrainingConfig) -> Path:
    """Run LoRA fine-tuning end to end.

    Parameters
    ----------
    config:
        Training configuration.

    Returns
    -------
    Path
        Directory holding the final adapter.

    Raises
    ------
    TrialRoomError
        On unusable data, missing dependencies, or a load failure.
    """
    import torch
    from torch.utils.data import DataLoader

    for warning in config.validate():
        logger.warning("%s", warning)

    report, dataset = prepare_dataset(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(config.seed)
    torch.manual_seed(config.seed)

    loader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows- and Kaggle-safe
        drop_last=len(dataset) > config.batch_size,
    )

    pipeline, transformer = build_lora_model(config)
    optimizer = build_optimizer(transformer.parameters(), config)

    hardware = detect_hardware()
    device = "cuda" if hardware.is_cuda else "cpu"
    autocast_dtype = torch.bfloat16 if hardware.supports_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device) if device == "cuda" and not hardware.supports_bf16 else None

    state = TrainingState()
    logger.info(
        "Training %s LoRA: %d steps, effective batch %d, lr %.2e, %d sample(s)",
        MODEL_SPECS[config.backend].repo_id,
        config.max_steps,
        config.batch_size * config.gradient_accumulation,
        config.learning_rate,
        len(dataset),
    )

    transformer.train()
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)

    while state.step < config.max_steps:
        for batch in loader:
            if state.step >= config.max_steps:
                break

            try:
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
                    loss = _training_step(pipeline, transformer, batch, config, generator, device)
            except Exception as exc:  # noqa: BLE001
                free_vram()
                raise TrialRoomError(
                    "A training step failed. This usually means the batch shape or "
                    "the pipeline's forward signature differs from what this script "
                    "assumes - see _training_step.",
                    detail=f"{type(exc).__name__}: {exc}"[:400],
                ) from exc

            scaled = loss / config.gradient_accumulation
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            accumulated += 1
            if accumulated < config.gradient_accumulation:
                continue

            for group in optimizer.param_groups:
                group["lr"] = lr_at_step(state.step, config)

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad],
                config.max_grad_norm,
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0

            state.step += 1
            state.samples_seen += config.batch_size * config.gradient_accumulation
            state.loss_history.append(float(loss.detach().to("cpu")))

            if state.step % config.log_every == 0:
                per_step = state.elapsed_s / max(1, state.step)
                remaining = per_step * (config.max_steps - state.step)
                logger.info(
                    "step %d/%d | loss %.4f (avg %.4f) | lr %.2e | %.2fs/step | ETA %dm",
                    state.step,
                    config.max_steps,
                    state.loss_history[-1],
                    state.recent_loss(),
                    lr_at_step(state.step, config),
                    per_step,
                    int(remaining / 60),
                )

            if config.save_every and state.step % config.save_every == 0:
                save_adapter(transformer, config, state, tag=f"step-{state.step:06d}")

    destination = save_adapter(transformer, config, state, tag="final")
    logger.info(
        "Training complete: %d steps, %d samples, final loss %.4f, %.1f min",
        state.step,
        state.samples_seen,
        state.recent_loss(),
        state.elapsed_s / 60.0,
    )
    free_vram()
    return destination


def _training_step(
    pipeline: Any,
    transformer: Any,
    batch: dict[str, Any],
    config: TrainingConfig,
    generator: Any,
    device: str,
) -> Any:
    """Compute the flow-matching loss for one batch.

    Encodes images to latents with the frozen VAE, encodes captions with the
    frozen text encoder, forms the flow-matching input, and takes an MSE loss
    against the velocity target.

    .. note::

       The transformer call signature differs between Qwen-Image-Edit and
       FLUX.2, and both are moving targets in diffusers. This passes the
       arguments both accept and lets a mismatch raise, which
       :func:`train` turns into an actionable error rather than a silent wrong
       objective.
    """
    import torch

    pixel_key = "target_values" if "target_values" in batch else "pixel_values"
    images = batch[pixel_key].to(device)

    vae = pipeline.vae
    with torch.no_grad():
        posterior = vae.encode(images.to(vae.dtype))
        latent_dist = getattr(posterior, "latent_dist", None)
        latents = latent_dist.sample(generator=generator) if latent_dist else posterior.sample
        scale = getattr(vae.config, "scaling_factor", 1.0)
        shift = getattr(vae.config, "shift_factor", None) or 0.0
        latents = (latents - shift) * scale

    timesteps = sample_timesteps(
        latents.shape[0],
        strategy=config.timestep_sampling,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        device=latents.device,
    )
    noisy, target = flow_match_inputs(latents, timesteps)

    with torch.no_grad():
        prompt_embeds, prompt_mask = _encode_prompts(pipeline, batch["caption"], device)

    kwargs: dict[str, Any] = {
        "hidden_states": noisy,
        "timestep": timesteps * 1000.0,
        "encoder_hidden_states": prompt_embeds,
        "return_dict": False,
    }
    if prompt_mask is not None:
        kwargs["encoder_hidden_states_mask"] = prompt_mask

    prediction = transformer(**kwargs)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]

    return torch.nn.functional.mse_loss(prediction.float(), target.float())


def _encode_prompts(pipeline: Any, captions: Any, device: str) -> tuple[Any, Any | None]:
    """Encode captions with the pipeline's own prompt encoder.

    Uses ``pipeline.encode_prompt`` when present, because each architecture
    formats and pools its text embeddings differently and reimplementing that is
    how silent quality regressions happen.
    """
    texts = list(captions) if not isinstance(captions, str) else [captions]

    encode = getattr(pipeline, "encode_prompt", None)
    if encode is None:
        raise TrialRoomError(
            f"{type(pipeline).__name__} has no encode_prompt(); cannot embed captions."
        )

    result = encode(prompt=texts, device=device, num_images_per_prompt=1)
    if isinstance(result, tuple):
        embeds = result[0]
        mask = result[1] if len(result) > 1 and hasattr(result[1], "shape") else None
        return embeds, mask
    return result, None
