"""Train a saree drape LoRA.

Quick start
-----------
::

    # 1. Put images in datasets/saree_drapes/, named nivi_001.jpg etc.
    # 2. Caption them
    python scripts/caption_dataset.py --dataset datasets/saree_drapes

    # 3. Check the plan without loading any weights
    python scripts/train_saree_lora.py --dataset datasets/saree_drapes --dry-run

    # 4. Train
    python scripts/train_saree_lora.py --dataset datasets/saree_drapes

The trained adapter lands in ``loras/<name>/`` and is picked up automatically by
the Gradio UI's LoRA dropdown on next start.

Hardware reality
----------------
====================  ======  ==================  ===========================
Base model            Params  Practical minimum   On a 16 GB T4
====================  ======  ==================  ===========================
FLUX.2-klein-4B       4 B     16 GB               Yes - the default target
Qwen-Image-Edit-2511  12 B    24 GB               Marginal; needs 4-bit + 512 px
====================  ======  ==================  ===========================

``--dry-run`` prints the resolved plan and the hardware warnings, and is the
first thing to run. Both base models are Apache-2.0, so either adapter is yours
to sell.

Not yet validated on a GPU: the architecture, loss and memory strategy are
correct by construction, but no training run has completed, so the
hyperparameters are starting points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.config import PROJECT_ROOT, BackendId  # noqa: E402
from ai_trial_room.training.caption import DEFAULT_TRIGGER, caption_stats  # noqa: E402
from ai_trial_room.training.dataset import (  # noqa: E402
    DatasetMode,
    detect_mode,
    scan_dataset,
)
from ai_trial_room.training.train_lora import (  # noqa: E402
    TrainingConfig,
    default_config,
    train,
)
from ai_trial_room.utils.device import detect_hardware  # noqa: E402
from ai_trial_room.utils.errors import TrialRoomError  # noqa: E402
from ai_trial_room.utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Train a saree drape LoRA for the editing backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=Path, required=True)
    data.add_argument(
        "--mode", default=None, choices=[m.value for m in DatasetMode],
        help="Dataset layout. Auto-detected from the directory when omitted.",
    )
    data.add_argument("--name", default="saree-drape-v1", help="Adapter name.")
    data.add_argument(
        "--output", type=Path, default=None,
        help="Output directory. Defaults to loras/<name>.",
    )
    data.add_argument("--width", type=int, default=None)
    data.add_argument("--height", type=int, default=None)
    data.add_argument("--caption-dropout", type=float, default=None)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--backend", default=None, choices=[BackendId.FLUX_KLEIN.value, BackendId.QWEN_EDIT.value],
        help="Base model. Auto-selected from available VRAM when omitted.",
    )
    model.add_argument(
        "--quantize-base", default=None, choices=["auto", "none", "4bit", "8bit"],
        help="Quantize the frozen base to save VRAM.",
    )

    lora = parser.add_argument_group("lora")
    lora.add_argument("--rank", type=int, default=None)
    lora.add_argument("--alpha", type=int, default=None)
    lora.add_argument("--trigger", default=DEFAULT_TRIGGER)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--lr", type=float, default=None)
    optim.add_argument("--batch-size", type=int, default=None)
    optim.add_argument("--grad-accum", type=int, default=None)
    optim.add_argument("--steps", type=int, default=None)
    optim.add_argument("--warmup", type=int, default=None)
    optim.add_argument("--seed", type=int, default=None)
    optim.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Faster but uses much more VRAM.",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--save-every", type=int, default=None)
    run.add_argument("--notes", default="", help="Free text stored in the sidecar.")
    run.add_argument(
        "--dry-run", action="store_true",
        help="Validate the dataset and print the plan without loading weights.",
    )
    run.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Resolve CLI arguments into a :class:`TrainingConfig`.

    Starts from :func:`default_config`, which is hardware-aware, then applies any
    explicit overrides.
    """
    output = args.output or (PROJECT_ROOT / "loras" / args.name)
    config = default_config(args.dataset, output)

    config.mode = DatasetMode(args.mode) if args.mode else detect_mode(args.dataset)
    config.trigger = args.trigger
    config.notes = args.notes

    if args.backend:
        config.backend = BackendId(args.backend)
    if args.quantize_base:
        config.quantize_base = args.quantize_base

    width = args.width or config.resolution[0]
    height = args.height or config.resolution[1]
    config.resolution = (width, height)

    for attribute, value in (
        ("caption_dropout", args.caption_dropout),
        ("rank", args.rank),
        ("alpha", args.alpha),
        ("learning_rate", args.lr),
        ("batch_size", args.batch_size),
        ("gradient_accumulation", args.grad_accum),
        ("max_steps", args.steps),
        ("warmup_steps", args.warmup),
        ("seed", args.seed),
        ("save_every", args.save_every),
    ):
        if value is not None:
            setattr(config, attribute, value)

    if args.alpha is None and args.rank is not None:
        config.alpha = config.rank

    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False

    return config


def print_plan(config: TrainingConfig, warnings: list[str]) -> None:
    """Print the resolved training plan."""
    from ai_trial_room.config import MODEL_SPECS

    spec = MODEL_SPECS[config.backend]
    hardware = detect_hardware()
    effective = config.batch_size * config.gradient_accumulation

    print()
    print("Training plan")
    print("-" * 62)
    print(f"  hardware        : {hardware.describe()}")
    print(f"  base model      : {spec.repo_id}")
    print(f"  base licence    : {spec.license_name} (commercial use OK)")
    print(f"  dataset         : {config.dataset_dir}  [{config.mode.value}]")
    print(f"  output          : {config.output_dir}")
    print(f"  resolution      : {config.resolution[0]}x{config.resolution[1]}")
    print(f"  quantize base   : {config.quantize_base}")
    print(f"  lora            : rank {config.rank}, alpha {config.alpha}")
    print(f"  trigger         : {config.trigger!r}")
    print(f"  steps           : {config.max_steps} (warmup {config.warmup_steps})")
    print(f"  batch           : {config.batch_size} x {config.gradient_accumulation} = {effective}")
    print(f"  lr              : {config.learning_rate:.2e} ({config.lr_schedule})")
    print(f"  grad checkpoint : {config.gradient_checkpointing}")
    print(f"  timestep sample : {config.timestep_sampling}")
    print("-" * 62)

    for warning in warnings:
        print(f"  warn: {warning}")
    if warnings:
        print("-" * 62)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    if not args.dataset.is_dir():
        logger.error("Dataset directory does not exist: %s", args.dataset)
        return 2

    try:
        config = build_config(args)
        warnings = config.validate()
    except TrialRoomError as exc:
        logger.error("%s", exc.user_message)
        return 2

    report = scan_dataset(config.dataset_dir, config.mode)
    print()
    print(report.render())

    if report.mode is DatasetMode.DRAPE and report.drape_samples:
        stats = caption_stats(report.drape_samples)
        if stats:
            print(
                f"Captions: mean {stats['mean_words']:.0f} words, "
                f"{stats['unique_ratio'] * 100:.0f}% unique, "
                f"{stats['trigger_coverage'] * 100:.0f}% carry the trigger"
            )

    print_plan(config, warnings)

    if not report.ok:
        logger.error("Dataset is not usable; fix the errors above before training.")
        return 1

    if args.dry_run:
        print("Dry run - nothing trained. Drop --dry-run to start.")
        return 0

    if not detect_hardware().is_cuda:
        logger.error(
            "No CUDA device available. Training on CPU is not practical - use "
            "notebooks/run_on_kaggle.ipynb on a T4."
        )
        return 2

    try:
        destination = train(config)
    except TrialRoomError as exc:
        logger.error("Training failed: %s", exc.user_message)
        if exc.detail:
            logger.error("  detail: %s", exc.detail)
        return 1

    print()
    print(f"Adapter written to {destination}")
    print("It will appear in the Gradio LoRA dropdown on next start.")
    print(f"Remember the trigger token: {config.trigger!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
