"""Deploy AI Trial Room to a Hugging Face Space.

Usage
-----
See exactly what would be uploaded, without touching the Hub::

    python scripts/deploy_space.py --repo your-name/ai-trial-room --dry-run

Create the Space (if needed) and push::

    export HF_TOKEN=hf_...
    python scripts/deploy_space.py --repo your-name/ai-trial-room

Push to a private Space on ZeroGPU::

    python scripts/deploy_space.py --repo your-name/ai-trial-room --private --hardware zero-a10g

What gets uploaded
------------------
The application package, ``app.py``, and the contents of ``space/`` promoted to
the Space root — so ``space/README.md`` becomes the Space card (its YAML front
matter is what configures the Space) and ``space/requirements.txt`` replaces the
project's root requirements.

Deliberately **not** uploaded: training code, tests, notebooks, datasets, trained
adapters, ``.env``, and anything gitignored. A Space is public infrastructure;
training photos and API tokens must never reach it.

Secrets
-------
``HF_TOKEN`` is read from the environment and used to authenticate. It is never
written into the Space. If your Space needs a token at runtime, set it as a Space
*secret* in the Space's own settings UI — this script will not do it for you,
because a token pushed as a file would be committed to the Space's git history.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trial_room.config import PROJECT_ROOT  # noqa: E402
from ai_trial_room.utils.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Directory whose contents are promoted to the Space root.
SPACE_DIR: Final[Path] = PROJECT_ROOT / "space"

#: Files and directories uploaded from the project root, relative to it.
INCLUDE_PATHS: Final[tuple[str, ...]] = (
    "app.py",
    "ai_trial_room",
    "assets/examples",
)

#: Never uploaded, matched against any path segment.
EXCLUDE_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        ".cache",
        ".hf",
        "datasets",
        "loras",
        "outputs",
        "tests",
        "notebooks",
        "third_party",
        ".venv",
        "venv",
    }
)

#: Never uploaded, matched against the filename.
#:
#: The project's root ``requirements.txt`` is not listed here because
#: :data:`INCLUDE_PATHS` never scans the root, so it is already excluded - and
#: listing it would also block the promoted ``space/requirements.txt``, which the
#: Space cannot start without.
EXCLUDE_NAMES: Final[frozenset[str]] = frozenset(
    {".env", ".env.local", ".DS_Store", "Thumbs.db"}
)

#: Never uploaded, matched against the suffix.
EXCLUDE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pyc", ".pyo", ".safetensors", ".ckpt", ".pth", ".bin", ".onnx", ".tflite", ".ipynb"}
)

#: Hardware tiers that can be requested for a Space.
HARDWARE_CHOICES: Final[tuple[str, ...]] = (
    "cpu-basic",
    "cpu-upgrade",
    "zero-a10g",
    "t4-small",
    "t4-medium",
    "l4x1",
    "a10g-small",
    "a100-large",
)


@dataclass(frozen=True)
class Upload:
    """One file to upload: where it is locally, and where it lands in the Space."""

    local: Path
    remote: str

    def size_kb(self) -> float:
        """File size in kilobytes."""
        return self.local.stat().st_size / 1024.0


def _is_excluded(path: Path, root: Path) -> bool:
    """True when ``path`` must not be uploaded."""
    relative = path.relative_to(root)
    if any(segment in EXCLUDE_SEGMENTS for segment in relative.parts):
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def collect_uploads(project_root: Path = PROJECT_ROOT) -> list[Upload]:
    """Build the upload manifest.

    ``space/`` contents are promoted to the Space root and take precedence, so
    ``space/requirements.txt`` becomes the Space's ``requirements.txt`` while the
    project's root one is excluded.

    Parameters
    ----------
    project_root:
        Repository root.

    Returns
    -------
    list[Upload]
        Sorted by remote path, with no duplicates.
    """
    uploads: dict[str, Upload] = {}

    for entry in INCLUDE_PATHS:
        source = project_root / entry
        if not source.exists():
            logger.warning("Skipping %s (not found)", entry)
            continue

        if source.is_file():
            if not _is_excluded(source, project_root):
                uploads[entry] = Upload(local=source, remote=entry)
            continue

        for path in sorted(source.rglob("*")):
            if not path.is_file() or _is_excluded(path, project_root):
                continue
            remote = path.relative_to(project_root).as_posix()
            uploads[remote] = Upload(local=path, remote=remote)

    # space/* is promoted to the root, overriding anything above.
    if SPACE_DIR.is_dir():
        for path in sorted(SPACE_DIR.rglob("*")):
            if not path.is_file() or _is_excluded(path, SPACE_DIR):
                continue
            remote = path.relative_to(SPACE_DIR).as_posix()
            uploads[remote] = Upload(local=path, remote=remote)
    else:
        logger.error("%s does not exist; the Space would have no README card.", SPACE_DIR)

    return sorted(uploads.values(), key=lambda upload: upload.remote)


def verify_manifest(uploads: Sequence[Upload]) -> list[str]:
    """Check the manifest for problems that would break or leak.

    Returns
    -------
    list[str]
        Problems found. Empty means the manifest looks deployable.
    """
    problems: list[str] = []
    remotes = {upload.remote for upload in uploads}

    for required in ("app.py", "README.md", "requirements.txt"):
        if required not in remotes:
            problems.append(f"missing {required} - the Space will not start")

    if not any(remote.startswith("ai_trial_room/") for remote in remotes):
        problems.append("the ai_trial_room package is missing")

    # A leak check that does not depend on the exclusion lists being right.
    for upload in uploads:
        lowered = upload.remote.lower()
        if any(token in lowered for token in (".env", "token", "secret", "credential")):
            problems.append(f"possible secret in the manifest: {upload.remote}")
        if lowered.startswith(("datasets/", "loras/", "outputs/")):
            problems.append(f"training or output data in the manifest: {upload.remote}")

    total_mb = sum(upload.size_kb() for upload in uploads) / 1024.0
    if total_mb > 40.0:
        problems.append(
            f"manifest is {total_mb:.1f} MB - unexpectedly large for source code; "
            "check for stray binaries"
        )

    return problems


def print_manifest(uploads: Sequence[Upload], *, limit: int = 40) -> None:
    """Print the manifest with sizes and a total."""
    total_kb = sum(upload.size_kb() for upload in uploads)

    print()
    print(f"Files to upload: {len(uploads)}  ({total_kb / 1024.0:.2f} MB)")
    print("-" * 66)
    for upload in uploads[:limit]:
        print(f"  {upload.size_kb():8.1f} KB  {upload.remote}")
    if len(uploads) > limit:
        print(f"  ... and {len(uploads) - limit} more")
    print("-" * 66)


def deploy(
    repo_id: str,
    uploads: Iterable[Upload],
    *,
    token: str,
    private: bool,
    hardware: str | None,
    commit_message: str,
) -> str:
    """Create the Space if needed and upload every file.

    Parameters
    ----------
    repo_id:
        ``owner/name``.
    uploads:
        Manifest from :func:`collect_uploads`.
    token:
        Hugging Face token with write access.
    private:
        Create the Space private.
    hardware:
        Requested hardware tier, or ``None`` to leave it alone. **Paid tiers bill
        the account** - the caller confirms this before we get here.
    commit_message:
        Commit message for the upload.

    Returns
    -------
    str
        The Space URL.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)

    logger.info("Ensuring Space %s exists (private=%s)", repo_id, private)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )

    operations = [
        CommitOperationAdd(path_in_repo=upload.remote, path_or_fileobj=str(upload.local))
        for upload in uploads
    ]
    logger.info("Uploading %d file(s)...", len(operations))
    api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message=commit_message,
    )

    if hardware:
        logger.info("Requesting hardware %s", hardware)
        try:
            api.request_space_hardware(repo_id=repo_id, hardware=hardware)
        except Exception as exc:  # noqa: BLE001 - quota and permission errors vary
            logger.warning(
                "Could not set hardware to %s (%s). Set it in the Space settings UI.",
                hardware,
                exc,
            )

    return f"https://huggingface.co/spaces/{repo_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line."""
    parser = argparse.ArgumentParser(
        description="Deploy AI Trial Room to a Hugging Face Space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo", required=True, help="Target Space as owner/name."
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the Space private."
    )
    parser.add_argument(
        "--hardware",
        default=None,
        choices=HARDWARE_CHOICES,
        help="Request a hardware tier. Paid tiers bill your account.",
    )
    parser.add_argument(
        "--message", default="Deploy AI Trial Room", help="Commit message."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest and exit without contacting the Hub.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt. Required for non-interactive use.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging(args.log_level)

    if "/" not in args.repo:
        logger.error("--repo must be owner/name, got %r", args.repo)
        return 2

    uploads = collect_uploads()
    print_manifest(uploads)

    problems = verify_manifest(uploads)
    if problems:
        print("Manifest problems:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        logger.error("Refusing to deploy with %d unresolved problem(s).", len(problems))
        return 1

    print(f"Target Space : https://huggingface.co/spaces/{args.repo}")
    print(f"Visibility   : {'private' if args.private else 'PUBLIC'}")
    print(f"Hardware     : {args.hardware or 'unchanged (set in the Space UI)'}")
    print()

    if args.dry_run:
        print("Dry run - nothing uploaded. Drop --dry-run to deploy.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN is not set. Create a write token at "
            "https://huggingface.co/settings/tokens and export it."
        )
        return 2

    # Deploying publishes a public page and can bill for hardware, so it is
    # confirmed explicitly rather than inferred from running the command.
    if not args.yes:
        visibility = "PRIVATE" if args.private else "PUBLIC"
        billing = (
            f"\n  Hardware {args.hardware} may incur charges on your account."
            if args.hardware and not args.hardware.startswith(("cpu-basic", "zero"))
            else ""
        )
        print(
            f"About to upload {len(uploads)} file(s) to a {visibility} Space."
            f"{billing}"
        )
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return 1

    try:
        url = deploy(
            args.repo,
            uploads,
            token=token,
            private=args.private,
            hardware=args.hardware,
            commit_message=args.message,
        )
    except Exception as exc:  # noqa: BLE001 - hub errors are varied
        logger.error("Deployment failed: %s: %s", type(exc).__name__, exc)
        return 1

    print()
    print(f"Deployed: {url}")
    print()
    print("Next steps:")
    print("  1. Open the Space and watch the build log for import errors.")
    print("  2. Set hardware to ZeroGPU in Settings if you did not pass --hardware.")
    print("  3. First request loads ~20 GB of weights and takes several minutes.")
    print("  4. Add HF_TOKEN as a Space *secret* if you hit download rate limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
