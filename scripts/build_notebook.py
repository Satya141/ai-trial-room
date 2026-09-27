"""Generate ``notebooks/run_on_kaggle.ipynb``.

The notebook is generated rather than hand-written so its cells stay in sync
with the project's real setup steps, and so the JSON is always valid.

Run after changing installation or launch instructions::

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "run_on_kaggle.ipynb"

REPO_URL = "https://github.com/YOUR_USERNAME/ai-trial-room.git"


def markdown(text: str) -> dict:
    """Build a markdown cell from a block of text."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """Build an unexecuted code cell from a block of text."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
# AI Trial Room - Virtual Try-On for Indian Wear

Run the full app on a **free Kaggle T4 (16 GB)** or **Colab T4** and get a public
share link.

| | |
|---|---|
| **Primary model** | `Qwen/Qwen-Image-Edit-2511` - Apache-2.0 ✅ commercial |
| **Fast alternative** | `black-forest-labs/FLUX.2-klein-4B` - Apache-2.0 ✅ commercial |
| **Categories** | Dress, Kurti, Kurta, Saree, Lehenga |
| **Saree drapes** | Nivi, Bengali, Gujarati, Nauvari |

## Before you start

1. **Turn on the GPU.** Kaggle: *Settings → Accelerator → GPU T4 x2*.
   Colab: *Runtime → Change runtime type → T4 GPU*.
2. **Turn on the internet.** Kaggle: *Settings → Internet → On* (needed to
   download weights).
3. **Add your Hugging Face token** as a Kaggle Secret named `HF_TOKEN`, or a
   Colab secret of the same name. Not strictly required for these Apache-2.0
   models, but it avoids anonymous rate limits.

> First run downloads ~20 GB of weights and takes 10-20 minutes. Subsequent
> runs in the same session are instant.
"""
    ),
    markdown("## 1. Check the GPU"),
    code(
        """
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"""
    ),
    markdown(
        """
## 2. Install dependencies

Kaggle and Colab already ship a CUDA build of PyTorch, so we leave `torch`
alone and install only what is missing. Pinning `numpy<2` matters: MediaPipe
0.10.x is not NumPy-2 safe.
"""
    ),
    code(
        """
%%capture install_log
# Core inference stack
!pip install -q "diffusers==0.36.0" "transformers==4.57.1" "accelerate==1.10.1" \\
               "safetensors==0.6.2" "sentencepiece==0.2.0" "peft==0.17.1"

# UI + preprocessing
!pip install -q "gradio==5.49.1" "mediapipe==0.10.21" "rembg==2.0.67" \\
               "onnxruntime==1.22.0" "opencv-python-headless==4.11.0.86" \\
               "numpy==1.26.4" "huggingface-hub==0.35.3"
"""
    ),
    code(
        """
# Surface only the errors from the install above, if any.
tail = install_log.stdout.strip().splitlines()[-12:]
print("\\n".join(tail) if tail else "install completed quietly")
"""
    ),
    markdown(
        """
### Optional: Nunchaku INT4 - big VRAM saving

Nunchaku's SVDQuant weights shrink the 12 B Qwen transformer from ~40 GB (bf16)
to ~7 GB, and with async CPU offload it runs in **3-4 GB**. On a 16 GB T4 this
is the difference between 40-step generations that fit comfortably and ones that
OOM halfway through a demo.

Wheels are built per (Python, PyTorch) pair, so pick the one matching this
runtime. Skip this cell if it errors - the app detects Nunchaku's absence and
falls back to bf16 with CPU offload automatically.
"""
    ),
    code(
        """
import sys, torch

py = f"cp{sys.version_info.major}{sys.version_info.minor}"
tv = ".".join(torch.__version__.split(".")[:2])
print(f"python={py}  torch={tv}")
print("Pick the matching wheel from:")
print("  https://github.com/nunchaku-tech/nunchaku/releases")

# Example (edit the URL to match the python/torch printed above):
# !pip install -q https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp311-cp311-linux_x86_64.whl
"""
    ),
    markdown(
        """
## 3. Get the code

Either clone your repo, or upload the project as a Kaggle Dataset and point
`PROJECT_DIR` at it.
"""
    ),
    code(
        f"""
import os, sys
from pathlib import Path

REPO_URL = "{REPO_URL}"
PROJECT_DIR = Path("/kaggle/working/ai-trial-room")

# A Kaggle Dataset upload lands under /kaggle/input - prefer it if present.
for candidate in Path("/kaggle/input").glob("*/app.py") if Path("/kaggle/input").exists() else []:
    PROJECT_DIR = candidate.parent
    print(f"Using uploaded dataset at {{PROJECT_DIR}}")
    break
else:
    if not PROJECT_DIR.exists():
        !git clone -q $REPO_URL {{PROJECT_DIR}}
    print(f"Using cloned repo at {{PROJECT_DIR}}")

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print(sorted(p.name for p in PROJECT_DIR.iterdir() if not p.name.startswith(".")))
"""
    ),
    markdown(
        """
## 4. Configure

Every setting is an environment variable, read by `ai_trial_room/config.py`.
The defaults below are tuned for a 16 GB T4.
"""
    ),
    code(
        """
import os

# --- secrets ---------------------------------------------------------------- #
# Read HF_TOKEN from Kaggle Secrets or Colab Secrets. Never hardcode a token.
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle Secrets")
    except Exception:
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            print("HF_TOKEN loaded from Colab Secrets")
        except Exception:
            print("No HF_TOKEN found - continuing anonymously (fine for Apache-2.0 models)")

# --- resolution ------------------------------------------------------------- #
os.environ["AITR_WIDTH"]  = "768"
os.environ["AITR_HEIGHT"] = "1024"

# --- precision & memory (T4 = Turing, no usable bf16) ----------------------- #
os.environ["AITR_DTYPE"]             = "float16"
os.environ["AITR_QUANTIZE"]          = "auto"   # int4 if nunchaku present, else fp8/none
os.environ["AITR_MODEL_CPU_OFFLOAD"] = "1"
os.environ["AITR_ATTENTION_SLICING"] = "1"
os.environ["AITR_VAE_TILING"]        = "1"
os.environ["AITR_MAX_RESIDENT_BACKENDS"] = "1"  # never two heavy models at once

# --- sampling --------------------------------------------------------------- #
os.environ["AITR_STEPS"]    = "30"
os.environ["AITR_TRUE_CFG"] = "4.0"

# --- licensing -------------------------------------------------------------- #
# Leave unset for a commercially usable deployment. Set to "1" only to enable
# the CatVTON research baseline for non-commercial comparison.
# os.environ["ALLOW_NONCOMMERCIAL"] = "1"

# --- quality (Phase 2) ------------------------------------------------------ #
# fast | balanced | best. "best" adds a second pass over the pallu / dupatta.
os.environ["AITR_PRESET"] = "balanced"
os.environ["AITR_IDENTITY_STRENGTH"] = "0.92"  # face-blend opacity, 0.5-1.0
os.environ["AITR_VALIDATE"] = "1"             # flag questionable results

os.environ["AITR_SHARE"] = "1"   # public Gradio link
os.environ["AITR_LOG_LEVEL"] = "INFO"

# HF cache on the writable disk, so weights survive within the session.
os.environ["HF_HOME"] = "/kaggle/working/.hf" if Path("/kaggle").exists() else str(Path.cwd() / ".hf")
print("configured")
"""
    ),
    markdown(
        """
## 5. Sanity check - no GPU needed, no weights downloaded

109 tests covering mask geometry, drape prompts, Laplacian blending, output
validation, licence gating and batch orchestration.
"""
    ),
    code(
        """
!python tests/test_core.py
!python tests/test_wiring.py
!python tests/test_phase2.py
!python tests/test_batch_e2e.py
"""
    ),
    code(
        """
from ai_trial_room.utils.logging_setup import setup_logging
setup_logging("INFO")

from ai_trial_room.utils.device import detect_hardware, resolve_dtype, resolve_quantization
from ai_trial_room.router import describe_routing
from ai_trial_room.backends.registry import available_backends
from ai_trial_room.config import MODEL_SPECS

hw = detect_hardware()
print("Hardware :", hw.describe())
print("dtype    :", resolve_dtype("auto"))
print("quantize :", resolve_quantization("auto"))
print()
print("Permitted backends:")
for bid in available_backends():
    spec = MODEL_SPECS[bid]
    flag = "commercial OK" if spec.is_commercial else "RESEARCH ONLY"
    print(f"  {spec.repo_id:42s} {spec.license_name:16s} {flag}")
print()
for row in describe_routing():
    print(f"  {row['Category']:10s} -> {row['Backend']}")
"""
    ),
    markdown(
        """
## 6. Verify the preprocessing pipeline on one photo

Cheap (a few seconds, no diffusion weights) and it catches the most common
problems: photo too small, person not detected, not enough of the body in frame.
"""
    ),
    code(
        """
from pathlib import Path
from PIL import Image
from ai_trial_room.config import Category
from ai_trial_room.preprocessing.person import prepare_person
from ai_trial_room.preprocessing.garment import prepare_garment
from ai_trial_room.utils.errors import TrialRoomError

PERSON  = "assets/examples/person_01.jpg"   # <-- change to your file
GARMENT = "assets/examples/saree_01.jpg"    # <-- change to your file
CATEGORY = Category.SAREE

if not Path(PERSON).exists() or not Path(GARMENT).exists():
    print(f"Upload your images and set PERSON / GARMENT above.")
else:
    try:
        person = prepare_person(PERSON, CATEGORY)
        garment = prepare_garment(GARMENT, CATEGORY)
        print(f"framing          : {person.framing}")
        print(f"dominant shoulder: {person.pose.dominant_shoulder}")
        print(f"parsing available: {person.parse.available}")
        print(f"bg removed       : {garment.background_removed}")
        print(f"looks unstitched : {garment.looks_unstitched}")
        display(Image.fromarray(
            __import__("numpy").hstack([
                __import__("numpy").asarray(person.image.resize((256, 341))),
                __import__("numpy").asarray(person.inpaint_mask.convert("RGB").resize((256, 341))),
                __import__("numpy").asarray(garment.image.resize((256, 341))),
            ])
        ))
        print("left: person  |  middle: inpaint mask (white = repainted)  |  right: garment")
    except TrialRoomError as exc:
        print(f"Rejected: {exc.user_message}")
        print(f"  detail: {exc.detail}")
"""
    ),
    markdown(
        """
## 7. Launch the app

This downloads the model weights on first use (~20 GB, 10-20 min) and prints a
public `*.gradio.live` link. Keep the cell running while you demo.
"""
    ),
    code(
        """
!python app.py --share
"""
    ),
    markdown(
        """
## 8. Optional: build a comparison grid for LinkedIn

Same inputs, every permitted backend, identical seed - so any difference you
see is the model, not the noise.
"""
    ),
    code(
        """
!python scripts/compare_backends.py \\
    --person assets/examples/person_01.jpg \\
    --garment assets/examples/saree_01.jpg \\
    --category saree --steps 30 --seed 12345
"""
    ),
    markdown("Or sweep all four saree drapes through one model:"),
    code(
        """
!python scripts/compare_backends.py \\
    --person assets/examples/person_01.jpg \\
    --garment assets/examples/saree_01.jpg \\
    --category saree --sweep-drapes --steps 30 --seed 12345
"""
    ),
    markdown(
        """
## 9. Optional: batch a whole catalogue

One model photo, a folder of garments, every combination rendered with a
`manifest.csv` you can sort by "needs a human look". Check the job count with
`--dry-run` before committing a GPU session to it.
"""
    ),
    code(
        """
!python scripts/batch_catalogue.py \\
    --person assets/examples/person_01.jpg \\
    --garments assets/examples \\
    --category saree --dry-run
"""
    ),
    code(
        """
# Drop --dry-run to actually render. Add --all-drapes for a 4x lookbook,
# --preset best for the pallu refinement pass, --resume to continue a stopped run.
!python scripts/batch_catalogue.py \\
    --person assets/examples/person_01.jpg \\
    --garments assets/examples \\
    --category saree --preset balanced
"""
    ),
    code(
        """
import csv
from pathlib import Path

runs = sorted(Path("outputs").glob("catalogue-*/manifest.csv"), key=lambda p: p.stat().st_mtime)
if runs:
    with runs[-1].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"{runs[-1]}  ({len(rows)} rows)")
    flagged = [r for r in rows if r["status"] == "ok" and r["validation_ok"] != "True"]
    print(f"  rendered : {sum(1 for r in rows if r['status'] == 'ok')}")
    print(f"  flagged  : {len(flagged)}")
    print(f"  failed   : {sum(1 for r in rows if r['status'] in ('failed', 'error'))}")
    for row in flagged[:5]:
        print(f"    {row['output']}: {row['findings']}")
else:
    print("No batch run found yet.")
"""
    ),
    code(
        """
from pathlib import Path
from PIL import Image

grids = sorted(Path("outputs").glob("compare-*.png"), key=lambda p: p.stat().st_mtime)
if grids:
    print(grids[-1])
    display(Image.open(grids[-1]))
else:
    print("No grid found - run a comparison cell above first.")
"""
    ),
    markdown(
        """
## 10. Optional: train a saree drape LoRA

The base models drape a Nivi saree competently and a **Nauvari** one badly. A
LoRA trained on your own correctly-draped photos is the fix.

**Which base can a T4 train?**

| Base | Params | On a 16 GB T4 |
|---|---|---|
| `FLUX.2-klein-4B` | 4 B | ✅ default target |
| `Qwen-Image-Edit-2511` | 12 B | ⚠️ marginal - needs 4-bit + 512 px |

Dataset layout, photo guidance and the consent checklist are in
[`datasets/README.md`](../datasets/README.md). Short version: name files
`nivi_001.jpg`, `bengali_002.jpg`, … and aim for 150+ balanced images.

> Nothing here has been validated on a GPU. The loss and memory strategy are
> correct by construction, but no run has completed - treat the
> hyperparameters as starting points.
"""
    ),
    code(
        """
!pip install -q "peft==0.17.1" "bitsandbytes==0.45.0"
"""
    ),
    code(
        """
# Upload your images to datasets/saree_drapes/ first, then caption them.
from pathlib import Path

dataset = Path("datasets/saree_drapes")
dataset.mkdir(parents=True, exist_ok=True)

count = len([p for p in dataset.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
print(f"{count} image(s) in {dataset}")
if count == 0:
    print("Upload images named nivi_001.jpg, bengali_001.jpg, ... then re-run.")
"""
    ),
    code(
        """
!python scripts/caption_dataset.py --dataset datasets/saree_drapes
"""
    ),
    code(
        """
# Always dry-run first: it validates the dataset and prints the resolved plan
# (base model, resolution, quantization, batch, lr) without loading weights.
!python scripts/train_saree_lora.py --dataset datasets/saree_drapes --dry-run
"""
    ),
    code(
        """
# Drop --dry-run to train. Add --backend qwen_edit --quantize-base 4bit --width 512
# --height 768 to attempt the 12B model on a T4 instead.
!python scripts/train_saree_lora.py \\
    --dataset datasets/saree_drapes \\
    --name saree-drape-v1 \\
    --steps 1200
"""
    ),
    code(
        """
# Confirm the adapter is discoverable; it appears in the UI dropdown on restart.
from ai_trial_room.lora import discover_loras, lora_choices

for spec in discover_loras():
    print(spec.describe())
print()
print("UI dropdown:", lora_choices())
"""
    ),
    markdown(
        """
## 11. Optional: deploy to a Hugging Face Space

Push this app to your own Space with ZeroGPU. The dry run prints the exact upload
manifest and refuses to proceed if anything is missing or looks like a leak -
training photos, trained adapters and `.env` are never uploaded.
"""
    ),
    code(
        """
!python scripts/deploy_space.py --repo YOUR_NAME/ai-trial-room --dry-run
"""
    ),
    code(
        """
# Needs a WRITE token: https://huggingface.co/settings/tokens
# --yes skips the interactive confirmation, which a notebook cannot answer.
# !python scripts/deploy_space.py \
#     --repo YOUR_NAME/ai-trial-room \
#     --hardware zero-a10g \
#     --yes
"""
    ),
    markdown(
        """
## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Install Nunchaku (cell 2b), or set `AITR_SEQ_CPU_OFFLOAD=1`, or drop `AITR_HEIGHT` to 768 |
| `No person detected` | Use a brighter, front-facing photo where both shoulders are visible |
| `needs a photo showing the person from head to at least the knees` | Sarees and lehengas require a full-length photo - this guard is deliberate |
| Weights download very slowly | Set `HF_TOKEN`; anonymous downloads are rate-limited |
| `QwenImageEditPlusPipeline` missing | `pip install -U diffusers` - needs ≥ 0.36.0 |
| Kaggle can't reach Hugging Face | *Settings → Internet → On* |
| LoRA trained but "does nothing" | The trigger token must be in the prompt; the app inserts it, manual calls must too |
| LoRA output is noise | The adapter was trained for a different base model - check `aitr_lora.json` |
| OOM while training | Use `--backend flux_klein`, or `--quantize-base 4bit --width 512 --height 768` |
| `numpy.dtype size changed` | A NumPy-2 wheel crept in: `pip install "numpy==1.26.4"` and restart the kernel |
| `libGL.so.1: cannot open shared object file` | On a Space, `space/packages.txt` supplies `libgl1`; locally `apt install libgl1` |
| Space build fails on `sdk_version` | It must match the pinned `gradio` in `space/requirements.txt` |
| ZeroGPU call killed mid-generation | The declared duration was too low - lower the steps or use the Fast preset |

## Licensing reminder

The default configuration uses **only Apache-2.0 / MIT models**, including the
human parser (MediaPipe Selfie Multiclass), so output may be used commercially.

Setting `ALLOW_NONCOMMERCIAL=1` enables CatVTON (CC BY-NC-SA 4.0) and the
SegFormer parser (NVIDIA research licence) - **do not sell anything produced that
way.** A deployed Space forces this off at startup regardless.
"""
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def main() -> int:
    """Write the notebook and report its size."""
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")

    code_cells = sum(1 for cell in CELLS if cell["cell_type"] == "code")
    print(
        f"Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)} "
        f"({len(CELLS)} cells: {code_cells} code, {len(CELLS) - code_cells} markdown)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
