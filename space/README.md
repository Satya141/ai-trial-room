---
title: AI Trial Room
emoji: 👗
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Virtual try-on for Indian wear - sarees, lehengas, kurtis
tags:
  - virtual-try-on
  - fashion
  - image-editing
  - diffusion
  - indian-wear
  - saree
models:
  - Qwen/Qwen-Image-Edit-2511
  - black-forest-labs/FLUX.2-klein-4B
preload_from_hub:
  - Qwen/Qwen-Image-Edit-2511
suggested_hardware: zero-a10g
suggested_storage: small
---

# AI Trial Room

**Virtual try-on built for Indian wear** - dresses, kurtis, kurtas, sarees and
lehengas.

Upload a photo of yourself and a photo of a garment, pick a category, and get a
photorealistic image of you wearing it - with your face, body shape, pose and
background preserved.

## Your photos are not stored

Uploads are processed in memory and in a temporary directory that is deleted when
the request finishes. Results have all EXIF metadata stripped, so a downloaded
image cannot leak where the original photo was taken. Nothing is logged to disk
beyond timings.

The consent checkbox is **required** - the Generate button stays disabled until
you confirm the photo is yours or you have permission to use it.

## How to get a good result

| | |
|---|---|
| **Do** | Stand facing the camera, arms away from your body, even lighting |
| **Do** | Use a **full-length** photo for sarees and lehengas - head to at least the knees |
| **Do** | Use a plain flat-lay or mannequin shot of the garment |
| **Avoid** | Seated or turned poses, folded arms, heavy filters |
| **Avoid** | Half-body photos for draped garments - these are rejected on purpose |

Sarees are rejected without a full-length photo because the drape *is* the
garment. Generating one from a headshot means hallucinating a lower body that
will not match you.

## Saree drape styles

Pick the regional drape you want:

| Style | Region | Pallu |
|---|---|---|
| **Nivi** | Andhra Pradesh - the modern standard | Over the left shoulder, down the back |
| **Bengali** (Atpoure) | Bengal | Over both shoulders, one end hanging in front |
| **Gujarati** (Seedha pallu) | Gujarat, Rajasthan | Over the **right** shoulder, spread across the chest |
| **Nauvari** | Maharashtra | Nine-yard dhoti-style drape |

Lehengas get four dupatta options: one shoulder, both shoulders, over the head
(bridal), or across the forearms.

## Quality presets

| Preset | Steps | Refines pallu | Typical time |
|---|---|---|---|
| Fast | 20 | no | ~1 min |
| Balanced | 30 | no | ~2 min |
| Best | 40 | **yes** | ~4 min |

**Best** runs a second pass over the pallu/dupatta region for sharper border and
zari detail, then composites only that region back - so it can improve the fabric
without touching your face or the background.

## Models and licences

Everything here is **Apache-2.0**, so output may be used commercially.

| Component | Model | Licence |
|---|---|---|
| Try-on (primary) | [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Apache-2.0 |
| Try-on (fast tier) | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 |
| Pose & face | MediaPipe Pose / FaceMesh | Apache-2.0 |
| Human parsing | MediaPipe Selfie Multiclass | Apache-2.0 |
| Garment cut-out | rembg + u2net | MIT |

### Why not IDM-VTON or CatVTON?

Two independent reasons:

1. **Licensing.** Every garment-warping try-on model - IDM-VTON, CatVTON,
   OOTDiffusion - is CC BY-NC-SA. None can be used in a product anyone sells.
2. **Capability.** A saree is a single unstitched 5-9 metre rectangle whose shape
   exists *only* as a function of how it is wrapped. Warping models learn a
   flat-garment-to-body correspondence, so there is nothing for them to warp.

This Space uses reference-based image editing instead, where the drape is
described in language and synthesised.

## Known limitations

- **Nauvari** (dhoti-style) drapes are the least reliable - thin training
  representation for that silhouette.
- Fine zari and mirror work can soften. Use the **Best** preset.
- Heavy occlusion (folded arms, a held bag) confuses the garment mask.
- Print *colour and motif* transfer well; exact border geometry may shift.
- This visualises drape and colour. **It does not predict your size.**

## Source

Full source, architecture notes, the LoRA training pipeline and the catalogue
batch tool: see the project repository.

Built as a portfolio project. Model weights are governed by their own licences,
listed above.
