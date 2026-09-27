# LoRA training datasets

Put your training images here. This directory is gitignored - training photos of
people must never be committed by accident.

---

## Which layout do I need?

| Mode | You need | Teaches | Realistic? |
|---|---|---|---|
| **`drape`** | Single photos of well-draped garments + a caption each | What a Nauvari / Bengali / Gujarati drape *looks like* | Yes - this is your catalogue photography |
| `paired` | The same person photographed before and after, per garment | The garment-transfer behaviour itself | Rarely - needs controlled reshoots |

**Start with `drape`.** It is the data a shop already owns.

### Why a `drape` LoRA helps an *editing* model

Qwen-Image-Edit and FLUX.2 klein use one transformer for both generation and
editing - the edit path just adds reference-image conditioning. Teaching that
transformer the drape distribution therefore improves the edit path too, because
it is the same weights answering the same visual question.

What it will **not** do is improve how faithfully a *specific* garment's print is
transferred. Only `paired` data teaches that. Expect **better drapes, not better
print fidelity.**

---

## `drape` layout

```
datasets/saree_drapes/
    nivi_001.jpg
    nivi_001.txt          ← caption, written by scripts/caption_dataset.py
    nivi_002.jpg
    nivi_002.txt
    bengali_001.jpg
    bengali_001.txt
    gujarati_001.jpg
    gujarati_001.txt
    nauvari_001.jpg
    nauvari_001.txt
```

**Naming matters.** The prefix before the first `_` is read as the drape style, so
the captioner can name the drape without you tagging anything by hand. Valid
prefixes: `nivi`, `bengali`, `gujarati`, `nauvari`.

A file named anything else still trains, but gets a generic caption - and a LoRA
whose captions never say "Nauvari" cannot be triggered by asking for one. The
captioner reports these, or use `--default-style nivi` to assume one.

### Good training photos

- **One person, standing, front-facing**, arms away from the torso
- **Full length** - head to feet. The hem and pleat fall are what you are teaching
- **The drape must be correct.** This is the whole point. A mis-draped photo
  teaches the model the wrong thing, and one bad photo in forty is visible
- Varied models, colours, fabrics and backgrounds - variety is what stops the
  LoRA memorising individual photos
- At least 768 px on the short edge

### Avoid

- Seated, turned or heavily cropped shots
- Heavy filters or unusual colour grading
- Duplicates or near-duplicates of the same photo
- Anything where the drape is ambiguous or partly hidden

---

## `paired` layout

```
datasets/saree_paired/
    condition/  0001.jpg      ← the person, before
    target/     0001.jpg      ← the same person wearing the garment
    garment/    0001.jpg      ← the garment reference (optional)
    prompts/    0001.txt      ← the instruction (optional)
```

Files are matched on **stem**, so `condition/0001.jpg` pairs with
`target/0001.png` regardless of extension. A condition image with no matching
target is skipped and reported. Missing prompts get a generic instruction.

Same lighting, same pose, same camera between condition and target - otherwise
the LoRA learns "change the lighting" alongside "change the garment".

---

## How many images?

| Count | Expect |
|---|---|
| < 40 | Overfits - starts reproducing individual training photos |
| 40-150 | Workable |
| **150+** | Good generalisation |
| 300+ | Diminishing returns for a single drape concept |

Keep the **styles balanced**. The scanner warns when one style has more than 4×
another, because the LoRA will simply favour the over-represented one.

---

## Workflow

```bash
# 1. Caption (free, instant - uses the filename drape styles)
python scripts/caption_dataset.py --dataset datasets/saree_drapes
```

```bash
# 2. Richer per-image captions, if you have 100+ images and a GPU
python scripts/caption_dataset.py --dataset datasets/saree_drapes --strategy vlm --overwrite
```

```bash
# 3. Check the plan before committing a GPU session
python scripts/train_saree_lora.py --dataset datasets/saree_drapes --dry-run
```

```bash
# 4. Train
python scripts/train_saree_lora.py --dataset datasets/saree_drapes --name saree-drape-v1
```

The adapter lands in `loras/saree-drape-v1/` and appears in the Gradio LoRA
dropdown on next start. **Remember the trigger token** (`aitrsaree` by default) -
without it in the prompt, the adapter does nothing. The app inserts it
automatically via `apply_trigger`.

---

## Consent and copyright

Training a model on photographs of people is a consent question, not just a
licensing one.

Before you train on an image, confirm:

- You have **written permission** from the person in the photo to use it for
  model training, or it is stock/synthetic imagery licensed for that purpose.
- You hold rights to the garment photography. Shop catalogue images are usually
  someone else's copyright.
- If you are training for a client shop, the permission should be **theirs to
  give** - models photographed for a catalogue have usually not agreed to AI
  training, and that is a separate conversation worth having explicitly.

A LoRA trained on 150 photos of the same three models will reproduce
recognisable features of those people. Treat the adapter as carrying their
likeness, because it does.
