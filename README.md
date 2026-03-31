# Image Super-Resolution — Beginner Baseline Project

**Goal:** Train a neural network that takes a low-resolution (LR) image and recovers
a high-resolution (HR) version. This is the foundation for the full
Domain-Conditioned Degradation-Aware Blind Super-Resolution project (CSC 4850, GSU Spring 2025).

**Dataset:** DIV2K — 1,000 high-quality 2K resolution images used as the standard
benchmark in super-resolution research.

---

## What Has Been Built So Far

This project is built in stages. Each stage adds one new capability on top of the last.

| Stage | Status | What it adds |
|-------|--------|--------------|
| Stage 1 | Done | SimpleSRCNN model + bicubic LR + training loop |
| Stage 2 | Done | Realistic on-the-fly degradation (blur + noise + JPEG) |
| Stage 3 | Next | Domain-conditioned presets (mobile / surveillance / dashcam) |
| Stage 4 | Future | RealSRSet no-reference evaluation (NIQE, BRISQUE, LPIPS) |
| Stage 5 | Future | VIRAT surveillance video frame evaluation |

---

## Folder Structure

```
ml_project/
│
├── data/
│   └── DIV2K/
│       ├── HR_train/        ← YOU place your HR training images here
│       ├── HR_valid/          ← YOU place your HR validation images here
│       ├── LR_train/        ← auto-created by prepare_data.py (Stage 1 only)
│       └── LR_valid/          ← auto-created by prepare_data.py (Stage 1 only)
│
├── scripts/
│   ├── prepare_data.py      ← Stage 1: creates LR images from HR images (pre-save to disk)
│   ├── degradation.py       ← Stage 2: generates LR images live during training
│   ├── dataset.py           ← Loads image pairs for training and validation
│   ├── model.py             ← SimpleSRCNN model definition
│   ├── train.py             ← Training loop with checkpoints and PSNR validation
│   └── baseline.py          ← Bicubic-only reference (no neural network)
│
├── outputs/
│   ├── checkpoints/         ← auto-created by train.py (best.pth, last.pth)
│   ├── bicubic_results/     ← auto-created by baseline.py
│   ├── degradation_test/    ← auto-created by degradation.py self-test
│   └── preprocessing.log    ← auto-created by prepare_data.py
│
├── requirements.txt
└── README.md
```

---

## What Each File Does

### `scripts/prepare_data.py` — Stage 1 LR generator
Reads every HR image from `HR_train/` and `HR_valid/`, shrinks it 4× using bicubic
interpolation, and saves the result to `LR_train/` and `LR_valid/`. Optionally applies`
Gaussian blur and JPEG compression before saving. You only need this for Stage 1
(pre-saved LR files). In Stage 2, LR images are generated live and this step is skipped.

**Settings at the top of the file:**
```python
SCALE        = 4      # downsample factor
ADD_BLUR     = True   # apply Gaussian blur before downsampling
BLUR_RADIUS  = 1      # blur strength (0.5–2.0)
ADD_JPEG     = True   # apply JPEG compression artefacts
JPEG_QUALITY = 75     # 0 (worst) to 95 (best)
```

---

### `scripts/degradation.py` — Stage 2 live degradation
Generates a degraded LR image from an HR image on-the-fly during training.
Each image gets a different random combination of:
- **Gaussian blur** — simulates lens softness (sigma randomly sampled 0.2–3.0)
- **Bicubic/bilinear/nearest/lanczos downsample** — the core LR step
- **Gaussian noise** — simulates camera sensor noise (sigma 0–25, applied 80% of the time)
- **JPEG compression** — simulates codec artefacts (quality 30–95, applied 80% of the time)

Because these are randomised every call, the model sees a different LR version of the
same HR image each epoch — effectively giving it unlimited training data.

**Settings at the top of the file:**
```python
BLUR_SIGMA_MIN / MAX    = 0.2, 3.0
NOISE_SIGMA_MIN / MAX   = 0, 25
NOISE_PROBABILITY       = 0.8    # 0.0 to disable noise entirely
JPEG_QUALITY_MIN / MAX  = 30, 95
JPEG_PROBABILITY        = 0.8    # 0.0 to disable JPEG entirely
```

---

### `scripts/dataset.py` — Image loader
A reusable data loader that supports two modes:

**Mode A — pre-saved LR (Stage 1)**
```python
DIV2KDataset(hr_dir="...", lr_dir="...", patch_size=48, scale=4)
```
Loads matching HR and LR files from disk. Requires `prepare_data.py` to have run first.

**Mode B — online degradation (Stage 2)**
```python
DIV2KDataset(hr_dir="...", degradation_fn=degrade, patch_size=48, scale=4)
```
Loads HR only. Generates LR live by calling `degradation_fn` on each image crop.

`patch_size` controls training vs validation:
- `patch_size=48` → returns 48×48 LR patch + 192×192 HR patch (for training batches)
- `patch_size=None` → returns the full image (for validation)

---

### `scripts/model.py` — SimpleSRCNN
A 3-layer convolutional neural network for super-resolution.

```
Input:  bicubic-upscaled LR image  (same spatial size as HR)
  → Conv 9×9 → ReLU   (64 filters, extracts edges and textures)
  → Conv 5×5 → ReLU   (32 filters, maps to sharper features)
  → Conv 5×5           (3 filters, reconstructs RGB)
  + residual connection (adds the input back to the output)
Output: sharpened SR image
```

The **residual connection** means the model learns what small corrections to add
to the bicubic image, rather than learning the full output from scratch. This makes
training much faster and more stable on small datasets.

Total parameters: ~57,000 (tiny — ResNet-50 has 25 million).

---

### `scripts/train.py` — Training loop
Trains SimpleSRCNN end-to-end. For each epoch:
1. Feeds training batches through the model
2. Computes MSE loss vs HR ground truth
3. Updates weights via Adam optimiser + backpropagation
4. Evaluates on the full validation set → prints PSNR
5. Saves `last.pth` (always) and `best.pth` (only if PSNR improved)

**Settings at the top of the file:**
```python
NUM_EPOCHS             = 50
BATCH_SIZE             = 16   # reduce to 8 or 4 if you get out-of-memory errors
LEARNING_RATE          = 1e-4
PATCH_SIZE             = 48   # LR crop size (HR crop = 48 × 4 = 192)
SCALE                  = 4
USE_ONLINE_DEGRADATION = True  # True = Stage 2, False = Stage 1
```

---

### `scripts/baseline.py` — Bicubic reference
Takes LR validation images, upscales them with bicubic interpolation (no neural network),
and reports PSNR. This is the score your trained model needs to beat.
Typical bicubic PSNR on DIV2K: **~28–30 dB**.

---

## Installation

Open the VS Code terminal (`Ctrl + \``) and run:

```bash
cd "C:\Users\vanoh\OneDrive\Desktop\machine_learning_project\ml_project"
pip install -r requirements.txt
```

This installs: `Pillow`, `torch`, `torchvision`, `numpy`.

---

## Step-by-Step: How to Run Everything

### Step 1 — Get HR images

Download images from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
and place them in:
- `data/DIV2K/HR_train/` — training images (up to 800)
- `data/DIV2K/HR_valid/`   — validation images (up to 100)

Even 5–10 images is enough to verify everything runs correctly.

---

### Step 2 — (Stage 1 only) Pre-generate LR images

Skip this step if you are using Stage 2 online degradation (`USE_ONLINE_DEGRADATION = True`).

```bash
python scripts/prepare_data.py
```

**What it does:** Reads each HR image, applies optional blur and JPEG compression,
downscales by 4×, and saves to `LR_train/` and `LR_valid/`.

**Expected output:**
```
Scale factor : 4x
--- Processing HR_train -> LR_train ---
  Found 10 image(s) in HR_train
  [1/10] 0001.png  HR=2040x1356 -> LR=510x339  saved to LR_train/
  ...
DONE — processed: 10 | skipped: 0 | errors: 0
```

A full log is also saved to `outputs/preprocessing.log`.

---

### Step 3 — Run the bicubic baseline

```bash
python scripts/baseline.py
```

**What it does:** Takes each LR validation image and enlarges it with bicubic
interpolation (no model). Computes and prints PSNR for each image.

**Expected output:**
```
Running bicubic baseline on 10 image(s)...
  [1/10] 0001.png  LR 510x339 -> SR 2040x1356  PSNR = 29.41 dB
  ...
Average PSNR : 29.18 dB  (higher is better)
This is your baseline score to beat with a real model.
```

Upscaled images are saved to `outputs/bicubic_results/`.

Note: Baseline requires pre-saved LR files. Run `prepare_data.py` first if you
haven't already.

---

### Step 4 — Verify the degradation pipeline (Stage 2)

```bash
python scripts/degradation.py
```

**What it does:** Creates a fake HR image, runs it through the degradation
pipeline with random parameters, and checks the output size. If HR images exist
in `HR_valid/`, it also saves a visual sample to `outputs/degradation_test/`.

**Expected output:**
```
Sampled params:
  blur_sigma   : 1.842
  resize_method: 2
  noise_sigma  : 14.3
  jpeg_quality : 67

HR size : (480, 320)
LR size : (120, 80)  (expected 120x80)
Size check passed.
Randomness check passed.
```

---

### Step 5 — Verify the dataset loader

```bash
python scripts/dataset.py
```

**What it does:** Tests both Mode A (pre-saved LR) and Mode B (online degradation).
Checks that tensor shapes are correct and that online degradation produces different
results each call.

**Expected output:**
```
Mode A — pre-saved LR files:
  Full image  — LR: (3, 339, 510)  HR: (3, 1356, 2040)
  Patch mode  — LR: (3, 48, 48)    HR: (3, 192, 192)
  Shape check passed.

Mode B — online degradation:
  10 HR image(s) found.
  Patch mode  — LR: (3, 48, 48)    HR: (3, 192, 192)
  Shape check passed.
  Randomness check passed.
```

---

### Step 6 — Verify the model

```bash
python scripts/model.py
```

**Expected output:**
```
Model: SimpleSRCNN
Total parameters: 57,219
Input  shape: (2, 3, 192, 192)
Output shape: (2, 3, 192, 192)
Shape check passed.
model.py is working correctly.
```

---

### Step 7 — Train

```bash
python scripts/train.py
```

**What it does:** Trains SimpleSRCNN for `NUM_EPOCHS` epochs.
Every epoch prints training loss and validation PSNR. Checkpoints are saved
to `outputs/checkpoints/`.

**Expected output (first few epochs):**
```
SimpleSRCNN Training
============================================================
Using GPU: NVIDIA GeForce RTX ...    ← or "Using CPU" if no GPU
Mode: online degradation (Stage 2)
  Training images : 10
  Validation images: 10

  Epoch   0 | Batch    0/  5 | Loss 0.043210
  Epoch   0 | Batch   10/  5 | Loss 0.021043
  ...
Epoch   0 | Train Loss: 0.018500 | Val PSNR: 27.84 dB | Best: 27.84 dB  ← new best
  *** New best model saved (PSNR 27.84 dB) ***

Epoch   1 | Train Loss: 0.012300 | Val PSNR: 28.61 dB | Best: 28.61 dB  ← new best
...
```

**If you want to resume training after stopping:**
Re-run `python scripts/train.py`. It will ask:
```
Checkpoint found. Resume training? [y/N]:
```
Type `y` and press Enter.

---

## Checkpoints

Two files are saved to `outputs/checkpoints/`:

| File | When it's saved | Use it for |
|------|-----------------|------------|
| `last.pth` | After every epoch | Resuming interrupted training |
| `best.pth` | Only when PSNR improves | Final evaluation / deployment |

---

## Troubleshooting

**Out-of-memory error during training:**
Open `train.py` and reduce `BATCH_SIZE`:
```python
BATCH_SIZE = 8    # try 8, then 4 if still failing
```

**LR images not found (Mode A):**
Run `python scripts/prepare_data.py` first to generate them.

**Images not found at all:**
Make sure your HR images are in `data/DIV2K/HR_train/` and `data/DIV2K/HR_valid/`.
Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.

**Training PSNR lower than bicubic baseline:**
This is normal for the first few epochs, especially with Stage 2 degradation.
Keep training — it typically surpasses bicubic by epoch 10–20.

**Want to go back to Stage 1 (pre-saved LR):**
In `train.py`, change:
```python
USE_ONLINE_DEGRADATION = False
```

---

## Key Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **MSE loss** | Training error (lower = better) | Decreasing steadily each epoch |
| **PSNR (dB)** | Image quality score (higher = better) | Beat bicubic (~28–30 dB) |

---

## Project Context

This baseline is Stage 1–2 of a larger project:
**Domain-Conditioned Degradation-Aware Blind Super-Resolution**

Upcoming stages will add:
- **Stage 3:** domain presets for mobile cameras, surveillance, and dashcam footage
- **Stage 4:** no-reference evaluation on RealSRSet (NIQE, BRISQUE, LPIPS)
- **Stage 5:** evaluation on VIRAT surveillance video frames

Team: Dany George Nishanth, Vanohra Gaspard, Sabirin Mohamed
Course: CSC 4850 Machine Learning — Georgia State University, Spring 2026
Instructor: Dr. Dong Hye Ye
