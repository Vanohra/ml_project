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
| Stage 3 | Done | Domain-conditioned presets with high-order degradation (`domain_degradation.py`) |
| Stage 3b | Done | Degradation-aware conditioning — ConditionedSRResNet with FiLM |
| Stage 4 | Done | RealSRSet inference + no-reference evaluation (NIQE, BRISQUE) |
| Stage 5 | Done | VIRAT surveillance video frame extraction + evaluation |

---

## Recommended Final Run

> **Use this config for the class project submission.**
> One command trains the strongest model that directly matches the proposal.

**Config:** `configs/final_cond_surveillance.yaml`  
**Experiment name:** `final_cond_surveillance`

```bash
python scripts/run_experiment.py \
    --config configs/final_cond_surveillance.yaml \
    --noref_dir data/RealSRSet
```

### Why this is the right config

| Choice | Reason |
|--------|--------|
| **ConditionedSRResNet** | The proposal's title is "Domain-Conditioned Degradation-Aware SR" — this is literally that model. FiLM layers initialise to identity (γ=1, β=0), so training starts as stable as plain SRResNet. |
| **Surveillance domain** | VIRAT is surveillance footage. Training on the domain you evaluate on is the whole point of domain conditioning. |
| **Curriculum (3 stages, 33 ep each)** | Heavy CCTV noise (σ up to 45) from epoch 1 makes training unstable. Curriculum lets the model learn basic SR on lighter degradation first, then graduate to the full range. Validated PSNR is always measured on the full preset, so numbers are comparable across all epochs. |
| **L1 pixel loss** | Sharper outputs than MSE. Standard for modern SR (Real-ESRGAN, SwinIR both use L1). Well-understood — easy to cite. |
| **Perceptual loss at weight 0.05** | VGG16 is frozen — it cannot destabilise training. 0.05 is conservative enough that if it ever causes issues you can set it to 0 and retrain in the same time. Adds visible texture detail that helps on surveillance footage where fine edges matter. |
| **No GAN** | GAN requires a discriminator, two-player training, and mode-collapse monitoring. Risk of wasted hours debugging a broken GAN outweighs any quality benefit at this stage. |

### If you are short on time

Edit two lines in `configs/final_cond_surveillance.yaml`:

```yaml
training:
  num_epochs: 50       # was 100

curriculum:
  stages:
    - name: easy
      start_epoch: 1   # unchanged
    - name: medium
      start_epoch: 17  # was 34  (1/3 of 50)
    - name: hard
      start_epoch: 34  # was 67  (2/3 of 50)
```

Results will be ~1–2 dB weaker but the architecture, domain, and curriculum are unchanged — the experiment is still valid and reportable.

---

## Folder Structure

```
ml_project/
│
├── configs/
│   └── default.yaml         ← ALL training settings live here (edit this, not train.py)
│
├── data/
│   ├── DIV2K/
│   │   ├── HR_train/        ← YOU place your HR training images here
│   │   ├── HR_valid/        ← YOU place your HR validation images here
│   │   ├── LR_train/        ← auto-created by prepare_data.py (Stage 1 only)
│   │   └── LR_valid/        ← auto-created by prepare_data.py (Stage 1 only)
│   ├── RealSRSet/           ← YOU place real-world LR images here (Stage 4)
│   └── VIRAT/
│       ├── videos/          ← YOU place VIRAT .mp4/.avi files here (Stage 5)
│       └── frames/          ← auto-created by extract_virat_frames.py
│           ├── quantitative/
│           │   ├── hr/      ← clean extracted frames (80 %)
│           │   └── lr/      ← surveillance-degraded counterparts
│           ├── preference/
│           │   ├── hr/      ← frames reserved for human preference study (20 %)
│           │   └── lr/
│           └── manifest.json← provenance record for every frame pair
│
├── scripts/
│   ├── prepare_data.py      ← Stage 1: creates LR images from HR images (pre-save to disk)
│   ├── degradation.py       ← Stage 2: generates LR images live during training
│   ├── domain_degradation.py← Stage 3: domain-conditioned presets (mobile/surveillance/dashcam)
│   ├── cond_utils.py        ← Stage 3b: builds degradation conditioning vectors (COND_DIM=13)
│   ├── curriculum.py        ← Curriculum training: stage scheduling and parameter sampling
│   ├── losses.py            ← Configurable loss functions: L1, MSE, VGG16 perceptual
│   ├── dataset.py           ← Loads image pairs for training and validation
│   ├── model.py             ← SimpleSRCNN model definition (kept for backward compat)
│   ├── models/
│   │   ├── __init__.py      ← Model registry: build_model(cfg) factory function
│   │   ├── srresnet.py      ← SRResNet model (~957K params, sub-pixel upsampling)
│   │   └── conditioned_sr.py← ConditionedSRResNet with FiLM conditioning (~1.03M params)
│   ├── train.py             ← Training loop — reads settings from configs/default.yaml
│   ├── run_experiment.py    ← Orchestrator: train + eval + append row to results.csv
│   ├── compare_experiments.py ← Reads all experiment JSONs and prints comparison table
│   ├── infer.py             ← Stage 4: folder-based inference on real-world LR images
│   ├── evaluate_paired.py   ← Paired evaluation: PSNR / SSIM / LPIPS (needs HR ground truth)
│   ├── evaluate_noref.py    ← No-ref evaluation: NIQE / BRISQUE (no HR needed)
│   ├── extract_virat_frames.py ← Stage 5: extract frames from VIRAT videos, apply surveillance degradation
│   ├── evaluate_virat.py    ← Stage 5: evaluate trained model on VIRAT HR/LR pairs
│   └── baseline.py          ← Bicubic-only reference (no neural network)
│
├── outputs/
│   ├── results.csv                  ← auto-appended by run_experiment.py (one row per run)
│   ├── comparison_table.csv         ← auto-written by compare_experiments.py
│   └── experiments/
│       └── baseline_srcnn/          ← auto-created per experiment
│           ├── checkpoints/         ← best.pth and last.pth
│           ├── metrics/             ← train_loss.json, val_psnr.json, eval_*.json
│           ├── samples/             ← visual comparisons every few epochs
│           │   └── epoch_005/
│           │       ├── img00_comparison.png   ← LR | Bicubic | SR | HR side-by-side
│           │       ├── img00_lr.png
│           │       ├── img00_bicubic.png
│           │       ├── img00_sr.png
│           │       └── img00_hr.png
│           ├── logs/                ← reserved for future log files
│           └── run_summary.json     ← full record of settings + results
├── bicubic_results/             ← auto-created by baseline.py
├── degradation_test/            ← auto-created by degradation.py self-test
└── preprocessing.log            ← auto-created by prepare_data.py
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

### `scripts/models/` — Model registry and SRResNet

#### `models/__init__.py`
Provides a single `build_model(cfg)` function. Reads `cfg["model"]["name"]`
from the config and returns the correct model instance. Supported values:
`"srcnn"` and `"srresnet"`.

#### `models/srresnet.py` — SRResNet
A deeper residual super-resolution network based on the generator of SRGAN
(Ledig et al., CVPR 2017).

```
Input:  raw LR image  (native resolution — NOT pre-upsampled)
  → Conv 9×9 + PReLU             64 feature maps
  → 8 × ResidualBlock            deep feature refinement at LR resolution
      (Conv 3×3 → BN → PReLU → Conv 3×3 → BN  +  skip)
  → Conv 3×3 + BN                post-residual merge
  → [global skip: add head]      stabilises training of the full network
  → 2 × UpsampleBlock            sub-pixel convolution (PixelShuffle ×2 each)
  → Conv 9×9                     final RGB reconstruction
Output: SR image  (4× upscaled)
```

Key differences from SimpleSRCNN:
- Runs convolutions at **LR resolution** (quarter of the pixels) — much more
  efficient, so many more layers fit in the same compute budget.
- Uses **PixelShuffle** (sub-pixel convolution) — a learned upsampler that
  outperforms fixed bicubic interpolation.
- ~16× more parameters → higher capacity to recover fine textures.

Total parameters: ~957,000.

---

### `scripts/train.py` — Training loop
Trains SimpleSRCNN end-to-end. For each epoch:
1. Feeds training batches through the model
2. Computes MSE loss vs HR ground truth
3. Updates weights via Adam optimiser + backpropagation
4. Evaluates on the full validation set → prints PSNR
5. Saves `last.pth` (always) and `best.pth` (only if PSNR improved)
6. Saves visual comparisons every few epochs to `samples/`
7. Writes `train_loss.json` and `val_psnr.json` after every epoch
8. Writes `run_summary.json` when training finishes

**All settings live in `configs/default.yaml` — do not edit train.py directly.**
Key settings to know:
```yaml
experiment:
  name: baseline_srcnn   # change this to name a new experiment
  seed: 42               # change to try a different random initialisation

training:
  num_epochs: 50
  batch_size: 16         # reduce to 8 or 4 if you get out-of-memory errors
  learning_rate: 0.0001

data:
  use_online_degradation: true   # true = Stage 2, false = Stage 1
```

---

### `scripts/domain_degradation.py` — Stage 3 domain presets
Extends the generic degradation pipeline with **domain-conditioned presets** and
**high-order (two-stage) degradation**. Three domains are supported:

| Domain | Blur | Noise | JPEG quality | Typical source |
|--------|------|-------|-------------|----------------|
| `mobile` | light (0.2–1.5σ) | low (0–15) | 60–95 | Phone photo → social app |
| `surveillance` | heavy (1.0–4.0σ) | heavy (10–45) | 20–60 | CCTV at distance, low-light |
| `dashcam` | moderate (0.5–2.5σ) | medium (5–30) | 30–75 | Vehicle camera, H.264 codec |

Each domain may optionally apply a **second degradation pass** at LR resolution
(no second downsample) to simulate re-encoding: DVR storage, social media
recompression, YouTube re-encoding, etc.

**Usage in training:**
```python
from domain_degradation import degrade_domain

# Returns (lr_image, metadata_dict)
lr_image, meta = degrade_domain(hr_image, scale=4, domain="surveillance")

# As a drop-in lambda for the DataLoader:
deg_fn = lambda img: degrade_domain(img, scale=4, domain="surveillance")[0]
```

**Tuning:** All parameter ranges live in the `DOMAIN_PRESETS` dict at the top of
the file — edit the numbers there to adjust a domain without touching any other code.

---

### `scripts/baseline.py` — Bicubic reference
Takes LR validation images, upscales them with bicubic interpolation (no neural network),
and reports PSNR. This is the score your trained model needs to beat.
Typical bicubic PSNR on DIV2K: **~28–30 dB**.

---

## 12-Hour Final Workflow

If you have one session to produce results, follow these commands in order.
Everything below runs from `ml_project/`.

### Prerequisites (10 minutes)

```bash
# 1. Install all dependencies
pip install torch torchvision Pillow numpy pyyaml scikit-image
pip install lpips piq                  # for SSIM / LPIPS / NIQE / BRISQUE
pip install opencv-python              # only needed for VIRAT extraction

# 2. Place your data
#    data/DIV2K/HR_train/   ← 100–800 HR training images
#    data/DIV2K/HR_valid/   ← 10–100 HR validation images
#    data/RealSRSet/        ← real-world LR images (no HR needed)
#    data/VIRAT/videos/     ← VIRAT .mp4/.avi files (optional)
```

### Step 1 — Run the strongest experiment (2–6 hours depending on GPU)

```bash
python scripts/run_experiment.py \
    --config configs/ablation_srresnet_l1.yaml \
    --noref_dir data/RealSRSet
```

This trains SRResNet for 100 epochs on the surveillance domain, runs no-ref
evaluation on RealSRSet, and appends a row to `outputs/results.csv`.
Training PSNR is logged to `outputs/experiments/ablation_srresnet_l1/metrics/val_psnr.json`.

### Step 2 — (Optional) Run baseline for comparison (30 minutes)

```bash
python scripts/run_experiment.py \
    --config configs/ablation_srcnn_l1.yaml \
    --noref_dir data/RealSRSet
```

### Step 3 — VIRAT evaluation (if videos available, 20 minutes)

```bash
# Extract frames + generate degraded LR pairs
python scripts/extract_virat_frames.py \
    --video_dir data/VIRAT/videos \
    --frames_per_video 20 \
    --seed 42

# Evaluate the trained model on VIRAT frames
python scripts/evaluate_virat.py \
    --experiment ablation_srresnet_l1
```

### Step 4 — Generate inference gallery (5 minutes)

```bash
python scripts/infer.py \
    --experiment ablation_srresnet_l1 \
    --img_dir data/RealSRSet \
    --save_gallery
```

Gallery saved to `outputs/experiments/ablation_srresnet_l1/inference/*/gallery.png`.

### Step 5 — Build comparison table (1 minute)

```bash
python scripts/compare_experiments.py
```

Results at `outputs/comparison_table.csv`.  Open in Excel or any CSV viewer.

### Key output files for the report

| File | Contents |
|------|----------|
| `outputs/comparison_table.csv` | All experiments side by side |
| `outputs/experiments/<name>/metrics/val_psnr.json` | PSNR per epoch (training) |
| `outputs/experiments/<name>/metrics/eval_noref_summary.json` | NIQE / BRISQUE means |
| `outputs/experiments/<name>/metrics/eval_virat_summary.json` | VIRAT PSNR / SSIM |
| `outputs/experiments/<name>/inference/*/gallery.png` | LR vs SR visual comparison |
| `outputs/experiments/<name>/samples/epoch_*/` | Visual samples during training |

> **Note on paired PSNR (SSIM/LPIPS):**
> - **ConditionedSRResNet (exp3):** `run_experiment.py` automatically runs
>   `evaluate_paired.py --regen_lr`, which re-degrades HR images on the fly and
>   passes real conditioning vectors to the model. No `LR_valid/` directory needed.
> - **SRResNet / SRCNN (exp1, exp2):** `evaluate_paired.py` needs pre-saved LR files
>   in `data/DIV2K/LR_valid/`. These are only created if you run
>   `python scripts/prepare_data.py` first (Stage 1 workflow). Without them, use
>   `best_psnr_db` from `run_summary.json` and NIQE/BRISQUE for perceptual quality.

---

## Installation

Open the VS Code terminal (`Ctrl + \``) and run:

```bash
cd "C:\Users\vanoh\OneDrive\Desktop\machine_learning_project\ml_project"
pip install -r requirements.txt
```

This installs: `Pillow`, `torch`, `torchvision`, `numpy`, `pyyaml`.

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

### Step 7 — Configure and Train

**Before training**, review `configs/default.yaml`. The key settings:

```yaml
experiment:
  name: baseline_srcnn   # all outputs go to outputs/experiments/baseline_srcnn/
  seed: 42               # fixed seed → same results every time

model:
  name: srcnn            # srcnn (fast) or srresnet (stronger)
  num_res_blocks: 8      # SRResNet only — more blocks = stronger but slower
  num_features: 64       # SRResNet only — standard width

training:
  num_epochs: 50
  batch_size: 16         # lower to 8 or 4 if you get out-of-memory errors
```

**To start training:**
```bash
python scripts/train.py
```

**To use a custom config:**
```bash
python scripts/train.py --config configs/my_experiment.yaml
```

**To quickly name a new run without editing the YAML:**
```bash
python scripts/train.py --experiment bigger_batch
```

**Expected output (first few epochs):**
```
[seed] Fixed to 42
[output] .../outputs/experiments/baseline_srcnn
[device] GPU: NVIDIA GeForce RTX ...    ← or "CPU" if no GPU

Loading datasets...
  Mode: online degradation (Stage 2)
  Training images  : 100
  Validation images: 40

============================================================
  Experiment : baseline_srcnn
  Epochs     : 50  |  Batch size : 16
  LR         : 0.0001  |  Seed      : 42
============================================================

── Epoch 1/50 ──
  Epoch   1/50 | Batch    0/50 | Loss 0.043210
  ...
  Train Loss : 0.018500
  Val PSNR   : 27.84 dB  (best: 27.84 dB)  ← new best!
  [checkpoint] best.pth updated (PSNR 27.84 dB)
  [samples] Saved 4 comparisons → .../samples/epoch_005
...
Training complete!
  Best PSNR  : 30.12 dB
  Summary    : .../run_summary.json
```

**If you want to resume training after stopping:**
Re-run the same command. It will ask:
```
Checkpoint found. Resume training? [y/N]:
```
Type `y` and press Enter.

**Outputs written during training:**

| File | Updated | Contents |
|------|---------|----------|
| `checkpoints/last.pth` | Every epoch | Model state for resuming |
| `checkpoints/best.pth` | When PSNR improves | Best model for evaluation |
| `metrics/train_loss.json` | Every epoch | `[{epoch, loss}, ...]` |
| `metrics/val_psnr.json` | Every epoch | `[{epoch, psnr}, ...]` |
| `samples/epoch_NNN/` | Every 5 epochs | LR / Bicubic / SR / HR comparisons |
| `run_summary.json` | End of run | All settings + final results |

---

## Checkpoints

Checkpoints are saved inside the experiment folder, never overwriting other runs:

```
outputs/experiments/baseline_srcnn/checkpoints/
  last.pth   ← saved every epoch — use this to resume
  best.pth   ← saved only when validation PSNR improves — use this for evaluation
```

To run a second experiment without losing the first, change `name` in the config:
```yaml
experiment:
  name: my_second_run   # outputs go to outputs/experiments/my_second_run/
```

---

## Troubleshooting

**Out-of-memory error during training:**
Open `configs/default.yaml` and reduce `batch_size`:
```yaml
training:
  batch_size: 8    # try 8, then 4 if still failing
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
In `configs/default.yaml`, change:
```yaml
data:
  use_online_degradation: false
```

---

## Domain Presets (Stage 3)

`scripts/domain_degradation.py` provides three degradation presets, each
matching the real-world image quality of a specific capture context.

### Why domain conditioning?

The generic pipeline in `degradation.py` samples blur/noise/JPEG from one wide
range. This trains a model that is "okay at everything" but not specialised.
Domain conditioning narrows the range to match a specific deployment target —
so a model trained on `surveillance` learns the heavy noise and blocking
artefacts that CCTV systems actually produce.

### What "high-order" means

Each domain's pipeline has two stages:

```
HR image
  ── Stage 1 (always) ──────────────────────────────────────────────────
  → Gaussian blur         (at HR resolution — lens, optics, distance)
  → Downsample by 4×      (the core LR step)
  → Gaussian noise        (at LR resolution — sensor noise)
  → JPEG / codec          (at LR resolution — capture compression)

  ── Stage 2 (domain-specific probability) ─────────────────────────────
  → Gaussian blur         (re-encode smoothing or de-blocking filter)
  → Gaussian noise        (transmission or re-encoding noise)
  → JPEG / codec          (DVR storage, social platform, streaming)
  ──────────────────────────────────────────────────────────────────────
  → LR image
```

Stage 2 has **no second downsample** — it operates at LR resolution to add
re-encoding artefacts without changing the scale factor.

### Domain parameter ranges

#### Mobile
Simulates: modern smartphone → shared via WhatsApp, Instagram, or iMessage.

| Parameter | Range | Reason |
|-----------|-------|--------|
| Blur sigma | 0.2 – 1.5 | Good phone optics; light blur only |
| Noise sigma | 0 – 15 | Decent BSI sensor in daylight |
| JPEG quality | 60 – 95 | App compression varies (WhatsApp ~75, Instagram ~80) |
| Resize methods | BICUBIC, BILINEAR, LANCZOS | Good ISP chips; no pixelated output |
| Stage 2 probability | 50% | Social media often re-compresses on upload |
| Stage 2 JPEG | 50 – 85 | Mild platform re-encode |

#### Surveillance
Simulates: fixed CCTV or IP camera, subject 5–20 m away, often at night.

| Parameter | Range | Reason |
|-----------|-------|--------|
| Blur sigma | 1.0 – 4.0 | Cheap wide-angle lens + subject distance |
| Noise sigma | 10 – 45 | Small sensor in low light (95% of frames) |
| JPEG quality | 20 – 60 | Aggressive bandwidth compression (200–500 kbps streams) |
| Resize methods | All four | Cheap encoding chips vary widely |
| Stage 2 probability | 75% | DVR re-encodes for storage; cloud viewer re-encodes again |
| Stage 2 JPEG | 20 – 50 | DVR storage is very aggressive |

#### Dashcam
Simulates: vehicle-mounted camera at speed, H.264/H.265 recorded, shared on YouTube.

| Parameter | Range | Reason |
|-----------|-------|--------|
| Blur sigma | 0.5 – 2.5 | Wide-angle + windshield + vibration |
| Noise sigma | 5 – 30 | Consumer sensor at fixed bitrate |
| JPEG quality | 30 – 75 | H.264 codec blocking at motion edges (JPEG approximates this) |
| Resize methods | All four | Consumer chips vary |
| Stage 2 probability | 60% | Upload to YouTube / insurance portal re-encodes |
| Stage 2 JPEG | 30 – 65 | Platform re-encode at lower bitrate |

### Verifying the presets

```bash
python scripts/domain_degradation.py
```

Saves one degraded LR image per domain to `outputs/domain_degradation_test/`
and prints the sampled parameters for each run. Open the images in VS Code
to visually compare what each domain looks like vs. the clean bicubic reference.

### Using a domain preset in training

In `train.py`, replace the generic `degrade` lambda with a domain-specific one:

```python
from domain_degradation import degrade_domain

# Example: train for surveillance
deg_fn = lambda img: degrade_domain(img, scale=4, domain="surveillance")[0]
```

Then pass `deg_fn` to `DIV2KDataset` as the `degradation_fn` argument.
The `[0]` discards the metadata dict and returns just the LR image,
which is what the DataLoader expects.

---

## Model Comparison: SimpleSRCNN vs SRResNet

Both models are selected from `configs/default.yaml` — no code changes needed.

| Property | SimpleSRCNN | SRResNet |
|----------|-------------|----------|
| Parameters | ~57K | ~957K |
| Depth | 3 conv layers | 8+ residual blocks (16+ layers) |
| Input to model | Bicubic-upsampled LR at HR size | Raw LR (native resolution) |
| Upsampling | Pre-upsampling (fixed bicubic) | Sub-pixel conv (PixelShuffle, learned) |
| Processes at | HR resolution (expensive) | LR resolution (efficient) |
| Training speed | Fast (~1× reference) | ~3–4× slower per epoch |
| Expected PSNR | ~30–31 dB | ~32–33 dB |
| GPU memory | Low | Moderate |

### When to use SimpleSRCNN

- Learning how the training pipeline works — tiny model, fast feedback loop
- Quick sanity checks after pipeline changes
- CPU-only training (low memory requirements)
- Verifying a new degradation preset before committing to a long SRResNet run

### When to use SRResNet

- Aiming for the highest PSNR your dataset and compute allow
- Writing up results for the CSC 4850 project report
- Comparing against a published baseline (original SRResNet is well-documented)
- You have a GPU and can afford 3–4× longer training time

### SRResNet config example

Copy `configs/default.yaml` to `configs/srresnet_run.yaml` and change:

```yaml
experiment:
  name: srresnet_baseline   # outputs go to outputs/experiments/srresnet_baseline/

model:
  name: srresnet
  num_res_blocks: 8    # start here; try 16 if you have time and GPU
  num_features: 64

training:
  num_epochs: 100      # SRResNet benefits from more epochs
  batch_size: 16       # reduce to 8 if out-of-memory
  learning_rate: 0.0001
```

```bash
python scripts/train.py --config configs/srresnet_run.yaml
```

### Why SRResNet is stronger

SimpleSRCNN must work at HR resolution — every convolution processes 4× more
pixels than necessary. It also has only 3 layers and no batch normalisation,
so there is a hard ceiling on what it can learn in 50 epochs.

SRResNet runs all its residual blocks at LR resolution (16× fewer pixels per
convolution), which means more layers fit within the same wall-clock time.
The PixelShuffle upsampler learns to fill in high-frequency detail during
upsampling itself, rather than relying on bicubic as a fixed starting point.
The combination of depth, skip connections, and learned upsampling consistently
yields 1.5–2.5 dB higher PSNR than SRCNN on the same dataset.

---

## Degradation-Aware Conditioning (Stage 3b)

`ConditionedSRResNet` extends SRResNet with **FiLM conditioning** — the model
receives both the LR image and a numeric description of how that image was
degraded, and can adapt its restoration strategy per image.

### Why conditioning helps

A standard SRResNet sees only the degraded pixels and must infer the degradation
severity from visual cues alone.  When trained on a domain like `surveillance`,
it will encounter images ranging from lightly blurred (clear day, close camera)
to severely noisy (night, distant subject), and will settle on one average
strategy for all of them.

By passing the actual degradation parameters, the model knows upfront whether
it is looking at a 45-sigma noise image or a 10-sigma one, and can adjust
its sharpening accordingly — stronger denoising when noise is high, more
aggressive edge reconstruction when blur is heavy.

### The conditioning vector (COND_DIM = 13)

Built by `cond_utils.build_cond_vector(metadata)` from the dict returned by
`domain_degradation.degrade_domain()`.  All values normalized to `[0, 1]`.

| Index | Name | Normalization | Description |
|-------|------|---------------|-------------|
| 0 | `blur_sigma_s1` | / 4.0 | Stage 1 Gaussian blur |
| 1 | `resize_method_s1` | / 3.0 | Resize algorithm (0=nearest, 1=bilinear, 2=bicubic, 3=lanczos) |
| 2 | `noise_sigma_s1` | / 45.0 | Stage 1 noise (0 = not applied) |
| 3 | `jpeg_quality_s1` | / 95.0 | Stage 1 JPEG quality (0 = not applied) |
| 4 | `jpeg_applied_s1` | {0, 1} | Explicit JPEG flag (avoids 0-quality ambiguity) |
| 5 | `stage2_active` | {0, 1} | 1 if second degradation stage ran |
| 6 | `blur_sigma_s2` | / 1.0 | Stage 2 blur (0 if inactive) |
| 7 | `noise_sigma_s2` | / 20.0 | Stage 2 noise (0 if inactive) |
| 8 | `jpeg_quality_s2` | / 85.0 | Stage 2 JPEG (0 if inactive) |
| 9 | `jpeg_applied_s2` | {0, 1} | Stage 2 JPEG flag |
| 10 | `domain_mobile` | {0, 1} | One-hot domain encoding |
| 11 | `domain_surveillance` | {0, 1} | |
| 12 | `domain_dashcam` | {0, 1} | |

### How FiLM conditioning works

FiLM (Perez et al., 2018) applies a per-channel affine transform to feature maps:

```
output_channel_c = γ_c(embedding) × feature_c  +  β_c(embedding)
```

`γ` (scale) and `β` (shift) are predicted by two small linear layers inside
each of the 8 conditioned residual blocks.  The embedding that feeds them comes
from a shared 2-layer MLP (`MetadataEncoder`) applied to the 13-d input vector.

**Initialization trick:** at the start of training, γ=1 and β=0 for any input,
so the model behaves exactly like a plain SRResNet.  The conditioning layers
learn to diverge from this identity only when there is a benefit, preventing the
conditioning signal from destabilising early training.

```
LR image  ──→  Head  ──→  FiLMResBlock × 8  ──→  PixelShuffle × 2  ──→  SR image
                               ↑
cond vec  ──→  MetadataEncoder ──→ embedding (64-d)  ─→  γ, β per block
```

### Data flow through the pipeline

```
HR image
  → domain_degradation.degrade_domain(hr, scale=4, domain="mobile")
  → (lr_image, metadata_dict)
  → cond_utils.build_cond_vector(metadata_dict)
  → cond_tensor  shape (13,)              ← this is what the model receives

Inside DataLoader (batch_size=B):
  lr_batch   (B, 3, 48, 48)
  hr_batch   (B, 3, 192, 192)
  cond_batch (B, 13)

model(lr_batch, cond_batch)  →  sr_batch  (B, 3, 192, 192)
```

### Training the conditioned model

```bash
python scripts/train.py --config configs/conditioned_srresnet.yaml
```

Change domain:
```bash
# For surveillance — edit configs/conditioned_srresnet.yaml:
#   data:
#     domain: surveillance
python scripts/train.py --config configs/conditioned_srresnet.yaml --experiment cond_surveillance
```

### Parameter count

| Component | Parameters |
|-----------|-----------|
| Backbone (same as SRResNet 8-block) | ~957K |
| MetadataEncoder MLP | ~5K |
| FiLM γ/β layers (8 blocks × 2 × 64×64) | ~66K |
| **Total** | **~1.03M** |

---

## Training Losses

The training loss is configured entirely from `configs/default.yaml` — no code changes needed.

### Available loss terms

| Term | Type | Package | When to use |
|------|------|---------|-------------|
| **Pixel** | L1 or MSE | built-in | Always — the main reconstruction signal |
| **Perceptual** | VGG16 features | `torchvision` (already installed) | When you want sharper textures and can tolerate slightly lower PSNR |
| ~~Adversarial~~ | ~~GAN~~ | — | Not implemented — too unstable for a class deadline |

### Pixel loss: L1 vs MSE

| | L1 (recommended) | MSE |
|-|------------------|-----|
| Formula | mean \|SR − HR\| | mean (SR − HR)² |
| Large errors | penalised linearly | penalised quadratically |
| Result | sharper output | smoother (blurrier) output |
| PSNR | slightly lower | slightly higher (PSNR is MSE-based) |
| Stability | stable | stable |

Use **L1** when you care about visual quality.  Use **MSE** only if the evaluation rubric uses PSNR as the sole metric and you want to optimise it directly.

### Perceptual loss (VGG16)

When enabled, the model is additionally trained to match intermediate features of a VGG16 network pretrained on ImageNet — not just pixel values.  This forces the SR output to look more natural to a deep network, which correlates with human preference.

**Network:** VGG16 (`torchvision.models.vgg16`, `VGG16_Weights.IMAGENET1K_V1`)  
**Layer:** `relu2_2` by default — the ReLU after the second Conv of VGG16 block 2 (128 feature maps at half spatial resolution).  Captures mid-level edges and textures.  
**Weights:** downloaded automatically from PyTorch Hub on the first run (~58 MB, cached after that).  
**VGG is frozen** — its weights are never updated during SR training.

Safe weight range: `0.05` – `0.1`.  Above `0.5` risks hallucinated textures.

### Config reference

```yaml
loss:
  pixel:
    type: l1        # "l1" (recommended) or "mse"
    weight: 1.0
  perceptual:
    enabled: false  # true to activate VGG16 feature loss
    weight: 0.1     # safe range: 0.05–0.1
    layer: relu2_2  # "relu2_2" (safer) or "relu3_3" (deeper, riskier)
```

### Loss logging

Every epoch the training loop prints and saves each active component:

```
  Train Loss : 0.015320  (pixel 0.012100 | perceptual 0.003220)
  Val PSNR   : 31.84 dB  (best: 31.84 dB)  <- new best!
```

`metrics/train_loss.json` now includes a column per active component:

```json
[
  {"epoch": 1,  "loss": 0.021, "pixel": 0.019, "perceptual": 0.021},
  {"epoch": 2,  "loss": 0.018, "pixel": 0.016, "perceptual": 0.018},
  ...
]
```

`run_summary.json` records the full loss configuration so every experiment is self-documenting.

### Recommended configs for this project

**Safest — pixel L1 only (default)**
```yaml
loss:
  pixel:
    type: l1
    weight: 1.0
  perceptual:
    enabled: false
```
Use this if you are close to a deadline or have not yet tuned other hyperparameters.  Stable, well-understood, and often sufficient to beat the bicubic baseline by 2–3 dB.

**Better visual quality — pixel + perceptual**
```yaml
loss:
  pixel:
    type: l1
    weight: 1.0
  perceptual:
    enabled: true
    weight: 0.1
    layer: relu2_2
```
Use this once your pixel-only run is working.  Expect slightly lower PSNR numbers but visibly sharper textures.  Train for more epochs (100+) with SRResNet or ConditionedSRResNet.

### Commands

```bash
# Default: L1 pixel loss only
python scripts/train.py --config configs/default.yaml

# Enable perceptual loss in a new experiment without editing default.yaml:
# 1. Copy the config
# cp configs/default.yaml configs/srresnet_perceptual.yaml
# 2. Edit: set perceptual.enabled: true
# 3. Run:
python scripts/train.py --config configs/srresnet_perceptual.yaml --experiment srresnet_perceptual
```

---

## Curriculum Training

Curriculum training progressively increases degradation difficulty during training
so the model learns on easy examples first and graduates to hard ones later.

### Why it helps

Starting on the full distribution (blur up to 4σ, noise up to 45, heavy JPEG)
from epoch 1 makes training noisy and unstable — the model is simultaneously
learning SR, denoising, deblurring, and JPEG artifact removal.

Starting easy lets the model develop a strong SR baseline on mild degradations
first.  Once that foundation exists, harder degradations improve generalization
rather than destabilising training from scratch.

### Requirements

Curriculum requires **Mode C** (domain degradation):
```yaml
data:
  use_online_degradation: true
  domain: mobile    # or surveillance / dashcam
curriculum:
  enabled: true
```

Validation always uses the **full domain preset** (no curriculum) so PSNR is
comparable across epochs regardless of what stage training is in.

### How stages are selected

Each stage has a `start_epoch`.  Before every training epoch, the training loop
picks the last stage whose `start_epoch ≤ current epoch`:

```
epoch  1–19  → easy    (blur 0.2–1.0,  noise 0–8,   JPEG 80–95)
epoch 20–39  → medium  (blur 0.3–2.5,  noise 5–25,  JPEG 50–85)
epoch 40+    → hard    (blur 0.5–4.0,  noise 10–45, JPEG 20–70)
```

The stage transition is automatic — no code changes needed, just edit `start_epoch`
in the YAML.

### Stage parameters

Each stage controls:

| Field | Description |
|-------|-------------|
| `name` | Label for logging (any string) |
| `start_epoch` | First epoch this stage is active |
| `blur_sigma` | `[min, max]` Gaussian blur range |
| `noise_sigma` | `[min, max]` noise range (0 min = some images get no noise) |
| `noise_probability` | Fraction of images that receive noise |
| `jpeg_quality` | `[min, max]` JPEG quality (higher = milder) |
| `jpeg_probability` | Fraction of images that get JPEG compression |
| `second_stage_probability` | Chance of a second degradation pass |

Optional second-stage fields (with defaults):
`second_blur_sigma [0.1, 0.5]`, `second_noise_sigma [0, 10]`,
`second_noise_probability 0.5`, `second_jpeg_quality [50, 85]`,
`second_jpeg_probability 0.75`

### Config example

```yaml
data:
  domain: mobile
  use_online_degradation: true

curriculum:
  enabled: true
  stages:
    - name: easy
      start_epoch: 1
      blur_sigma: [0.2, 1.0]
      noise_sigma: [0, 8]
      noise_probability: 0.50
      jpeg_quality: [80, 95]
      jpeg_probability: 0.70
      second_stage_probability: 0.0

    - name: medium
      start_epoch: 20
      blur_sigma: [0.3, 2.5]
      noise_sigma: [5, 25]
      noise_probability: 0.75
      jpeg_quality: [50, 85]
      jpeg_probability: 0.85
      second_stage_probability: 0.30

    - name: hard
      start_epoch: 40
      blur_sigma: [0.5, 4.0]
      noise_sigma: [10, 45]
      noise_probability: 0.90
      jpeg_quality: [20, 70]
      jpeg_probability: 0.95
      second_stage_probability: 0.70
```

### Training command

```bash
# Enable curriculum in default.yaml (set enabled: true  and data.domain: mobile), then:
python scripts/train.py --experiment curriculum_mobile

# Or use a dedicated config:
python scripts/train.py --config configs/conditioned_srresnet.yaml --experiment curriculum_test
# (also set curriculum.enabled: true in conditioned_srresnet.yaml)
```

### Verifying the curriculum is applied

**1. Console output** — every epoch prints the active stage and its parameter ranges:
```
── Epoch 1/100 ──
  [curriculum] Stage: easy  (blur [0.2, 1.0], noise [0, 8], jpeg [80, 95])
...
── Epoch 20/100 ──
  [curriculum] Stage: medium  (blur [0.3, 2.5], noise [5, 25], jpeg [50, 85])
```

**2. `metrics/curriculum_log.json`** — updated every epoch:
```json
[
  {"epoch": 1,  "stage": "easy",   "blur_sigma": [0.2, 1.0], ...},
  {"epoch": 20, "stage": "medium", "blur_sigma": [0.3, 2.5], ...},
  {"epoch": 40, "stage": "hard",   "blur_sigma": [0.5, 4.0], ...}
]
```
Read it mid-run: the file is rewritten after every epoch.

**3. `run_summary.json`** — includes `curriculum_enabled` and `curriculum_stages` keys
so the experiment record is self-documenting.

**4. Visual check** — compare `samples/epoch_001/` (easy stage) vs `samples/epoch_045/`
(hard stage).  LR panels should look noticeably noisier and blurrier in later epochs.

### Adjusting stage transitions

Change `start_epoch` values to match your `num_epochs`:

```yaml
# For a 50-epoch run:
- name: easy    start_epoch: 1
- name: medium  start_epoch: 17   # 1/3 of 50
- name: hard    start_epoch: 34   # 2/3 of 50

# For a 200-epoch run:
- name: easy    start_epoch: 1
- name: medium  start_epoch: 66
- name: hard    start_epoch: 133
```

---

## Evaluation

Two evaluation scripts are provided.  Run them from `ml_project/`.

### Which script to use

| Situation | Script |
|-----------|--------|
| You have matched LR + HR pairs (e.g. DIV2K validation) | `evaluate_paired.py` |
| You have real-world LR images with no ground truth | `evaluate_noref.py` |

### Installing evaluation dependencies

```bash
# Required for SSIM (evaluate_paired.py):
pip install scikit-image

# Optional — install for additional metrics:
pip install lpips    # LPIPS perceptual similarity (paired eval)
pip install piq      # NIQE + BRISQUE no-reference quality (no-ref eval)
```

Scripts run without optional packages — metrics that need a missing package are
skipped with a clear warning, so PSNR is always reported.

### Paired evaluation — `scripts/evaluate_paired.py`

Requires ground-truth HR images.  Computes PSNR, SSIM, and LPIPS for both the
bicubic baseline and the trained model in one pass.

| Metric | Needs GT | Package | Better = |
|--------|----------|---------|----------|
| PSNR (dB) | Yes | built-in | Higher |
| SSIM | Yes | `scikit-image` | Higher |
| LPIPS | Yes | `lpips` | Lower |

```bash
# Bicubic baseline only (no model needed):
python scripts/evaluate_paired.py \
    --experiment baseline_srcnn \
    --hr_dir data/DIV2K/HR_valid \
    --lr_dir data/DIV2K/LR_valid \
    --bicubic_only

# Trained model vs bicubic (config auto-discovered from experiment dir):
python scripts/evaluate_paired.py \
    --experiment baseline_srcnn \
    --hr_dir data/DIV2K/HR_valid \
    --lr_dir data/DIV2K/LR_valid

# Explicit config + last checkpoint:
python scripts/evaluate_paired.py \
    --experiment srresnet_run \
    --hr_dir data/DIV2K/HR_valid \
    --lr_dir data/DIV2K/LR_valid \
    --config configs/srresnet_run.yaml \
    --checkpoint last
```

`train.py` now saves a copy of its config to `outputs/experiments/<name>/config.yaml`
automatically so the eval script can auto-discover it.  For older runs, pass
`--config` explicitly.

**Outputs** saved to `outputs/experiments/<name>/metrics/`:

| File | Contents |
|------|----------|
| `eval_paired.csv` | Per-image: `filename, psnr_bicubic, ssim_bicubic, lpips_bicubic, psnr_model, ...` |
| `eval_paired_summary.json` | Mean ± std / min / max for each metric and method |

### No-reference evaluation — `scripts/evaluate_noref.py`

Does **not** require ground-truth images.  Runs the model on real-world LR images
(e.g. RealSRSet) and scores the SR output using blind quality metrics.

| Metric | Needs GT | Package | Better = |
|--------|----------|---------|----------|
| NIQE | No | `piq` | Lower |
| BRISQUE | No | `piq` | Lower |

NIQE measures distance from natural image statistics.  BRISQUE uses local natural
scene statistics via SVM.  Both capture perceptual quality without a reference.

```bash
# Bicubic SR quality on real images:
python scripts/evaluate_noref.py \
    --experiment baseline_srcnn \
    --img_dir data/RealSRSet \
    --bicubic_only

# Trained model vs bicubic:
python scripts/evaluate_noref.py \
    --experiment baseline_srcnn \
    --img_dir data/RealSRSet

# Also save the SR images for visual inspection:
python scripts/evaluate_noref.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --save_sr
```

**Outputs** saved to `outputs/experiments/<name>/metrics/`:

| File | Contents |
|------|----------|
| `eval_noref.csv` | Per-image: `filename, niqe_bicubic, brisque_bicubic, niqe_model, brisque_model` |
| `eval_noref_summary.json` | Mean ± std for each method |
| `eval_noref_sr/` | SR images (only with `--save_sr`) |

**Note:** NIQE requires images >= 64 × 64.  Smaller images are skipped for NIQE;
BRISQUE still runs on them.

### ConditionedSRResNet and evaluation — true vs fallback conditioning

The model receives a 13-element conditioning vector describing the degradation applied
to each LR image.  Evaluation quality depends on whether that vector is accurate.

| Evaluation path | Conditioning mode | How it works |
|-----------------|-------------------|--------------|
| `evaluate_paired.py --regen_lr` | **True** | Re-degrades each HR image using the training domain; captures exact params → real cond vector per image. Most faithful eval for the proposal. |
| `evaluate_virat.py` | **True** | Reads per-frame surveillance params from `manifest.json` (saved by `extract_virat_frames.py`) → real cond vector per frame. |
| `evaluate_paired.py` (no `--regen_lr`) | Fallback (zero) | Loads pre-saved LR files which have no metadata. Zero vector is passed — model still runs but conditioning head has no information. |
| `infer.py` on RealSRSet | Fallback (zero) | Real-world images have unknown degradation. Zero vector is the only option. **This is a known limitation** for unseen images; the model still performs SR via its backbone. |

**Recommended commands for conditioned model evaluation:**

```bash
# Paired DIV2K eval — true conditioning via re-degradation from HR:
python scripts/evaluate_paired.py \
    --experiment exp3_final \
    --hr_dir data/DIV2K/HR_valid \
    --regen_lr

# VIRAT eval — true conditioning from manifest (automatic):
python scripts/evaluate_virat.py --experiment exp3_final

# RealSRSet inference — fallback conditioning (documented limitation):
python scripts/infer.py \
    --experiment exp3_final \
    --img_dir data/RealSRSet \
    --save_gallery
```

`run_experiment.py` automatically selects `--regen_lr` when the config specifies
`model.name: conditioned_srresnet` and `data.domain` is set — no manual flag needed.

---

## Experiment Management

Two scripts — `run_experiment.py` and `compare_experiments.py` — let you run
ablations and collect results into a single comparison table with no manual
bookkeeping.

### `scripts/run_experiment.py` — full pipeline in one command

Runs training → paired evaluation → no-reference evaluation → appends one row
to `outputs/results.csv`.  Every step is optional via flags.

```
run_experiment.py --config <yaml>
                  [--experiment <name>]      override experiment name
                  [--checkpoint best|last]   checkpoint to evaluate (default: best)
                  [--noref_dir <path>]       real-world LR images for NIQE/BRISQUE
                  [--skip_train]             evaluate an already-trained model
                  [--skip_eval]             train only, do not write results row
```

**Example — full pipeline:**
```bash
python scripts/run_experiment.py \
    --config configs/ablation_srcnn_l1.yaml \
    --noref_dir data/RealSRSet
```

**Example — re-evaluate a finished run:**
```bash
python scripts/run_experiment.py \
    --config configs/ablation_srresnet_l1.yaml \
    --skip_train
```

After each run, `outputs/results.csv` gains one row:

| Column | Source |
|--------|--------|
| `experiment` | config `experiment.name` |
| `model` | config `model.name` |
| `domain` | config `data.domain` |
| `curriculum` | yes / no |
| `loss_pixel` | l1 / mse |
| `loss_perceptual` | yes / no |
| `best_psnr_train` | `run_summary.json` |
| `psnr_model`, `ssim_model`, `lpips_model` | `eval_paired_summary.json` |
| `niqe_model`, `brisque_model` | `eval_noref_summary.json` |

### `scripts/compare_experiments.py` — comparison table

Scans every directory under `outputs/experiments/`, reads its JSON files, and
prints a formatted table sorted by the chosen metric.  Also writes
`outputs/comparison_table.csv`.

```bash
# Compare all finished experiments (sorted by model PSNR):
python scripts/compare_experiments.py

# Sort by a different metric:
python scripts/compare_experiments.py --sort ssim_model
python scripts/compare_experiments.py --sort niqe_model   # lower is better

# Compare only specific experiments:
python scripts/compare_experiments.py \
    --experiments ablation_srcnn_l1 ablation_srresnet_l1 ablation_srresnet_perceptual
```

**Example output:**
```
============================================================
  Experiment Comparison  (3 run(s))
  Sorted by: psnr_model
============================================================

Experiment               Model                Domain        Curr.  Pixel  Perc.  Train  PSNR    PSNR   SSIM   LPIPS  NIQE   BRISQUE
                                                                                  PSNR   Bicubic Model  Model  Model  Model  Model
------------------------------------------------------------------------------------------------------------------------------
ablation_srresnet_perc.  SRResNet             surveillance  no     l1     yes    32.18  28.41   32.18  0.8821 0.1942 3.912  24.31
ablation_srresnet_l1     SRResNet             surveillance  no     l1     no     31.74  28.41   31.74  0.8703 0.2210 4.103  26.44
ablation_srcnn_l1        SimpleSRCNN          generic       no     l1     no     29.83  28.31   29.83  0.8121 0.2951 4.871  31.20

  PSNR/SSIM: higher is better   LPIPS/NIQE/BRISQUE: lower is better

Comparison CSV : outputs/comparison_table.csv
```

### The three experiments

| # | Config | Experiment name | Model | Domain | Curriculum | Perceptual | What it answers |
|---|--------|-----------------|-------|--------|------------|------------|-----------------|
| 1 | `exp1_baseline.yaml` | `exp1_baseline` | SimpleSRCNN | generic | no | no | Does neural SR beat bicubic at all? |
| 2 | `exp2_srresnet_domain.yaml` | `exp2_srresnet_domain` | SRResNet | surveillance | no | no | Does domain-specific training with a deeper backbone help? |
| 3 | `exp3_final.yaml` | `exp3_final` | ConditionedSRResNet | surveillance | yes | yes (0.05) | Does the full system (conditioning + curriculum + perceptual) outperform the simpler model? |

**Reading the table:**
- Exp 1 → Exp 2: two things change together (model depth, domain). Shows whether targeting the right degradation domain with a stronger backbone is worth it.
- Exp 2 → Exp 3: three things change (conditioning, curriculum, perceptual). Shows whether the full proposal system beats the plain domain-specific baseline. This is the core claim.

### Running the three experiments in order

Run these one at a time. Each one trains to completion before the next starts.

```bash
# Experiment 1 — SRCNN baseline (~30 min GPU)
python scripts/run_experiment.py \
    --config configs/exp1_baseline.yaml \
    --noref_dir data/RealSRSet

# Experiment 2 — SRResNet + surveillance domain (~2 hrs GPU)
python scripts/run_experiment.py \
    --config configs/exp2_srresnet_domain.yaml \
    --noref_dir data/RealSRSet

# Experiment 3 — Full final model (~3 hrs GPU, downloads VGG16 on first run)
python scripts/run_experiment.py \
    --config configs/exp3_final.yaml \
    --noref_dir data/RealSRSet

# Generate comparison table (run after all three finish)
python scripts/compare_experiments.py
```

The comparison table is written to `outputs/comparison_table.csv`.
Open in Excel or any CSV viewer to get the full side-by-side results.

---

## RealSRSet: Real-World Inference and No-Reference Evaluation

RealSRSet is a small set of real-world low-resolution images (no HR ground truth)
used to test how well a model generalises beyond the synthetic training distribution.

### Step 1 — Place RealSRSet images

Download RealSRSet (or any folder of real-world LR images) and place them at:

```
ml_project/
└── data/
    └── RealSRSet/
        ├── img001.png
        ├── img002.png
        └── ...
```

Any mix of `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.tiff` files is accepted.
There is no required naming convention.

### Step 2 — Run inference with `scripts/infer.py`

`infer.py` reads every image in a directory, runs the trained model, and saves
SR outputs to `outputs/experiments/<name>/inference/<tag>/`.

```bash
# Basic inference (uses best.pth, auto-discovers config):
python scripts/infer.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet

# Give the output run a descriptive tag so results don't get overwritten:
python scripts/infer.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --tag realSRSet_best

# Use last checkpoint instead:
python scripts/infer.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --checkpoint last \
    --tag realSRSet_last

# Also save a gallery contact sheet (LR | Bicubic | SR):
python scripts/infer.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --save_gallery

# Bicubic-only baseline (no model needed — useful for visual comparison):
python scripts/infer.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --bicubic_only \
    --tag bicubic_baseline \
    --save_gallery
```

**Outputs** saved to `outputs/experiments/<name>/inference/<tag>/`:

| File | Contents |
|------|----------|
| `<stem>_sr.png` | SR image for each input |
| `<stem>_bicubic.png` | Bicubic upsampled version (always saved for comparison) |
| `gallery.png` | Contact sheet: LR thumbnail \| Bicubic \| SR  (with `--save_gallery`) |
| `run_info.json` | Provenance: experiment, checkpoint, scale, source dir, timestamp |

**Expected output:**
```
Device       : cuda
  Loaded best.pth  (epoch 98, val PSNR 32.41 dB)
Input images : 20  in  data/RealSRSet
Output dir   : outputs/experiments/srresnet_run/inference/best_20260405_143021
Scale        : x4

  [  1/ 20] img001.png               LR 128x96 -> SR 512x384
  [  2/ 20] img002.png               LR 160x120 -> SR 640x480
  ...
  Gallery      : .../inference/best_.../gallery.png

Provenance   : .../inference/best_.../run_info.json
SR images    : .../inference/best_.../

To score these SR images with NIQE / BRISQUE:
  python scripts/evaluate_noref.py \
      --experiment srresnet_run \
      --img_dir data/RealSRSet \
      --checkpoint best
```

### Step 3 — Score with NIQE / BRISQUE (no-reference metrics)

After inference, score the original LR images as processed by the model using
`evaluate_noref.py`.  This runs the model again on the same images internally
and computes blind quality metrics on the SR outputs.

```bash
# Install piq first if you haven't:
pip install piq

# Score bicubic vs trained model:
python scripts/evaluate_noref.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet

# Bicubic-only (no model checkpoint needed):
python scripts/evaluate_noref.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --bicubic_only

# Also save the SR images produced during scoring:
python scripts/evaluate_noref.py \
    --experiment srresnet_run \
    --img_dir data/RealSRSet \
    --save_sr
```

Results are saved to `outputs/experiments/<name>/metrics/`:

| File | Contents |
|------|----------|
| `eval_noref.csv` | Per-image NIQE and BRISQUE for bicubic and model |
| `eval_noref_summary.json` | Mean ± std / min / max for each method |

### Gallery: qualitative inspection

Open `gallery.png` in VS Code or any image viewer.  Each row is one input image.
Columns (left to right): **LR thumbnail** | **Bicubic upsampled** | **Model SR**.

Good SR results look sharper than bicubic, recover fine texture (hair, text, edges),
and do not introduce obvious hallucinated patterns.

### Folder layout after a full RealSRSet run

```
outputs/experiments/srresnet_run/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── inference/
│   ├── realSRSet_best/
│   │   ├── img001_sr.png
│   │   ├── img001_bicubic.png
│   │   ├── ...
│   │   ├── gallery.png        ← LR | Bicubic | SR contact sheet
│   │   └── run_info.json
│   └── bicubic_baseline/
│       ├── img001_bicubic.png
│       ├── gallery.png
│       └── run_info.json
└── metrics/
    ├── eval_noref.csv
    └── eval_noref_summary.json
```

---

## VIRAT Surveillance Frame Evaluation (Stage 5)

This stage extracts real surveillance frames from VIRAT videos, degrades them
with the fixed surveillance preset, and evaluates trained models against the
clean originals — giving a ground-truth quality score on genuine surveillance footage.

### Folder layout

```
data/VIRAT/
├── videos/          ← YOU place .mp4 / .avi / .mov / .mkv files here
└── frames/          ← auto-created by extract_virat_frames.py
    ├── quantitative/
    │   ├── hr/      0000_hr.png, 0001_hr.png, ...   clean extracted frames  (80%)
    │   └── lr/      0000_lr.png, 0001_lr.png, ...   surveillance-degraded
    ├── preference/
    │   ├── hr/      frames reserved for human preference study  (20%)
    │   └── lr/
    └── manifest.json                                 full provenance for every pair
```

Place your VIRAT video files in `data/VIRAT/videos/`.  Any mix of `.mp4`, `.avi`,
`.mov`, or `.mkv` is accepted.  VIRAT Ground Dataset videos are available from
the [VIRAT project page](https://viratdata.org/).

### Step 1 — Install OpenCV

```bash
pip install opencv-python
```

### Step 2 — Extract frames

```bash
# Default: 20 frames per video, scale x4, seed 42, 80/20 split
python scripts/extract_virat_frames.py \
    --video_dir data/VIRAT/videos \
    --out_dir   data/VIRAT/frames

# More frames per video:
python scripts/extract_virat_frames.py \
    --video_dir data/VIRAT/videos \
    --frames_per_video 30

# Dry run — list videos and frame counts without extracting anything:
python scripts/extract_virat_frames.py \
    --video_dir data/VIRAT/videos \
    --dry_run
```

**What it does:**
1. Finds every video in `--video_dir` (sorted alphabetically for determinism).
2. Samples `--frames_per_video` evenly-spaced frame positions per video, with
   a small seed-controlled jitter so frames spread across the whole clip.
3. Extracts each frame as a clean PNG (the HR image).
4. Applies the **surveillance degradation preset** (heavy blur, noise, JPEG,
   optional second-stage re-encoding) to produce the matched LR image.
   Degradation parameters are seeded per frame — always reproducible.
5. Shuffles frames with the master seed and assigns 80% to `quantitative/`
   and 20% to `preference/`.
6. Writes `manifest.json` recording video name, frame index, timestamp,
   HR/LR paths, and exact degradation parameters for every pair.

**Expected output:**
```
Found 3 video(s) in data/VIRAT/videos

Pass 1 - planning frame indices...
  VIRAT_S_000001.mp4                       3600 frames  -> sample 20 positions
  VIRAT_S_000002.mp4                       5400 frames  -> sample 20 positions
  VIRAT_S_000003.mp4                       4200 frames  -> sample 20 positions

Split assignment (seed=42, fraction=0.8)
  quantitative : 48 frames
  preference   :  12 frames

Pass 2 - extracting and degrading 60 frames (scale x4)...
  [  10/ 60]  quantitative    VIRAT_S_000001.mp4  frame 183
  ...
=============================================
  Extraction complete
  Saved    : 60 frame pairs  (skipped 0)
  quantitative/ : 48 pairs  -> evaluate_virat.py
  preference/   :  12 pairs  -> human preference study
  Manifest : data/VIRAT/frames/manifest.json
```

### Step 3 — Evaluate a trained model

```bash
# Evaluate trained model vs bicubic on the quantitative split:
python scripts/evaluate_virat.py \
    --experiment srresnet_run

# Bicubic baseline only (no model checkpoint needed):
python scripts/evaluate_virat.py \
    --experiment srresnet_run \
    --bicubic_only

# Evaluate on the preference split (for visual comparison context):
python scripts/evaluate_virat.py \
    --experiment srresnet_run \
    --split preference

# Use the last checkpoint instead of best:
python scripts/evaluate_virat.py \
    --experiment srresnet_run \
    --checkpoint last

# Explicit config (for older runs without auto-saved config.yaml):
python scripts/evaluate_virat.py \
    --experiment srresnet_run \
    --config configs/srresnet_run.yaml
```

**Outputs** saved to `outputs/experiments/<name>/metrics/`:

| File | Contents |
|------|----------|
| `eval_virat.csv` | Per-frame PSNR / SSIM / LPIPS + video name, frame index, timestamp |
| `eval_virat_summary.json` | Mean ± std / min / max for each metric and method |

The CSV includes provenance columns (`video`, `frame_idx`, `timestamp_s`) so
you can trace any result back to its exact source clip and position.

**Expected summary output:**
```
==============================================================
  Experiment : srresnet_run  |  Split : quantitative  |  Frames : 48
  Method          PSNR (dB)      SSIM     LPIPS
  --------------------------------------------------------------
  bicubic         24.31 ± 1.42   0.6821    0.3214
  model           26.87 ± 1.19   0.7503    0.2541
  (PSNR/SSIM higher = better   LPIPS lower = better)
==============================================================
```

### How reproducibility is enforced

Every random decision in the pipeline is controlled by a single `--seed` integer:

| Decision | How seed is applied |
|----------|-------------------|
| Frame index jitter | `np.random.RandomState(seed + video_index)` per video |
| Degradation parameters | `random.seed(seed + global_frame_id)` before each frame's degradation call; reset to non-deterministic after |
| 80/20 split shuffling | `np.random.RandomState(seed * 31337)` — separate sub-seed so changing `--frames_per_video` does not reshuffle the split |

Running `extract_virat_frames.py` twice with the same `--seed` always produces:
- The same frame indices selected from each video
- Byte-identical LR images (same degradation parameters)
- The same assignment of frames to `quantitative/` vs `preference/`

The `manifest.json` records the seed, scale, and per-frame degradation parameters,
so results from any experiment can be traced and reproduced exactly.

### Human preference study (the `preference/` split)

The 20% preference split is kept **separate from quantitative evaluation** so
the images shown to human raters are never used to optimise or report PSNR.

Typical workflow:
1. Run inference with `scripts/infer.py --img_dir data/VIRAT/frames/preference/lr`
   for each model variant.
2. Show raters side-by-side panels: **Bicubic | Model A | Model B** (or similar).
3. Record preference counts.  The `manifest.json` provides the source video and
   timestamp so raters can be told what they are looking at.

The preference images are full-resolution HR/LR pairs — not thumbnails — so they
are suitable for both pixel-level inspection and print-quality comparison.

### Full output layout after a VIRAT run

```
outputs/experiments/srresnet_run/
└── metrics/
    ├── eval_virat.csv          per-frame results with provenance
    └── eval_virat_summary.json mean ± std table

data/VIRAT/frames/
├── quantitative/
│   ├── hr/  0000_hr.png ... 0047_hr.png
│   └── lr/  0000_lr.png ... 0047_lr.png
├── preference/
│   ├── hr/  0048_hr.png ... 0059_hr.png
│   └── lr/  0048_lr.png ... 0059_lr.png
└── manifest.json
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
