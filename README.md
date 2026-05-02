# Does Realistic Degradation Modeling Improve Super-Resolution? A Controlled Study

**CSC 4850 Machine Learning — Georgia State University, Spring 2025**  
Team: Dany George Nishanth, Vanohra Gaspard, Sabirin Mohamed  
IDs: 002750923, 002768369, 002598700  
Instructor: Dr. Dong Hye Ye  

---

## Research Question

Can matching a model's training degradation to the target deployment domain
measurably improve super-resolution quality — and does increasing model capacity
compound that gain?

Zhang et al. (BSRGAN, ICCV 2021) showed that realistic degradation modeling
dramatically improves blind SR for large pretrained models. We ask a different,
complementary question: **does the same principle hold for small models trained
from scratch, and how does degradation modeling compare to model capacity as a
lever for improvement?** We quantify both effects with controlled experiments on
DIV2K, using a surveillance-domain degradation preset.

Our contribution is not to replicate BSRGAN (we do not) but to isolate and
measure two factors BSRGAN does not separately study:
1. The PSNR cost of training-distribution mismatch (training on clean bicubic LR,
   deploying on realistically degraded inputs).
2. The additional gain from deeper architecture (SRResNet) on top of matched degradation.

---

## Connection to Target Paper

**Target paper:** Zhang et al., "Designing a Practical Degradation Model for Deep
Blind Image Super-Resolution," ICCV 2021 (BSRGAN).

BSRGAN's contribution: a randomized degradation pipeline (blur → resize → noise →
JPEG in random order) that makes a large pretrained RRDB model robust to real-world
inputs. They demonstrate this on no-reference real images (RealSRSet) where ground
truth is unavailable.

**What we add:** BSRGAN never directly measures the quantitative cost of
training-distribution mismatch in a controlled setting — they compare to bicubic
degradation baselines but do not train two identical models with clean vs. realistic
degradation and measure the PSNR gap. We do exactly that. We also study whether the
same effect appears for small models (~57k parameters) trained from scratch, not
just for large pretrained ones. This makes our work a controlled ablation that
BSRGAN itself does not perform.

We are inspired by but do not claim to reproduce BSRGAN. Our degradation pipeline
is single-order (not their randomized shuffle), our backbone is 15× smaller, and
we evaluate on matched DIV2K pairs with PSNR/SSIM rather than no-reference metrics.

---

## The Learning Problem

Image super-resolution (SR) is a supervised regression problem. Given a
low-resolution input image **y**, recover the high-resolution original **x**.

- **Input space:** LR images, produced by applying a degradation pipeline D to HR images
- **Output space:** HR images at 4× the spatial resolution of the input
- **Training signal:** pixel-level reconstruction loss between SR output and ground-truth HR

The problem is **ill-posed**: the same LR image can correspond to many plausible
HR images (information is irreversibly lost in downsampling). The model learns to
recover the most likely HR given the LR input, regularized by the loss function.

The mapping difficulty depends critically on what degradations are present. A model
trained only to invert bicubic downsampling has no capacity to remove blur, noise,
or JPEG artifacts — it has never seen them. This is the training-distribution mismatch
problem our experiments measure.

---

## The Degradation Model

The LR image **y** is produced from HR image **x** via: `y = D(x ; ψ)`
where ψ = (σ_blur, method, σ_noise, q_JPEG, scale) are degradation parameters
sampled fresh for every image during training.

**Pipeline (applied in order):**

```
x  (HR, full resolution)
↓  Gaussian blur with kernel σ_blur        [lens softness, applied at HR resolution]
↓  Downsample by 4× (bicubic/bilinear/lanczos, randomly chosen)
↓  Gaussian noise σ_noise  (90% of images)  [sensor noise at LR resolution]
↓  JPEG compression q_JPEG (90% of images)  [codec artifacts]
= y  (LR, 1/4 resolution)
```

**Why this order?** Optical blur precedes downsampling because cameras blur scenes
before the sensor samples them. Noise is introduced at sensor level (after downsampling).
JPEG is the final stage of a camera's image-processing pipeline.

**Surveillance domain preset:** Surveillance cameras have cheap optics, small sensors,
and aggressive stream compression. We use domain-matched parameter ranges:

| Parameter        | Generic (default) | Surveillance preset      |
|------------------|-------------------|--------------------------|
| Blur σ           | 0.2 – 3.0         | **0.8 – 2.5**            |
| Noise σ          | 0 – 25 (80%)      | **5 – 30 (90%)**         |
| JPEG quality     | 30 – 95 (80%)     | **30 – 70 (90%)**        |
| Resize methods   | all 4             | bicubic / bilinear / lanczos |

Training on this preset teaches the model what surveillance artifacts actually look
like, so it learns to undo them rather than averaging them away.

---

## Models

### Bicubic Baseline (0 parameters)

Upsamples LR 4× with bicubic interpolation. No training, no parameters.
Establishes the minimum floor any trained model must exceed. Typical PSNR on DIV2K:
~28–30 dB on clean inputs, lower on realistically degraded inputs.

### SimpleSRCNN (~57,000 parameters)

A 3-layer CNN operating at HR resolution (takes bicubic-pre-upscaled LR as input):

```
Input: bicubic-upscaled LR  (3, H, W)  — same spatial size as HR
→ Conv(3→64,  9×9, pad=4) → ReLU
→ Conv(64→32, 5×5, pad=2) → ReLU
→ Conv(32→3,  5×5, pad=2)

residual skip: output = conv_out + input
Output: refined SR image, clamped to [0, 1]
```

The residual connection makes training fast: the model learns small sharpening
corrections on top of bicubic, rather than learning the full mapping.

**Loss:** MSE — appropriate for small models, directly optimizes PSNR.

### SRResNet (~957,000 parameters)

A deep residual network operating at LR resolution internally, using learned
sub-pixel convolution for upsampling:

```
Input: raw LR  (3, H/4, W/4)  — NOT pre-upscaled
Head: Conv(3→64, 9×9) + PReLU
Body: 8 × ResidualBlock
      each: Conv(64→64,3×3) → BN → PReLU → Conv(64→64,3×3) → BN + skip
Post-body: Conv(64→64, 3×3) + BN
Global skip: add Head output to Post-body output
2 × UpsampleBlock:
    Conv(64→256, 3×3) → PixelShuffle(2) → PReLU   [×2 = 4× total upscaling]
Tail: Conv(64→3, 9×9), clamped to [0, 1]
```

**Why stronger:** The 8 residual blocks run at 1/16 the pixel count of HR, so deep
feature learning is cheap. PixelShuffle is a *learned* upsampler — the model decides
how to fill in the upsampled grid rather than using a fixed bicubic formula.

**Loss:** L1 — sharper than MSE for deep models; gives less weight to outlier pixels.

---

## Loss Functions

### MSE (SimpleSRCNN)

```
L_MSE = (1/N) Σ (SR_i − HR_i)²
```

Penalizes errors quadratically. Directly optimizes PSNR. Appropriate for small,
fast models but encourages spatial smoothing under high uncertainty.

### L1 (SRResNet)

```
L_L1 = (1/N) Σ |SR_i − HR_i|
```

Penalizes errors linearly. Produces sharper outputs than MSE for deep models at
the cost of slightly lower PSNR numbers but visually better textures.

**No GAN/adversarial loss.** GAN training requires a discriminator network,
two-player optimization scheduling, and careful monitoring for mode collapse.
The added instability is not justified for a class-project scope. L1 with a deep
backbone achieves strong quality without it.

---

## Optimization

| Setting          | SimpleSRCNN     | SRResNet        |
|------------------|-----------------|-----------------|
| Optimizer        | Adam            | Adam            |
| Learning rate    | 1e-4            | 1e-4            |
| LR schedule      | constant        | StepLR ×0.5 / 25 epochs |
| Loss             | MSE             | L1              |
| Batch size       | 16              | 8               |
| Epochs           | 50              | 100             |
| Input patch      | 48×48 LR → 192×192 HR | 48×48 LR → 192×192 HR |
| Input format     | bicubic pre-upscaled  | raw LR          |

**Patch-based training:** DIV2K images are up to 2040×1356 pixels. Loading full
images in batches requires gigabytes of GPU memory. We instead randomly crop 48×48
LR patches (192×192 HR) per sample. Benefits: (1) memory efficiency — each batch
is ~1 MB not ~500 MB; (2) augmentation — each epoch uses different random crops
from the same images, giving rich training variety from a modest dataset size.

---

## Experiments

| # | Question | LR Generation | Model | Epochs | Checkpoint |
|---|----------|---------------|-------|--------|------------|
| 1 | Does any CNN beat bicubic on clean LR? | Pure bicubic | SimpleSRCNN | 50 | checkpoints_srcnn_clean/ |
| 2 | Does surveillance-matched degradation improve SRCNN? | Surveillance preset | SimpleSRCNN | 50 | checkpoints_srcnn_surv/ |
| 3 | Does model capacity add further gain? | Surveillance preset | SRResNet | 100 | checkpoints_srresnet/ |

**Cross-condition evaluation (the key result):** After training, all models are
evaluated on the SAME surveillance-degraded validation set, regardless of training
distribution. This yields the primary result table showing: (a) how much PSNR a
model trained on clean LR loses when tested on realistic degradation, (b) how much
matched degradation training recovers, and (c) how much additional depth adds.

---

## Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 · log10(1 / MSE)    [dB]
```

Higher is better. Typical values: bicubic ~28–30 dB on clean LR, lower on
degraded. Each +1 dB is a meaningful visual improvement. Reported per model on
the full validation set.

### SSIM (Structural Similarity Index)

```
SSIM(x,y) = [2μ_x μ_y + C₁][2σ_xy + C₂] / [(μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)]
```

Computed with Gaussian window size=11, σ=1.5, k₁=0.01, k₂=0.03. Range [0,1],
higher is better. Captures luminance, contrast, and structural similarity jointly.
Correlates better with human perception than PSNR alone.

Both implemented in pure PyTorch (`scripts/metrics.py`) — no scikit-image or scipy.

---

## Folder Structure

```
ml_project/
├── configs/
│   └── default.yaml            ← YAML config for the YAML-driven train.py
│
├── data/
│   └── DIV2K/
│       ├── HR_train/           ← place HR training images here (800 max)
│       ├── HR_valid/           ← place HR validation images here (100 max)
│       ├── LR_train/           ← auto-created by prepare_data.py (Exp 1 only)
│       └── LR_valid/           ← auto-created by prepare_data.py (Exp 1 only)
│
├── scripts/
│   ├── model.py                ← SimpleSRCNN + SRResNet definitions
│   ├── metrics.py              ← PSNR and SSIM (pure PyTorch)
│   ├── degradation.py          ← degradation pipeline + domain presets
│   ├── dataset.py              ← DIV2KDataset: Modes A/B + upsample_lr flag
│   ├── train.py                ← SimpleSRCNN trainer (argparse, CSV logging)
│   ├── train_srresnet.py       ← SRResNet trainer (L1, StepLR, CSV logging)
│   ├── evaluate.py             ← cross-condition evaluation + visual grids
│   ├── run_experiments.py      ← experiment documentation + command printer
│   ├── baseline.py             ← bicubic-only evaluation
│   └── prepare_data.py         ← pre-save bicubic LR to disk (Exp 1)
│
├── outputs/
│   ├── checkpoints_srcnn_clean/    ← Exp 1 checkpoints
│   ├── checkpoints_srcnn_surv/     ← Exp 2 checkpoints
│   ├── checkpoints_srresnet/       ← Exp 3 checkpoints
│   ├── training_log_srcnn_clean.csv
│   ├── training_log_srcnn_surv.csv
│   ├── training_log_srresnet.csv
│   └── eval_results/
│       ├── eval_results.csv
│       ├── sample_00.png ... sample_04.png   ← side-by-side visual grids
│       └── mismatch_example.png              ← the "money figure"
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 0 — Install dependencies

```bash
cd "path/to/ml_project"
pip install -r requirements.txt
```

Installs: `torch`, `torchvision`, `Pillow`, `numpy`, `matplotlib`, `pyyaml`.

### Step 1 — Place HR images

Download the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and place images in:
- `data/DIV2K/HR_train/` — training images (100–800 recommended)
- `data/DIV2K/HR_valid/` — validation images (10–100 recommended)

Even 5–10 images work for a quick sanity check.

### Step 2 — Verify the pipeline

```bash
python scripts/model.py          # checks SimpleSRCNN (~57k) + SRResNet (~957k)
python scripts/metrics.py        # checks PSNR and SSIM
python scripts/degradation.py    # checks default + surveillance preset
python scripts/dataset.py        # checks all dataset modes
```

### Step 3 — Run Experiment 1 (SimpleSRCNN on clean LR)

```bash
python scripts/prepare_data.py   # generate pre-saved bicubic LR once

python scripts/train.py \
    --use_online_degradation False \
    --use_surv_preset False \
    --checkpoint_dir outputs/checkpoints_srcnn_clean \
    --log_file outputs/training_log_srcnn_clean.csv
```

### Step 4 — Run Experiment 2 (SimpleSRCNN on surveillance LR)

```bash
python scripts/train.py \
    --use_online_degradation True \
    --use_surv_preset True \
    --checkpoint_dir outputs/checkpoints_srcnn_surv \
    --log_file outputs/training_log_srcnn_surv.csv
```

### Step 5 — Run Experiment 3 (SRResNet on surveillance LR)

```bash
python scripts/train_srresnet.py
```

Progress is logged to `outputs/training_log_srresnet.csv`.
Checkpoints go to `outputs/checkpoints_srresnet/`.

### Step 6 — Cross-condition evaluation (the key result)

```bash
python scripts/evaluate.py \
    --srcnn_clean_ckpt  outputs/checkpoints_srcnn_clean/best.pth \
    --srcnn_surv_ckpt   outputs/checkpoints_srcnn_surv/best.pth \
    --srresnet_ckpt     outputs/checkpoints_srresnet/best.pth \
    --hr_dir            data/DIV2K/HR_valid \
    --n_samples         5 \
    --cross_condition
```

Prints the comparison table, saves visual grids to `outputs/eval_results/`,
writes `outputs/eval_results/eval_results.csv`, and saves `mismatch_example.png`.

### Step 7 — View experiment descriptions

```bash
python scripts/run_experiments.py          # print all experiment descriptions
python scripts/run_experiments.py --run 1  # print Experiment 1 commands
python scripts/run_experiments.py --run 2  # print Experiment 2 commands
python scripts/run_experiments.py --run 3  # print Experiment 3 + eval commands
```

---

## Expected Results

The cross-condition evaluation should produce a table similar to:

```
==============================================================
EVALUATION RESULTS — Surveillance-Degraded Inputs
==============================================================
Method                             PSNR (dB)    SSIM    Notes
--------------------------------------------------------------
Bicubic baseline                     26.xx      0.7xx   No ML, fixed filter
SRCNN (trained on clean LR)          25.xx      0.7xx   WORSE than bicubic — mismatch!
SRCNN (trained on surv. preset)      28.xx      0.8xx   +~2.5 dB over bicubic
SRResNet (trained on surv. preset)   30.xx      0.8xx   +~4.5 dB over bicubic
==============================================================
KEY FINDING: Training on matched degradation adds ~2.5 dB.
             Deeper model adds another ~1.9 dB on top.
==============================================================
```

(Exact numbers depend on training duration, dataset size, and hardware.)

---

## Future Work

Several extensions were descoped from the class version but remain natural continuations.
**Video temporal consistency**: real surveillance footage has inter-frame motion;
a natural extension would use motion-aligned feature aggregation to enforce consistency
across frames. **GAN-based perceptual training**: replacing L1 loss with an adversarial
loss (SRGAN-style) typically recovers finer textures at the cost of slight PSNR reduction
and substantially more complex training. **No-reference evaluation on RealSRSet**: PSNR
and SSIM require ground-truth HR images; real-world deployment cannot always provide
these, so blind metrics (NIQE, BRISQUE) on RealSRSet would test generalisation beyond
the DIV2K distribution. **Evaluation on VIRAT surveillance video frames**: VIRAT provides
real surveillance footage with known provenance, offering a more realistic test bed than
the synthetic DIV2K degradation pipeline.
