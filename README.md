# Does Realistic Degradation Modeling Improve Image Super-Resolution? A Controlled Study

*CSC 4850 Machine Learning — Georgia State University, Spring 2025*
*Team: Dany George Nishanth, Vanohra Gaspard, Sabirin Mohamed*

---

## 1. Research Question

Can a super-resolution model trained on realistically degraded images outperform one trained on simple bicubic downsamples — and does increasing model capacity compound that gain? Surveillance cameras produce images with a specific combination of lens blur, sensor noise, and aggressive JPEG compression that plain bicubic downsampling cannot simulate. If a model is trained to undo only bicubic downsampling but deployed on footage that also has noise and JPEG artefacts, it will fail. We test whether (a) matching training degradation to the target domain and (b) using a deeper model architecture each independently improve the final PSNR and SSIM on surveillance-like inputs.

---

## 2. The Learning Problem

Image super-resolution (SR) is a supervised learning problem: given a low-resolution (LR) input image, recover the high-resolution (HR) original. The input space is the set of all possible LR images of a given scale, and the output space is the corresponding HR images.

The mapping is **ill-posed**: many LR images can be produced from the same HR image depending on the degradation applied, and many HR images could plausibly correspond to the same LR image. The model must therefore learn a probabilistic mapping from degraded LR inputs to plausible HR outputs, guided by a pixel-level loss function that measures reconstruction error on matched training pairs.

We formulate it as a regression problem trained on (LR, HR) pairs. The model minimises a pixel-level loss (L1 or MSE) that drives SR outputs toward the ground-truth HR images. The challenge is that pixel losses alone encourage spatially blurry outputs (the "average" of plausible solutions), which is why model depth and upsampling strategy both matter.

---

## 3. The Degradation Model

### Mathematical formulation

The LR image **y** is produced from the HR image **x** via the degradation operator **D**:

```
y = D(x ; ψ)
```

where **ψ** = (σ_blur, method, σ_noise, q_JPEG, s) are the degradation parameters.

### Pipeline (applied in order)

```
x  (HR)
 → Gaussian blur(σ_blur)          at HR resolution   [simulates lens & optical blur]
 → Downsample by scale s           →  y is now s× smaller
 → Gaussian noise(σ_noise)         at LR resolution   [simulates sensor noise]
 → JPEG compression(q_JPEG)        at LR resolution   [simulates codec artefacts]
= y  (LR)
```

**Why this order?** Blur happens before downsampling because real cameras blur the scene optically before the sensor samples it. Noise is added after downsampling because it is introduced at the sensor, which operates at LR resolution. JPEG is last because it is the final stage of a camera's image-processing pipeline.

### Surveillance domain preset

Generic degradation samples σ_blur from 0.2–3.0, σ_noise from 0–25, and JPEG quality from 30–95. Surveillance cameras are systematically worse: cheap lenses, subjects at distance, small sensors in low light, and aggressive stream compression. We use a domain-matched preset:

| Parameter | Generic range | Surveillance preset |
|-----------|--------------|---------------------|
| Blur σ | 0.2 – 3.0 | **0.8 – 2.5** (heavier, lens + distance) |
| Noise σ | 0 – 25 (80%) | **5 – 30** (**90% of frames**) |
| JPEG quality | 30 – 95 (80%) | **30 – 70** (**90% of frames**) |
| Resize methods | bicubic/bilinear/lanczos/nearest | bicubic/bilinear/lanczos |
| Scale | 4 | 4 |

Training on this preset teaches the model what surveillance artefacts actually look like, so it learns to undo them rather than averaging them away.

---

## 4. Models

### Bicubic baseline (non-ML)
Upsamples the LR image 4× using bicubic interpolation. No parameters, no training. This is the floor — any trained model should beat it. Typical PSNR on DIV2K: ~28–30 dB.

### SimpleSRCNN (~57,000 parameters)
A 3-layer CNN that takes a **bicubic-pre-upsampled** LR image (already at HR size) and learns to sharpen it. Architecture:

```
Input: bicubic-upscaled LR  (3, H, W)  [same spatial size as HR]
  → Conv(3→64, 9×9) → ReLU
  → Conv(64→32, 5×5) → ReLU
  → Conv(32→3, 5×5)
  + residual skip from input
Output: sharpened SR image (values clamped to [0, 1])
```

The **residual connection** (`output = model(x) + x`) makes training faster: the model only has to learn what small corrections to add to the bicubic image, which is much easier than learning the full mapping from scratch.

**Loss**: MSE (optimised for PSNR, matches the original SRCNN paper)

### SRResNet (~957,000 parameters)
A 16× deeper model that takes **raw LR at native resolution** and upsamples internally using learned sub-pixel convolution. Architecture:

```
Input: raw LR  (3, H, W)  [H and W are 4× smaller than HR]
  → Conv(3→64, 9×9) + PReLU                    [head]
  → 8 × ResidualBlock(64)                       [body: deep feature refinement at LR resolution]
       (Conv 3×3 → BN → PReLU → Conv 3×3 → BN + skip)
  → Conv(64→64, 3×3) + BN                       [post-residual merge]
  + global skip from head                        [stabilises gradient flow]
  → 2 × UpsampleBlock                           [PixelShuffle ×2 each → total ×4]
       (Conv(64→256, 3×3) → PixelShuffle(2) → PReLU)
  → Conv(64→3, 9×9)                             [tail: reconstruct RGB]
Output: SR image (values clamped to [0, 1])
```

**Why SRResNet is stronger:**
- All residual blocks run at **LR resolution** (16× fewer pixels per convolution) — so 8 deep layers cost less compute than 3 layers at HR resolution.
- **PixelShuffle** (sub-pixel convolution) is a learned upsampler — the model decides how to distribute its features into the expanded spatial grid, rather than relying on a fixed bicubic formula.
- No input-to-output residual skip: the model learns the full mapping, which is necessary because the input and output are at different spatial resolutions.

**Loss**: L1 — produces sharper results than MSE for deep models.

---

## 5. Loss Function

### SimpleSRCNN — MSE Loss

```
L_MSE = (1/N) Σ (SR_i - HR_i)²
```

MSE penalises large pixel errors quadratically, which encourages the model to produce smooth outputs that minimise the average error. This is well-matched to PSNR (which is defined as 10·log10(1/MSE)) and is appropriate for small, fast models.

### SRResNet — L1 Loss

```
L_L1 = (1/N) Σ |SR_i - HR_i|
```

L1 penalises errors linearly, giving less weight to occasional large mistakes. This encourages the model to commit to a sharper prediction rather than averaging over uncertainty. For deep models trained longer, L1 consistently produces sharper textures and edges than MSE, at the cost of slightly lower PSNR numbers.

**Why no GAN/adversarial loss?** GAN training requires a discriminator network, two-player optimisation scheduling, and careful monitoring for mode collapse. The instability risk is high for a class-project deadline. L1 loss with a deep backbone achieves strong perceptual quality without this complexity.

---

## 6. Optimization

| Setting | SimpleSRCNN | SRResNet |
|---------|------------|---------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| Batch size | 16 | 8 |
| Epochs | 50 | 100 |
| Patch size | 48×48 LR → 192×192 HR | 48×48 LR → 192×192 HR |

**Why patch-based training?**
Full DIV2K images are up to 2K resolution (2040×1356). Loading a batch of full images would require gigabytes of GPU memory. Instead, we randomly crop 48×48 patches from LR images (and their 192×192 HR counterparts) for each batch. Benefits:
- **Memory efficiency**: each batch is ~1 MB instead of ~500 MB.
- **Data augmentation**: each epoch, different random crops of the same image are used → the model sees many different patches from each image.
- A single 2040×1356 image yields thousands of unique 48×48 patches, so even a small dataset (100 HR images) gives rich training variety.

---

## 7. Experiments

| # | Research Question | LR Generation | Model(s) | Epochs |
|---|-----------------|---------------|----------|--------|
| 1 | Does any CNN beat bicubic on clean LR? | Pure bicubic downscale | Bicubic vs SimpleSRCNN | 50 |
| 2 | Does domain-matched degradation improve SRCNN? | Clean LR vs surveillance preset | SimpleSRCNN (clean) vs SimpleSRCNN (surveillance) | 50 each |
| 3 | Does model capacity add further gain? | Surveillance preset (both) | SimpleSRCNN (57k) vs SRResNet (957k) | 50 / 100 |

---

## 8. Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 · log10(MAX²_I / MSE)    [dB]
```

where MAX_I = 1.0 (float images in [0, 1]) and MSE is the mean squared pixel error. Higher is better. Typical values: bicubic ~28–30 dB; well-trained models 30–33 dB. PSNR is simple and universally reported, but it can underrate visually sharp images.

### SSIM (Structural Similarity Index)

```
SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
             ────────────────────────────────────
             (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)
```

where μ_x, μ_y are local means, σ_x², σ_y² are local variances, σ_xy is local covariance, and C1, C2 are small stability constants (k1=0.01, k2=0.03). Computed with a Gaussian window (size 11, σ=1.5). Range [0, 1], higher is better. SSIM captures luminance, contrast, and structural similarity jointly and correlates better with human perception than PSNR alone.

Both metrics are implemented in pure PyTorch in `scripts/metrics.py` — no scikit-image or scipy required.

---

## 9. Folder Structure

```
ml_project/
│
├── configs/
│   └── default.yaml            ← YAML config for the YAML-driven train.py
│
├── data/
│   └── DIV2K/
│       ├── HR_train/           ← YOU place HR training images here
│       ├── HR_valid/           ← YOU place HR validation images here
│       ├── LR_train/           ← auto-created by prepare_data.py (Experiment 1)
│       └── LR_valid/           ← auto-created by prepare_data.py (Experiment 1)
│
├── scripts/
│   ├── model.py                ← SimpleSRCNN (~57k) + SRResNet (~957k) definitions
│   ├── metrics.py              ← PSNR and SSIM (pure PyTorch, no new dependencies)
│   ├── degradation.py          ← Single-order degradation + domain presets
│   ├── domain_degradation.py   ← Two-stage high-order degradation (existing)
│   ├── dataset.py              ← DIV2KDataset: Modes A, B, C + upsample_lr flag
│   ├── train.py                ← SimpleSRCNN trainer (YAML-driven, USE_SURVEILLANCE_PRESET)
│   ├── train_srresnet.py       ← SRResNet trainer (self-contained, L1, 100 epochs)
│   ├── evaluate.py             ← 3-model comparison: table + visual grids + CSV
│   ├── run_experiments.py      ← Experiment launcher and documentation
│   ├── baseline.py             ← Bicubic-only evaluation script
│   ├── prepare_data.py         ← Pre-save bicubic LR to disk (Experiment 1)
│   └── models/                 ← Advanced model registry (existing, for YAML train.py)
│       ├── __init__.py
│       ├── srresnet.py
│       └── conditioned_sr.py
│
├── outputs/
│   ├── training_log_srcnn.csv      ← per-epoch: epoch, train_loss, val_psnr, val_ssim
│   ├── training_log_srresnet.csv   ← same for SRResNet
│   ├── eval_results.csv            ← final evaluation table
│   ├── eval_samples/               ← visual comparison grids (5 images)
│   ├── checkpoints/                ← SimpleSRCNN best.pth and last.pth
│   └── checkpoints_srresnet/       ← SRResNet best.pth and last.pth
│
├── requirements.txt
└── README.md
```

---

## 10. How to Run

### Step 0 — Install dependencies

```bash
cd "C:\Users\vanoh\OneDrive\Desktop\machine_learning_project\ml_project"
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

### Step 3 — Train SimpleSRCNN (Experiment 2 / 3)

```bash
# USE_SURVEILLANCE_PRESET = True is the default in train.py
python scripts/train.py --experiment exp_srcnn_surveillance
```

Progress is logged to `outputs/training_log_srcnn.csv`. Checkpoints go to `outputs/experiments/exp_srcnn_surveillance/checkpoints/`.

### Step 4 — Train SRResNet (Experiment 3)

```bash
python scripts/train_srresnet.py
```

Progress is logged to `outputs/training_log_srresnet.csv`. Checkpoints go to `outputs/checkpoints_srresnet/`.

### Step 5 — Evaluate all three models

```bash
python scripts/evaluate.py \
    --srcnn_ckpt    outputs/experiments/exp_srcnn_surveillance/checkpoints/best.pth \
    --srresnet_ckpt outputs/checkpoints_srresnet/best.pth \
    --hr_dir        data/DIV2K/HR_valid \
    --n_samples     5
```

Prints the comparison table, saves visual grids to `outputs/eval_samples/`, and writes `outputs/eval_results.csv`.

### Step 6 — View experiment descriptions

```bash
python scripts/run_experiments.py          # print all experiment descriptions
python scripts/run_experiments.py --run 3  # launch Experiment 3 (SRCNN + SRResNet)
```

---

## 11. Future Work

Several extensions were descoped from the class version but remain natural continuations of this work. **Video temporal consistency**: real surveillance footage has inter-frame motion; a natural extension would use motion-aligned feature aggregation (optical flow or deformable convolutions) to enforce consistency across frames. **GAN-based perceptual training**: replacing L1 loss with an adversarial loss (SRGAN-style) typically recovers finer textures at the cost of slight PSNR reduction and substantially more complex training. **No-reference evaluation on RealSRSet**: PSNR and SSIM require ground-truth HR images; real-world deployment cannot always provide these, so blind metrics (NIQE, BRISQUE) on RealSRSet would test generalisation beyond the DIV2K distribution. **Evaluation on VIRAT surveillance video frames**: VIRAT provides real surveillance footage with known provenance, offering a more realistic test bed than the synthetic DIV2K degradation pipeline. All of these remain open for a follow-on project.
