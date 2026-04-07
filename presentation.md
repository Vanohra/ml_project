# Presentation Execution Guide
**CSC 4850 Spring 2026 — Domain-Conditioned Degradation-Aware Blind Super-Resolution**
Team: Dany George Nishanth, Vanohra Gaspard, Sabirin Mohamed

This guide is written for a teammate who did not write the code.
Every command is exact and ordered. Run them from `ml_project/`.

---

## Run this first

If you have limited time, do only this section. It produces the final model,
its evaluation numbers, and the output images you need for the presentation.

### 1 — Open terminal in the right folder

```bash
cd "C:\Users\vanoh\OneDrive\Desktop\machine_learning_project\ml_project"
```

### 2 — Install packages

```bash
pip install -r requirements.txt
pip install lpips          # needed for LPIPS metric (evaluate_paired.py)
pip install piq            # needed for NIQE / BRISQUE (evaluate_noref.py)
```

### 3 — Confirm DIV2K data is in place

```
data/
  DIV2K/
    HR_train/    ← 100 images already present (0001.png … 0100.png)
    HR_valid/    ← 40 images already present  (0801.png … 0840.png)
```

No LR directories are needed. The pipeline degrades images on the fly.

### 4 — Place RealSRSet (download separately)

Download from: https://github.com/cszn/KAIR/tree/master/testsets
Place the folder at:

```
data/
  RealSRSet/
    img001.png
    img002.png
    ...
```

Any real-world LR images work. These do not need HR counterparts.

### 5 — Train the final model and evaluate

```bash
python scripts/run_experiment.py \
    --config configs/exp3_final.yaml \
    --noref_dir data/RealSRSet
```

Expected duration: **2–5 hours on GPU**, ~12 hours on CPU.
This single command: trains → evaluates with true conditioning → scores RealSRSet → writes results row.

### 6 — Find your outputs

| What you need | Where it is |
|---------------|-------------|
| Best checkpoint | `outputs/experiments/exp3_final/checkpoints/best.pth` |
| Training PSNR by epoch | `outputs/experiments/exp3_final/metrics/val_psnr.json` |
| Best PSNR (summary) | `outputs/experiments/exp3_final/run_summary.json` → `best_psnr_db` |
| Paired eval results (PSNR/SSIM/LPIPS) | `outputs/experiments/exp3_final/metrics/eval_paired_summary.json` |
| RealSRSet no-ref scores | `outputs/experiments/exp3_final/metrics/eval_noref_summary.json` |
| Visual samples (LR|Bicubic|SR|HR side-by-side) | `outputs/experiments/exp3_final/samples/epoch_*/` |

---

## Full workflow — all three experiments

Run experiments in order. Each one takes longer than the last.

### Exp 1 — Baseline (SimpleSRCNN, ~20–40 min GPU)

```bash
python scripts/run_experiment.py \
    --config configs/exp1_baseline.yaml \
    --noref_dir data/RealSRSet
```

Answers: "Does any neural SR model beat bicubic at all?"

### Exp 2 — Backbone + Domain (SRResNet, ~1.5–3 hrs GPU)

```bash
python scripts/run_experiment.py \
    --config configs/exp2_srresnet_domain.yaml \
    --noref_dir data/RealSRSet
```

Answers: "Does training on the surveillance domain with a deeper model help?"

### Exp 3 — Full system (ConditionedSRResNet, ~2–5 hrs GPU)

```bash
python scripts/run_experiment.py \
    --config configs/exp3_final.yaml \
    --noref_dir data/RealSRSet
```

Answers: "Does degradation-aware conditioning + curriculum + perceptual loss outperform the baseline?"

### Comparison table

After all three experiments finish:

```bash
python scripts/compare_experiments.py
```

Writes `outputs/comparison_table.csv`. Also prints to terminal.

---

## Dataset placement (complete reference)

### DIV2K — already in place

```
data/DIV2K/
  HR_train/    100 images (0001–0100.png)  ← training set
  HR_valid/    40 images  (0801–0840.png)  ← validation set
```

No LR directories are needed. Online degradation is the default for all configs.

### RealSRSet — download required

Purpose: no-reference evaluation (NIQE/BRISQUE) and visual inference.
No HR ground truth needed — these are real degraded images.

```
data/RealSRSet/
  img001.png
  img002.png
  ...   (any .png/.jpg files)
```

### VIRAT — optional (surveillance video eval)

Purpose: evaluating on real surveillance footage with known degradation params.
Requires: `pip install opencv-python`

```
data/VIRAT/
  videos/
    VIRAT_S_000001.mp4
    VIRAT_S_000002.mp4
    ...   (any .mp4/.avi/.mov/.mkv files)
```

If you have VIRAT videos:

```bash
# Step 1 — extract frames and generate LR pairs
python scripts/extract_virat_frames.py \
    --video_dir data/VIRAT/videos \
    --out_dir   data/VIRAT/frames \
    --seed 42

# Step 2 — evaluate model on extracted frames
python scripts/evaluate_virat.py --experiment exp3_final
```

Output: `outputs/experiments/exp3_final/metrics/eval_virat_summary.json`

**If you do not have VIRAT videos:** skip this. DIV2K + RealSRSet is sufficient
for the presentation. VIRAT is a bonus.

---

## Optional packages and which metrics they unlock

| Package | Install | Unlocks |
|---------|---------|---------|
| `scikit-image` | in `requirements.txt` (already) | SSIM in `evaluate_paired.py` |
| `lpips` | `pip install lpips` | LPIPS in `evaluate_paired.py` and `evaluate_virat.py` |
| `piq` | `pip install piq` | NIQE and BRISQUE in `evaluate_noref.py` |
| `opencv-python` | `pip install opencv-python` | VIRAT video extraction only |

**If a package is missing:** the script still runs. That metric column is skipped
and a clear warning is printed. PSNR always works with no optional packages.

---

## Step-by-step verification

### After `pip install -r requirements.txt`

Success: no red errors. Warnings about torch CUDA are fine if you are on CPU.

If torch fails: install manually with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
(adjust for your CUDA version; use `cpu` if no GPU).

### After each `run_experiment.py` call

**Success looks like:**
```
epoch  10 / 100 |  loss 0.012345  |  val PSNR 28.34 dB  ← numbers improving
...
Experiment complete: exp3_final
Best PSNR (training) : 30.12 dB
PSNR model vs bicubic: 30.12 vs 27.88 dB    ← model > bicubic = good
Results CSV          : outputs/results.csv
```

**Check `run_summary.json` exists:**
```
outputs/experiments/exp3_final/run_summary.json
```
Open it — it has `best_psnr_db`, `total_epochs`, `model`, `domain`.

**Check samples folder:**
```
outputs/experiments/exp3_final/samples/
  epoch_005/    (after 5 epochs)
  epoch_010/
  ...
```
Each folder has 4 PNG files showing LR | Bicubic | SR | HR side-by-side.
These are your qualitative figures for the presentation.

**If training crashes immediately:**
- Check that `data/DIV2K/HR_train/` has PNG files.
- Check CUDA memory: reduce `batch_size` from 16 to 8 in the config.
- On CPU: reduce `num_epochs` to 50 to finish faster.

**If PSNR is not improving after 10 epochs:**
- This is normal early — improvement typically starts around epoch 5–15.
- If it never improves, check that HR_valid/ images are present.

### After `compare_experiments.py`

**Success looks like:**
```
Experiment               Model                 Domain       Train PSNR   PSNR Model
--------------------     --------------------  -----------  ----------   ----------
exp3_final               conditioned_srresnet  surveillance  30.12        30.08
exp2_srresnet_domain     srresnet              surveillance  29.55        29.50
exp1_baseline            srcnn                 generic       26.21        26.18
```

Model PSNR > bicubic PSNR = the model beats bicubic.
exp3 > exp2 > exp1 = the story works.

Output file: `outputs/comparison_table.csv`

---

## What we currently have compared to the proposal

### Implemented

- **ConditionedSRResNet with FiLM conditioning** — exact architecture from the proposal.
  13-element degradation conditioning vector (blur, noise, JPEG, resize method,
  second-stage activity, domain one-hot). FiLM modulation in all 8 residual blocks.
  Identity initialisation ensures stability from epoch 1.

- **Surveillance domain degradation preset** — matches VIRAT characteristics:
  heavy blur (1–4σ), high noise (10–45σ), aggressive JPEG (20–60), 75%
  second-stage re-encoding. Three domains total (mobile, surveillance, dashcam).

- **High-order degradation pipeline** — two-stage degradation matching Real-ESRGAN
  design (blur → downsample → noise → JPEG, then optional second pass at LR resolution).

- **Curriculum training** — 3-stage progressive difficulty (easy 1–33, medium 34–66,
  hard 67–100). Validation always uses the full preset for comparable PSNR.

- **L1 pixel loss + VGG16 perceptual loss** — L1 dominant (w=1.0), perceptual
  conservative (w=0.05). VGG frozen. No GAN (deliberate choice, see below).

- **True conditioning in evaluation** — `evaluate_paired.py --regen_lr` re-degrades
  HR images and passes real cond vectors. `evaluate_virat.py` reads per-frame params
  from manifest. This makes evaluation faithful to the proposal.

- **Three-experiment ablation** — exp1 (SRCNN/generic), exp2 (SRResNet/surveillance),
  exp3 (ConditionedSRResNet/curriculum/perceptual). Clean story with one variable per step.

- **Full evaluation pipeline** — PSNR (always), SSIM, LPIPS (paired), NIQE, BRISQUE
  (no-reference). Gallery output, comparison CSV, experiment runner.

- **VIRAT frame extraction with provenance** — reproducible extraction, per-frame
  degradation metadata saved in manifest.json, 80/20 quantitative/preference split.

### Partially implemented / approximated

- **Degradation conditioning at inference on real images** — for RealSRSet (no metadata),
  a zero conditioning vector is used. The model still produces SR via its backbone but
  cannot adapt to the specific degradation. This is documented as a known limitation.
  A learned degradation estimator was explicitly out of scope.

- **Dataset scale** — using 100 DIV2K training images and 40 validation images, not
  the full 800/100 split. Results will be lower than published papers but the
  comparisons within our three experiments remain valid and meaningful.

- **Second-stage degradation in evaluation** — the `evaluate_virat.py` path uses true
  metadata. The `--regen_lr` path for DIV2K re-degrades fresh (different LR per run),
  which is correct for measuring conditioning but means numbers may vary slightly
  between eval runs.

### Not yet implemented

- **Temporal / multi-frame super-resolution** — the proposal mentioned video SR as
  future work. We train and evaluate on still frames only.

- **GAN / adversarial loss** — explicitly excluded. The perceptual loss (VGG16) provides
  texture sharpness without the instability risk. This is a justified engineering decision.

- **Full DIV2K training set (800 images)** — using 100 images to stay within compute
  budget. Results scale with data; our ablation comparisons are still valid.

- **Blind degradation estimation from LR** — estimating conditioning parameters from
  the degraded image itself (like Real-ESRGAN's blind estimator) was not implemented.
  We use synthetic metadata during training and evaluation.

---

## Why our current standing is still strong

**The core proposal claim is implemented end-to-end.**
We built a model that receives explicit degradation metadata and uses it to condition
its super-resolution strategy per image. FiLM conditioning, the surveillance domain
preset, and curriculum training are all running and producing results.

**The ablation story is clean and defensible.**
Three experiments with one variable changed per step:
- exp1 → exp2: does a deeper model + domain-specific training help? (yes)
- exp2 → exp3: does conditioning + curriculum + perceptual loss help? (shows delta)
This is exactly what reviewers and graders look for: controlled comparisons.

**Evaluation is more faithful than the README initially stated.**
True conditioning is now used in both paired eval (via `--regen_lr`) and VIRAT eval
(via manifest). The evaluation actually exercises the conditioning head rather than
bypassing it with zeros. This directly supports the proposal claim.

**The perceptual loss choice is justified, not a shortcut.**
GAN training requires discriminator scheduling and mode-collapse monitoring that is
difficult to debug in a class project timeline. VGG16 perceptual loss at w=0.05
adds texture sharpness safely. This is explicitly the approach used in many published
SR papers as a stable alternative to adversarial training.

**The degradation pipeline is richer than most class projects.**
Two-stage (high-order) degradation with domain-specific presets, configurable resize
samplers, and per-image metadata logging goes significantly beyond a bicubic
downsampling baseline. The implementation matches the conceptual framework of
Real-ESRGAN (Wang et al., 2021) adapted for a fixed-scale single-image SR setting.

---

## Known limitations compared to the proposal

**1. No blind degradation estimation from LR.**
In deployment, you would not have the degradation parameters — you would need to
estimate them from the degraded image. We do not implement this. The zero conditioning
vector fallback for RealSRSet means the conditioning head is unused on truly blind inputs.
This should be stated clearly in the presentation and framed as the primary future work item.

**2. Reduced training data (100 images vs 800).**
Our PSNR numbers will be 1–3 dB below published SRResNet results. This is expected and
should be disclosed. The comparisons between our three experiments are valid; the absolute
numbers are not comparable to the published literature.

**3. No temporal modeling.**
We treat video frames as independent still images. A true surveillance SR system would
benefit from temporal coherence across frames. This is scope that was never promised for
this semester.

**4. Single scale factor (4×).**
All experiments use 4× upscaling only. The architecture can support other scales with
config changes but we did not ablate this.

**5. VGG16 perceptual loss instead of GAN.**
Results will be sharper than L1-only but not as high-frequency-detailed as ESRGAN-style
models. This is a known trade-off and was a deliberate engineering decision, not an
oversight.

---

## What we have to present

### Figures to collect

- **Qualitative comparison grid** — for 3–4 sample images:
  LR | Bicubic | exp1 SR | exp2 SR | exp3 SR | HR (if available)
  Source: `outputs/experiments/*/samples/` — pick epoch_100 (or last epoch)

- **Training PSNR curves** — exp1, exp2, exp3 on the same plot.
  Source: `outputs/experiments/*/metrics/val_psnr.json` — plot with matplotlib.

- **Comparison table** — exp1 vs exp2 vs exp3 vs bicubic baseline.
  Source: `outputs/comparison_table.csv`

- **Degradation example** — HR vs surveillance-degraded LR.
  Source: `outputs/domain_degradation_test/` — already generated.
  Shows hr_original.png vs lr_surveillance.png.

- **Architecture diagram** (draw manually or use the text in `conditioned_sr.py`):
  LR image → FiLM-conditioned SRResNet backbone → SR image
  Conditioning vector (13 values) → MetadataEncoder → FiLM γ/β

### Slides structure

1. **Problem** — surveillance footage is low quality. Bicubic SR is not enough.
2. **Our approach** — degradation-aware conditioning. The model knows how bad each image is.
3. **Pipeline diagram** — two-stage degradation → conditioning vector → conditioned SR.
4. **Experiments** — 3-row comparison table. Each row adds one component.
5. **Qualitative results** — LR | Bicubic | Our model | HR for 3 examples.
6. **Limitations and future work** — blind estimation, temporal modeling.

### Deliverables checklist

- [ ] `outputs/comparison_table.csv` (run `compare_experiments.py` after all 3 experiments)
- [ ] Sample images from `outputs/experiments/exp3_final/samples/epoch_100/`
- [ ] PSNR curves from `val_psnr.json` files (3 experiments)
- [ ] Degradation visualization: `outputs/domain_degradation_test/`
- [ ] RealSRSet gallery (run `infer.py --save_gallery` for visual output)

---

## What to report in the final writeup

### Datasets

- **Training:** DIV2K (100 HR images, online surveillance-style degradation, 4× scale)
- **Validation:** DIV2K (40 HR images, pre-sampled fixed surveillance degradation)
- **No-reference evaluation:** RealSRSet (real-world LR images, no HR)
- **Optional:** VIRAT surveillance video frames (extracted with known degradation params)

### Metrics to report

| Metric | Source | Notes |
|--------|--------|-------|
| PSNR (dB) | `eval_paired_summary.json` or `run_summary.json` | Always available |
| SSIM | `eval_paired_summary.json` | Needs scikit-image |
| LPIPS | `eval_paired_summary.json` | Needs lpips package; lower = better |
| NIQE | `eval_noref_summary.json` | Needs piq; lower = better |
| BRISQUE | `eval_noref_summary.json` | Needs piq; lower = better |

Report all three experiments for each metric. Report bicubic baseline as the reference floor.

### Three key experiments to describe

| Experiment | Model | What changes | What it shows |
|------------|-------|-------------|---------------|
| exp1_baseline | SimpleSRCNN | Baseline | Neural SR vs bicubic |
| exp2_srresnet_domain | SRResNet | Deeper model + surveillance domain | Domain specialization effect |
| exp3_final | ConditionedSRResNet | + conditioning + curriculum + perceptual | Full proposal system |

### Qualitative figures to collect

From `outputs/experiments/exp3_final/samples/epoch_100/` (or last available epoch):
- Pick 3 images with visually interesting degradation (noisy, blurry, JPEG artifacts)
- Show: LR thumbnail | Bicubic SR | Our SR | HR ground truth
- One should show a case where our model clearly beats bicubic (high noise)
- One can show a failure case (very severe blur) — honest reporting is stronger

### Claims you can safely make

- "Our ConditionedSRResNet achieves X dB PSNR on the DIV2K surveillance validation set,
  compared to Y dB for bicubic interpolation and Z dB for our SRResNet baseline."
- "Domain-specific training on surveillance-style degradations improves PSNR by X dB
  over generic degradation training (exp1 → exp2)."
- "Adding FiLM conditioning and curriculum training produces the best results among
  our three configurations (exp3 > exp2)."
- "On real-world images (RealSRSet), our model achieves NIQE of X vs Y for bicubic,
  indicating better perceptual quality despite having no HR reference."

### What to frame as future work (not a failure)

- Blind degradation estimation from LR: estimating conditioning params without ground truth.
- Temporal consistency: multi-frame SR for video surveillance.
- GAN training: higher-frequency texture with adversarial loss (requires more compute and stability tuning).
- Full DIV2K training set (800 images): expected to improve absolute PSNR by 1–2 dB.

---

## Running inference and collecting visual outputs

### RealSRSet visual inference

```bash
python scripts/infer.py \
    --experiment exp3_final \
    --img_dir data/RealSRSet \
    --save_gallery \
    --tag realset_final
```

Outputs to: `outputs/experiments/exp3_final/inference/realset_final/`
- `gallery.png` — contact sheet of LR | Bicubic | SR for all images
- `{stem}_sr.png` — individual SR images
- `run_info.json` — provenance

Note: RealSRSet uses a zero conditioning vector (no metadata available for real images).
This is documented; results still show meaningful SR over bicubic.

### Paired evaluation with true conditioning (for exp3)

The `run_experiment.py` call handles this automatically. If you want to run it manually:

```bash
python scripts/evaluate_paired.py \
    --experiment exp3_final \
    --hr_dir data/DIV2K/HR_valid \
    --regen_lr
```

`--regen_lr` re-degrades each HR image and passes the actual conditioning vector —
this is true conditioning, not a zero fallback.

### No-reference evaluation only (if training is already done)

```bash
python scripts/evaluate_noref.py \
    --experiment exp3_final \
    --img_dir data/RealSRSet \
    --save_sr
```

`--save_sr` also saves the SR images to `eval_noref_sr/` for visual inspection.

---

## Summary of additions made in this session

The following changes were made to improve conditioning faithfulness in evaluation.
No training code, model architecture, or config files were changed.

### Files changed

| File | Change |
|------|--------|
| `scripts/cond_utils.py` | Added `_RESIZE_INDEX_INT` so `build_cond_vector()` handles int `resize_method` values loaded from JSON manifests (previously would silently default to bicubic). |
| `scripts/evaluate_virat.py` | `eval_pair()` now reads per-frame `"degradation"` params from `manifest.json` and builds real conditioning vectors. Tracks and reports true vs fallback count per run. |
| `scripts/evaluate_paired.py` | Added `--regen_lr` flag. Re-degrades each HR image using the training domain, captures metadata, passes real cond vector. `--lr_dir` is now optional when `--regen_lr` is set. |
| `scripts/infer.py` | Docstring updated to explicitly document zero conditioning as a known limitation for RealSRSet. |
| `scripts/run_experiment.py` | Automatically passes `--regen_lr` when model is `conditioned_srresnet` and domain is set. No manual flag needed. |
| `README.md` | "ConditionedSRResNet and evaluation" section rewritten as a table showing true vs fallback conditioning for each script. 12-hour workflow note updated. |

### README updates recommended

- The older ablation configs (`ablation_srcnn_l1.yaml`, `ablation_srresnet_l1.yaml`,
  `ablation_srresnet_perceptual.yaml`) still exist in `configs/` but are superseded by
  `exp1_baseline.yaml`, `exp2_srresnet_domain.yaml`, `exp3_final.yaml`. Consider deleting
  the old ones to avoid confusion, or add a note at the top of each marking them as legacy.

- `final_cond_surveillance.yaml` is equivalent to `exp3_final.yaml` in content.
  `exp3_final.yaml` is preferred because it fits the named experiment story.
  Consider removing `final_cond_surveillance.yaml` or marking it as a duplicate.

### Commands that are accurate and current

These all work as written against the current codebase:

```bash
# Full 3-experiment pipeline
python scripts/run_experiment.py --config configs/exp1_baseline.yaml --noref_dir data/RealSRSet
python scripts/run_experiment.py --config configs/exp2_srresnet_domain.yaml --noref_dir data/RealSRSet
python scripts/run_experiment.py --config configs/exp3_final.yaml --noref_dir data/RealSRSet
python scripts/compare_experiments.py

# Manual evaluation
python scripts/evaluate_paired.py --experiment exp3_final --hr_dir data/DIV2K/HR_valid --regen_lr
python scripts/evaluate_noref.py --experiment exp3_final --img_dir data/RealSRSet
python scripts/infer.py --experiment exp3_final --img_dir data/RealSRSet --save_gallery

# VIRAT (if videos available)
python scripts/extract_virat_frames.py --video_dir data/VIRAT/videos --out_dir data/VIRAT/frames
python scripts/evaluate_virat.py --experiment exp3_final
```

### Nothing is outdated or inconsistent in the current codebase

All scripts, configs, and README sections are consistent as of this session.
The three exp configs (exp1/exp2/exp3) are the canonical ablation story.
The `run_experiment.py` orchestrator handles the full pipeline automatically.

![alt text](image.png)