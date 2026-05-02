"""
train.py
--------
Trains a super-resolution model (SimpleSRCNN, SRResNet, or ConditionedSRResNet)
on your DIV2K image pairs.

What this script does, step by step:
  1. Reads all settings from configs/default.yaml (no more hardcoded constants)
  2. Fixes a random seed so results are reproducible run-to-run
  3. Creates an experiment output folder under outputs/experiments/<name>/
  4. Loads training image patches from HR_train / LR_train
  5. Loads full validation images from HR_valid / LR_valid
  6. For each epoch:
       a. Feeds batches of LR patches through the model
       b. Compares output to HR patches using MSE loss
       c. Adjusts model weights to reduce that loss (backpropagation)
       d. After training, evaluates on validation set → prints PSNR
       e. Saves last.pth (always) and best.pth (only if PSNR improved)
       f. Every few epochs, saves side-by-side visual comparisons
  7. Writes a run_summary.json with all results when done

Key concepts:
  epoch     : one full pass through the entire training dataset
  batch     : a small group of images processed together (e.g., 16 at once)
  loss      : a number measuring how wrong the model is (lower = better)
  MSE loss  : Mean Squared Error — average of (predicted - target)^2
  PSNR      : image quality score in dB (higher = better, goal: beat bicubic ~28-30 dB)
  backprop  : the algorithm that figures out how to nudge weights to reduce loss
  optimizer : the algorithm that applies those nudges (we use Adam)
  checkpoint: a saved snapshot of model weights so you can resume later
  seed      : a starting value for random number generators — fixing it makes
              every run produce the same results (reproducibility)

Usage:
  python scripts/train.py                               # use configs/default.yaml
  python scripts/train.py --config configs/my_run.yaml  # use a different config
  python scripts/train.py --experiment fast_test        # override experiment name
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

# Import our own modules (files in the same scripts/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DIV2KDataset
from degradation import degrade
from losses import build_loss
from metrics import compute_ssim
from models import build_model

# ── Experiment preset override ─────────────────────────────────────────────────
# Set True to force surveillance domain degradation regardless of the YAML config.
# This is the top-level toggle for the class project's primary experiment.
USE_SURVEILLANCE_PRESET = True


# ── Seed control — makes results reproducible ─────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Sets a fixed random seed for Python, NumPy, and PyTorch.

    Without this, every run produces slightly different results because weight
    initialisation and data shuffling both use random numbers.
    With a fixed seed, you can reproduce any run exactly by using the same seed.

    Why all three?
      - Python's random module: used for general randomness
      - NumPy: used internally by the dataset and degradation pipeline
      - PyTorch: used for weight initialisation and GPU operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # exact same GPU ops every run
    torch.backends.cudnn.benchmark = False       # no auto-tuning (would be non-deterministic)
    print(f"[seed] Fixed to {seed}")


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """
    Loads a YAML config file and returns its contents as a Python dict.

    YAML is a simple text format — cleaner than hardcoded constants.
    Example: 'learning_rate: 0.0001' in YAML becomes cfg['training']['learning_rate']
    in Python.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict, project_dir: Path) -> dict:
    """
    Converts relative paths in the config to absolute paths.

    All data/output paths in default.yaml are written relative to ml_project/
    (e.g. 'data/DIV2K/HR_train'). This function prepends the project directory
    so they work regardless of where you run the script from.
    """
    d = cfg["data"]
    for key in ("hr_train_dir", "hr_valid_dir", "lr_train_dir", "lr_valid_dir"):
        d[key] = str(project_dir / d[key])
    cfg["output"]["base_dir"] = str(project_dir / cfg["output"]["base_dir"])
    return cfg


# ── Experiment directory setup ─────────────────────────────────────────────────

def setup_experiment_dir(cfg: dict) -> Path:
    """
    Creates the experiment output folder and its subfolders.

    Each experiment gets its own folder so multiple runs never overwrite each other:
      outputs/experiments/<name>/
        checkpoints/   ← best.pth and last.pth
        metrics/       ← train_loss.json and val_psnr.json
        samples/       ← visual comparisons every few epochs
        logs/          ← reserved for future log files

    Returns the Path to the experiment root directory.
    """
    exp_name = cfg["experiment"]["name"]
    exp_dir = Path(cfg["output"]["base_dir"]) / exp_name
    for sub in ("checkpoints", "logs", "samples", "metrics"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"[output] {exp_dir}")
    return exp_dir


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Returns the best available device for training.

    CUDA = NVIDIA GPU (much faster for deep learning)
    MPS  = Apple Silicon GPU (Mac M1/M2)
    CPU  = fallback — works, but slower
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[device] CPU (no GPU found — training will be slower)")
    return device


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates PSNR between predicted and target tensors (values in [0, 1]).

    PSNR = 10 * log10(1 / MSE)

    Returns a float in dB. Typical values:
      ~28-30 dB  bicubic baseline
      ~30-32 dB  a small trained CNN
      ~32+ dB    a good modern model
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        if mse == 0:
            return float("inf")
        return 10 * math.log10(1.0 / mse)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimizer, epoch: int, psnr: float) -> None:
    """
    Saves a checkpoint containing everything needed to resume training.

    What we save:
      model weights   → model.state_dict()
      optimizer state → optimizer.state_dict()  (includes momentum, learning rate)
      epoch number    → so we know where to resume from
      psnr            → so we can restore the best-PSNR tracking
    """
    torch.save(
        {
            "epoch":     epoch,
            "psnr":      psnr,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model, optimizer, path: Path, device: torch.device
) -> tuple[int, float]:
    """
    Loads a checkpoint and restores model + optimizer state.

    Returns (epoch, psnr) from the saved checkpoint so training
    can continue from where it left off.
    """
    print(f"[resume] Loading {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    psnr  = state["psnr"]
    print(f"  Resuming from epoch {epoch}  (saved PSNR: {psnr:.2f} dB)")
    return epoch, psnr


# ── Sample image saving ────────────────────────────────────────────────────────

def save_sample_images(
    model,
    val_dataset,
    device: torch.device,
    exp_dir: Path,
    epoch: int,
    scale: int,
    num_images: int = 4,
    max_height: int = 360,
    use_cond: bool = False,
) -> None:
    """
    Saves visual comparisons of the model's SR output vs alternatives.

    For each sample, four panels are saved side-by-side in a single image:
      [LR (zoomed up) | Bicubic upsample | Model SR | HR ground truth]

    This lets you visually track whether the model is actually learning to
    recover details beyond what simple bicubic interpolation can do.

    Files are saved to:
      outputs/experiments/<name>/samples/epoch_NNN/img00_comparison.png
      outputs/experiments/<name>/samples/epoch_NNN/img00_lr.png
      outputs/experiments/<name>/samples/epoch_NNN/img00_bicubic.png
      outputs/experiments/<name>/samples/epoch_NNN/img00_sr.png
      outputs/experiments/<name>/samples/epoch_NNN/img00_hr.png

    max_height: validation images can be very large (2K). Images taller than
    this value are scaled down proportionally before saving so files stay
    manageable.
    """
    model.eval()
    out_dir = exp_dir / "samples" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_images = min(num_images, len(val_dataset))

    with torch.no_grad():
        for idx in range(num_images):
            item = val_dataset[idx]
            if use_cond:
                lr_tensor, hr_tensor, cond_tensor = item
                cond = cond_tensor.unsqueeze(0).to(device)   # (1, COND_DIM)
            else:
                lr_tensor, hr_tensor = item
                cond = None

            lr = lr_tensor.unsqueeze(0).to(device)   # (1, 3, lrH, lrW)
            hr = hr_tensor.unsqueeze(0).to(device)   # (1, 3, hrH, hrW)

            h, w = hr.shape[2], hr.shape[3]

            # Bicubic upsample — the baseline we want to beat (always shown)
            bicubic = F.interpolate(lr, size=(h, w), mode="bicubic",
                                    align_corners=False).clamp(0, 1)

            # Model SR — input depends on model type; pass cond if conditioning
            if model.expects_upsampled_input:
                model_input = F.interpolate(lr, size=(h, w), mode="bicubic",
                                            align_corners=False)
            else:
                model_input = lr
            sr = (model(model_input, cond) if cond is not None
                  else model(model_input)).clamp(0, 1)

            # LR shown at HR size using nearest-neighbor so you can see the actual
            # pixels without any smoothing (makes the sharpening effect more obvious)
            lr_display = F.interpolate(lr, size=(h, w),
                                       mode="nearest").clamp(0, 1)

            def to_pil(t: torch.Tensor) -> Image.Image:
                """Convert a (1, 3, H, W) float tensor in [0,1] to a PIL image."""
                arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                return Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))

            panels = [to_pil(lr_display), to_pil(bicubic), to_pil(sr), to_pil(hr)]
            labels = ["lr", "bicubic", "sr", "hr"]

            # Resize very tall images so saved files stay a manageable size
            if h > max_height:
                scale_factor = max_height / h
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)
                panels = [p.resize((new_w, new_h), Image.BICUBIC) for p in panels]
                ph, pw = new_h, new_w
            else:
                ph, pw = h, w

            # Save individual panels
            for label, panel in zip(labels, panels):
                panel.save(out_dir / f"img{idx:02d}_{label}.png")

            # Save 4-panel comparison strip: [LR | Bicubic | SR | HR]
            strip = Image.new("RGB", (pw * 4, ph))
            for i, panel in enumerate(panels):
                strip.paste(panel, (pw * i, 0))
            strip.save(out_dir / f"img{idx:02d}_comparison.png")

    print(f"  [samples] Saved {num_images} comparisons → {out_dir}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    scale: int,
    log_every: int,
    use_cond: bool = False,
) -> dict:
    """
    Runs one full pass through the training data.

    For each batch:
      1. Move data to GPU/CPU
      2. Prepare model input (bicubic pre-upsample for SRCNN, raw LR for ResNets)
      3. Forward pass — passes conditioning vector too when use_cond=True
      4. Calculate loss via CombinedLoss (pixel + optional perceptual)
      5. Backward pass: compute gradients
      6. Optimizer step: update weights
      7. Zero gradients for next batch

    Returns a dict of average loss values across all batches:
      {"total": float, "pixel": float, "perceptual": float (if enabled)}
    """
    model.train()
    num_batches     = len(loader)
    sum_components  = {}   # accumulates per-component totals

    for batch_idx, batch in enumerate(loader):
        # 1. Unpack batch — conditioned datasets return a 3-tuple
        if use_cond:
            lr_batch, hr_batch, cond_batch = batch
            cond_batch = cond_batch.to(device)   # (B, COND_DIM)
        else:
            lr_batch, hr_batch = batch
            cond_batch = None

        lr_batch = lr_batch.to(device)   # (B, 3, patch, patch)
        hr_batch = hr_batch.to(device)   # (B, 3, patch*scale, patch*scale)

        # 2. Prepare model input
        #    SRCNN expects bicubic-upsampled LR at HR size.
        #    SRResNet / ConditionedSRResNet upsample internally.
        if model.expects_upsampled_input:
            model_input = F.interpolate(lr_batch, scale_factor=scale,
                                        mode="bicubic", align_corners=False)
        else:
            model_input = lr_batch

        # 3-7. Standard PyTorch training step
        optimizer.zero_grad()
        if cond_batch is not None:
            sr_batch = model(model_input, cond_batch)   # conditioned forward pass
        else:
            sr_batch = model(model_input)               # standard forward pass

        loss, components = criterion(sr_batch, hr_batch)
        loss.backward()
        optimizer.step()

        # Accumulate component totals
        for k, v in components.items():
            sum_components[k] = sum_components.get(k, 0.0) + v

        if batch_idx % log_every == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Batch {batch_idx:4d}/{num_batches} | "
                  f"Loss {components['total']:.6f}")

    return {k: v / num_batches for k, v in sum_components.items()}


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    scale: int,
    use_cond: bool = False,
) -> tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Returns (avg_psnr, avg_ssim) across all validation images.
    """
    model.eval()
    psnr_scores = []
    ssim_scores = []

    for batch in loader:
        if use_cond:
            lr, hr, cond = batch
            cond = cond.to(device)
        else:
            lr, hr = batch
            cond = None

        lr = lr.to(device)
        hr = hr.to(device)

        if model.expects_upsampled_input:
            model_input = F.interpolate(lr, scale_factor=scale,
                                        mode="bicubic", align_corners=False)
        else:
            model_input = lr

        sr = model(model_input, cond) if cond is not None else model(model_input)
        psnr_scores.append(calculate_psnr(sr, hr))
        ssim_scores.append(compute_ssim(sr, hr))

    return (sum(psnr_scores) / len(psnr_scores),
            sum(ssim_scores) / len(ssim_scores))


# ── Main ──────────────────────────────────────────────────────────────────────

def _str2bool(v: str) -> bool:
    """Parse a boolean command-line argument: 'true/1/yes' → True, else False."""
    return v.strip().lower() not in ("false", "0", "no", "n")


def main():
    # ── Parse command-line arguments ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train SimpleSRCNN")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config relative to ml_project/ (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override the experiment name from the config file",
    )
    # ── New simple-mode overrides ──────────────────────────────────────────────
    parser.add_argument(
        "--use_online_degradation", type=_str2bool, default=None, metavar="BOOL",
        help="Override use_online_degradation (True/False)",
    )
    parser.add_argument(
        "--use_surv_preset", type=_str2bool, default=None, metavar="BOOL",
        help="Override surveillance preset (True=surveillance domain, False=clean LR)",
    )
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Save checkpoints directly to this dir (bypasses experiment dir)",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Override CSV training log file path",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Override number of training epochs from config",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    # project_dir is ml_project/ — the parent of this scripts/ folder
    project_dir = Path(__file__).resolve().parent.parent
    config_path = project_dir / args.config

    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}")
        print("  Make sure you are running from ml_project/ or pass --config correctly.")
        sys.exit(1)

    cfg = load_config(config_path)
    cfg = resolve_paths(cfg, project_dir)

    # Apply surveillance preset override (module-level flag at top of file)
    if USE_SURVEILLANCE_PRESET:
        cfg["data"]["domain"] = "surveillance"
        cfg["data"]["use_online_degradation"] = True
        print("[preset] USE_SURVEILLANCE_PRESET=True → "
              "domain=surveillance, online degradation enabled")

    # ── Apply command-line overrides (highest precedence) ─────────────────────
    if args.use_online_degradation is not None:
        cfg["data"]["use_online_degradation"] = args.use_online_degradation
        print(f"[arg] --use_online_degradation {args.use_online_degradation}")
    if args.use_surv_preset is not None:
        if args.use_surv_preset:
            cfg["data"]["domain"] = "surveillance"
            cfg["data"]["use_online_degradation"] = True
            print("[arg] --use_surv_preset True → domain=surveillance")
        else:
            cfg["data"]["domain"] = None
            print("[arg] --use_surv_preset False → domain=None (clean/generic)")
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
        print(f"[arg] --num_epochs {args.num_epochs}")

    # Allow overriding the experiment name on the command line without editing YAML
    if args.experiment:
        cfg["experiment"]["name"] = args.experiment

    # ── Seed — set this before creating any datasets or models ────────────────
    set_seed(cfg["experiment"]["seed"])

    # ── Experiment output directory ───────────────────────────────────────────
    exp_dir = setup_experiment_dir(cfg)
    # Save a copy of the config so evaluation scripts can auto-discover it.
    shutil.copy2(config_path, exp_dir / "config.yaml")

    # ── Checkpoint and log-file paths (may be overridden by --checkpoint_dir /
    #    --log_file command-line args to support the three-experiment interface)
    if args.checkpoint_dir is not None:
        ckpt_dir = (Path(args.checkpoint_dir) if Path(args.checkpoint_dir).is_absolute()
                    else project_dir / args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[checkpoint_dir] {ckpt_dir}")
    else:
        ckpt_dir = exp_dir / "checkpoints"

    if args.log_file is not None:
        csv_log_path = (Path(args.log_file) if Path(args.log_file).is_absolute()
                        else project_dir / args.log_file)
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[log_file] {csv_log_path}")
    else:
        csv_log_path = project_dir / "outputs" / "training_log_srcnn.csv"

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()

    # ── Unpack config into plain variables (keeps the rest of the code clean) ─
    t = cfg["training"]
    d = cfg["data"]
    o = cfg["output"]

    NUM_EPOCHS         = t["num_epochs"]
    BATCH_SIZE         = t["batch_size"]
    LEARNING_RATE      = t["learning_rate"]
    PATCH_SIZE         = t["patch_size"]
    SCALE              = t["scale"]
    LOG_EVERY          = t["log_every"]
    USE_ONLINE_DEG     = d["use_online_degradation"]
    DOMAIN             = d.get("domain", None)   # None = generic, or "mobile" etc.
    SAVE_SAMPLES_EVERY = o["save_samples_every"]
    NUM_SAMPLE_IMGS    = o["num_sample_images"]

    # Determine whether this run uses degradation-aware conditioning.
    # Derived from config so datasets are created correctly BEFORE model init.
    _model_name_cfg = cfg.get("model", {}).get("name", "srcnn").lower().strip()
    USE_COND        = (_model_name_cfg == "conditioned_srresnet")

    if USE_COND and DOMAIN is None:
        print("\n[error] conditioned_srresnet requires a domain in the config.")
        print("  Add this under the 'data:' section of your YAML:")
        print("    domain: mobile   # or 'surveillance' or 'dashcam'")
        sys.exit(1)

    # Curriculum training: progressive degradation difficulty.
    # Only active when curriculum.enabled=true AND a domain is set (Mode C).
    curriculum_cfg = cfg.get("curriculum", {"enabled": False, "stages": []})
    USE_CURRICULUM = curriculum_cfg.get("enabled", False) and DOMAIN is not None

    if curriculum_cfg.get("enabled", False) and DOMAIN is None:
        print("[warn] curriculum.enabled=true but data.domain is null — "
              "curriculum requires domain degradation (Mode C).  Disabling.")

    if USE_CURRICULUM:
        from curriculum import get_active_stage
        _stage_names = [s["name"] for s in curriculum_cfg["stages"]]
        print(f"[curriculum] Enabled — stages: {_stage_names}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    #
    # Training: use patch_size so batches are small fixed-size crops
    # Validation: use patch_size=None to evaluate on full images
    #
    # Three modes (controlled by use_online_degradation and domain in config):
    #   Stage 1: pre-saved LR files (use_online_degradation: false)
    #   Stage 2: generic random degradation (domain: null)
    #   Stage 3: domain-conditioned degradation (domain: "mobile" etc.)
    #
    # Note: the seed is already set above, so any pre-sampling that happens
    # inside DIV2KDataset (for deterministic validation) is reproducible.
    #
    print("\nLoading datasets...")

    if USE_ONLINE_DEG:
        if DOMAIN is not None:
            # Stage 3: domain-conditioned degradation preset.
            # DIV2KDataset handles calling degrade_domain() internally.
            # Validation params are pre-sampled once so PSNR is comparable
            # across epochs (deterministic because seed is already fixed above).
            cond_tag = " + conditioning vectors" if USE_COND else ""
            print(f"  Mode: domain-conditioned degradation  (domain='{DOMAIN}'){cond_tag}")
            train_dataset = DIV2KDataset(
                hr_dir=d["hr_train_dir"],
                patch_size=PATCH_SIZE,
                scale=SCALE,
                domain=DOMAIN,
                return_cond_vector=USE_COND,
                upsample_lr=False,  # train_one_epoch handles upsampling
            )
            val_dataset = DIV2KDataset(
                hr_dir=d["hr_valid_dir"],
                patch_size=None,
                scale=SCALE,
                domain=DOMAIN,
                return_cond_vector=USE_COND,
                upsample_lr=False,
            )
        else:
            # Stage 2: generic random degradation (original behaviour).
            print("  Mode: generic online degradation  (Stage 2)")
            deg_fn = lambda img: degrade(img, scale=SCALE)
            train_dataset = DIV2KDataset(
                hr_dir=d["hr_train_dir"],
                degradation_fn=deg_fn,
                patch_size=PATCH_SIZE,
                scale=SCALE,
                upsample_lr=False,
            )
            val_dataset = DIV2KDataset(
                hr_dir=d["hr_valid_dir"],
                lr_dir=d["lr_valid_dir"],
                patch_size=None,
                scale=SCALE,
                upsample_lr=False,
            )
    else:
        # Stage 1: load pre-saved LR files from disk (requires prepare_data.py)
        print("  Mode: pre-saved LR files  (Stage 1)")
        train_dataset = DIV2KDataset(
            hr_dir=d["hr_train_dir"],
            lr_dir=d["lr_train_dir"],
            patch_size=PATCH_SIZE,
            scale=SCALE,
            upsample_lr=False,
        )
        val_dataset = DIV2KDataset(
            hr_dir=d["hr_valid_dir"],
            lr_dir=d["lr_valid_dir"],
            patch_size=None,
            scale=SCALE,
            upsample_lr=False,
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1,
                              shuffle=False, num_workers=0)

    print(f"  Training images  : {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = build_loss(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Adam is an adaptive learning-rate optimiser — the standard choice
    # for image tasks. Converges faster than plain SGD.

    num_params = sum(p.numel() for p in model.parameters())
    model_name = type(model).__name__
    print(f"\nModel: {model_name} ({num_params:,} parameters)")

    # ── Resume from checkpoint? ───────────────────────────────────────────────
    start_epoch = 1
    best_psnr   = 0.0
    last_ckpt   = ckpt_dir / "last.pth"

    if last_ckpt.exists():
        answer = input(f"\nCheckpoint found. Resume training? [y/N]: ").strip().lower()
        if answer == "y":
            resumed_epoch, best_psnr = load_checkpoint(model, optimizer, last_ckpt, device)
            start_epoch = resumed_epoch + 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Model      : {model_name} ({num_params:,} params)")
    print(f"  Epochs     : {NUM_EPOCHS}  |  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}  |  Seed      : {cfg['experiment']['seed']}")
    print(f"  Output dir : {exp_dir}")
    print(f"{'='*60}\n")

    # Lists that grow each epoch — written to JSON after every epoch
    # so you can inspect training progress without waiting for the run to finish
    train_loss_history = []   # [{epoch:1, loss:0.012, pixel:0.010, perceptual:0.002}, ...]
    val_psnr_history   = []   # [{epoch: 1, psnr: 28.4},  ...]
    curriculum_log     = []   # [{epoch: 1, stage: "easy", blur_max: 1.0, ...}, ...]

    run_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{NUM_EPOCHS} ──")

        # ── Curriculum stage update ────────────────────────────────────────
        # Must happen before train_one_epoch so the dataset uses the right
        # parameter ranges immediately.
        if USE_CURRICULUM:
            active_stage = get_active_stage(epoch, curriculum_cfg)
            train_dataset.curriculum_stage = active_stage   # mutable update
            stage_name = active_stage["name"]
            print(f"  [curriculum] Stage: {stage_name}  "
                  f"(blur {active_stage['blur_sigma']}, "
                  f"noise {active_stage['noise_sigma']}, "
                  f"jpeg {active_stage['jpeg_quality']})")
            curriculum_log.append({
                "epoch":      epoch,
                "stage":      stage_name,
                "blur_sigma": active_stage["blur_sigma"],
                "noise_sigma": active_stage["noise_sigma"],
                "jpeg_quality": active_stage["jpeg_quality"],
                "second_stage_probability": active_stage.get("second_stage_probability", 0.0),
            })

        # ── Train ──────────────────────────────────────────────────────────
        loss_components = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, NUM_EPOCHS, SCALE, LOG_EVERY,
            use_cond=USE_COND,
        )
        avg_loss = loss_components["total"]

        # ── Validate ───────────────────────────────────────────────────────
        avg_psnr, avg_ssim = validate(model, val_loader, device, SCALE,
                                      use_cond=USE_COND)

        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        # Build a compact loss string for the console showing all active terms
        extra_terms = [f"{k} {v:.6f}"
                       for k, v in loss_components.items()
                       if k != "total"]
        loss_detail = (f"  ({' | '.join(extra_terms)})" if extra_terms else "")
        print(f"\n  Train Loss : {avg_loss:.6f}{loss_detail}")
        print(f"  Val PSNR   : {avg_psnr:.2f} dB  (best: {best_psnr:.2f} dB)"
              + ("  <- new best!" if is_best else ""))
        print(f"  Val SSIM   : {avg_ssim:.4f}")

        # ── Record metrics ─────────────────────────────────────────────────
        train_loss_history.append({"epoch": epoch, "loss": avg_loss,
                                   **{k: round(v, 6) for k, v in loss_components.items()
                                      if k != "total"}})
        val_psnr_history.append({"epoch": epoch, "psnr": avg_psnr,
                                  "ssim": round(avg_ssim, 6)})

        # ── CSV logging ────────────────────────────────────────────────────
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_log_path.exists()
        with open(csv_log_path, "a", newline="") as _f:
            import csv as _csv
            _w = _csv.writer(_f)
            if write_header:
                _w.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])
            _w.writerow([epoch, round(avg_loss, 6),
                         round(avg_psnr, 4), round(avg_ssim, 6)])

        # Save JSON after every epoch — live updates you can read mid-run
        with open(exp_dir / "metrics" / "train_loss.json", "w") as f:
            json.dump(train_loss_history, f, indent=2)
        with open(exp_dir / "metrics" / "val_psnr.json", "w") as f:
            json.dump(val_psnr_history, f, indent=2)
        if USE_CURRICULUM:
            with open(exp_dir / "metrics" / "curriculum_log.json", "w") as f:
                json.dump(curriculum_log, f, indent=2)

        # ── Save checkpoints ───────────────────────────────────────────────
        # last.pth: always overwritten — used for resuming
        save_checkpoint(ckpt_dir / "last.pth",
                        model, optimizer, epoch, avg_psnr)

        # best.pth: only written when PSNR improves — this is the final model
        if is_best:
            save_checkpoint(ckpt_dir / "best.pth",
                            model, optimizer, epoch, avg_psnr)
            print(f"  [checkpoint] best.pth updated (PSNR {best_psnr:.2f} dB)")

        # ── Save sample images every N epochs ─────────────────────────────
        # Also always saves on the final epoch
        if epoch % SAVE_SAMPLES_EVERY == 0 or epoch == NUM_EPOCHS:
            save_sample_images(
                model, val_dataset, device, exp_dir,
                epoch, SCALE, num_images=NUM_SAMPLE_IMGS,
                use_cond=USE_COND,
            )

    # ── Final run summary ─────────────────────────────────────────────────────
    run_seconds = round(time.time() - run_start, 1)

    _loss_cfg = cfg.get("loss", {})
    summary = {
        "experiment":             cfg["experiment"]["name"],
        "model":                  model_name,
        "conditioned":            USE_COND,
        "curriculum_enabled":     USE_CURRICULUM,
        "curriculum_stages":      [s["name"] for s in curriculum_cfg.get("stages", [])]
                                  if USE_CURRICULUM else None,
        "seed":                   cfg["experiment"]["seed"],
        "num_epochs":             NUM_EPOCHS,
        "batch_size":             BATCH_SIZE,
        "learning_rate":          LEARNING_RATE,
        "patch_size":             PATCH_SIZE,
        "scale":                  SCALE,
        "use_online_degradation": USE_ONLINE_DEG,
        "domain":                 DOMAIN,
        "loss": {
            "pixel_type":          _loss_cfg.get("pixel", {}).get("type", "l1"),
            "pixel_weight":        _loss_cfg.get("pixel", {}).get("weight", 1.0),
            "perceptual_enabled":  _loss_cfg.get("perceptual", {}).get("enabled", False),
            "perceptual_weight":   _loss_cfg.get("perceptual", {}).get("weight", 0.1),
            "perceptual_layer":    _loss_cfg.get("perceptual", {}).get("layer", "relu2_2"),
        },
        "best_psnr_db":           round(best_psnr, 4),
        "final_train_loss":       round(train_loss_history[-1]["loss"], 6),
        "total_train_images":     len(train_dataset),
        "total_val_images":       len(val_dataset),
        "runtime_seconds":        run_seconds,
    }

    summary_path = exp_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best PSNR  : {best_psnr:.2f} dB")
    print(f"  Output dir : {exp_dir}")
    print(f"  Summary    : {summary_path}")
    print(f"  Runtime    : {run_seconds:.1f}s")
    print(f"{'='*60}")
    print("\nNext step: compare to the bicubic baseline")
    print("  python scripts/baseline.py")


if __name__ == "__main__":
    # The  if __name__ == "__main__":  guard is required on Windows when using
    # DataLoader with num_workers > 0. Always include it.
    main()