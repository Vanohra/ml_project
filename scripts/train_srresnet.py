"""
train_srresnet.py
-----------------
Self-contained training script for SRResNet on the surveillance domain.

Differences from train.py (SimpleSRCNN):
  - Model  : SRResNet (imports from model.py, ~957k params)
  - Input  : raw LR at native resolution (SRResNet upsamples internally)
  - Loss   : L1 — sharper results than MSE for deeper models
  - LR     : 1e-4 (Adam)
  - Epochs : 100
  - Batch  : 8
  - Domain : surveillance preset (heavy blur + noise + JPEG)
  - Output : outputs/checkpoints_srresnet/  and  outputs/training_log_srresnet.csv

Usage (from ml_project/ directory):
  python scripts/train_srresnet.py
  python scripts/train_srresnet.py --resume           # resume from last.pth

The script is intentionally simple: no YAML config, no perceptual loss, no
curriculum — just the essentials needed to produce SRResNet results for the
three-experiment comparison.
"""

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
# Allows importing from the same scripts/ directory regardless of where the
# script is launched from.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from dataset import DIV2KDataset
from degradation import degrade, get_domain_preset
from metrics import compute_psnr, compute_ssim
from model import SRResNet

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_EPOCHS             = 100
BATCH_SIZE             = 8
LEARNING_RATE          = 1e-4
PATCH_SIZE             = 48       # LR patch size; HR patch = 48×4 = 192
SCALE                  = 4
SEED                   = 42
LOG_EVERY              = 20       # print loss every N batches
SAVE_SAMPLES_EVERY     = 10       # save visual comparisons every N epochs
NUM_SAMPLE_IMGS        = 4

USE_SURVEILLANCE_PRESET = True    # always True for this script

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent
HR_TRAIN_DIR  = PROJECT_DIR / "data" / "DIV2K" / "HR_train"
HR_VALID_DIR  = PROJECT_DIR / "data" / "DIV2K" / "HR_valid"
CKPT_DIR      = PROJECT_DIR / "outputs" / "checkpoints_srresnet"
SAMPLES_DIR   = PROJECT_DIR / "outputs" / "samples_srresnet"
CSV_LOG_PATH  = PROJECT_DIR / "outputs" / "training_log_srresnet.csv"


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple Silicon GPU (MPS)")
    else:
        dev = torch.device("cpu")
        print("[device] CPU (no GPU — training will be slow)")
    return dev


def save_checkpoint(path: Path, model, optimizer,
                    epoch: int, psnr: float, ssim: float) -> None:
    torch.save({
        "epoch":     epoch,
        "psnr":      psnr,
        "ssim":      ssim,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path: Path,
                    device: torch.device) -> tuple[int, float, float]:
    print(f"[resume] Loading {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    psnr  = state.get("psnr", 0.0)
    ssim  = state.get("ssim", 0.0)
    print(f"  Resuming from epoch {epoch}  "
          f"(saved PSNR {psnr:.2f} dB, SSIM {ssim:.4f})")
    return epoch, psnr, ssim


def append_csv_row(path: Path, row: list) -> None:
    """Appends one row to a CSV, writing a header if the file is new."""
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])
        w.writerow(row)


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader: DataLoader, optimizer,
                    criterion, device: torch.device,
                    epoch: int, num_epochs: int) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (lr_batch, hr_batch) in enumerate(loader):
        lr_batch = lr_batch.to(device)   # (B, 3, patch, patch)    raw LR
        hr_batch = hr_batch.to(device)   # (B, 3, patch*4, patch*4)

        # SRResNet takes raw LR — NO bicubic pre-upsample here
        optimizer.zero_grad()
        sr_batch = model(lr_batch)
        loss     = criterion(sr_batch, hr_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % LOG_EVERY == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Batch {batch_idx:4d}/{num_batches} | "
                  f"L1 Loss {loss.item():.6f}")

    return total_loss / num_batches


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_scores = []
    ssim_scores = []

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)
        psnr_scores.append(compute_psnr(sr, hr))
        ssim_scores.append(compute_ssim(sr, hr))

    return (sum(psnr_scores) / len(psnr_scores),
            sum(ssim_scores) / len(ssim_scores))


# ── Visual sample saving ──────────────────────────────────────────────────────

def save_sample_images(model, val_dataset, device: torch.device,
                       epoch: int, num: int = 4) -> None:
    from PIL import Image

    model.eval()
    out_dir = SAMPLES_DIR / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    num = min(num, len(val_dataset))

    with torch.no_grad():
        for idx in range(num):
            lr_t, hr_t = val_dataset[idx]
            lr = lr_t.unsqueeze(0).to(device)
            hr = hr_t.unsqueeze(0).to(device)
            h, w = hr.shape[2], hr.shape[3]

            bicubic = F.interpolate(lr, size=(h, w), mode="bicubic",
                                    align_corners=False).clamp(0, 1)
            sr = model(lr).clamp(0, 1)
            lr_display = F.interpolate(lr, size=(h, w),
                                       mode="nearest").clamp(0, 1)

            def _pil(t):
                arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                return Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))

            panels = [_pil(lr_display), _pil(bicubic), _pil(sr), _pil(hr)]
            labels = ["lr", "bicubic", "sr", "hr"]

            max_h = 360
            if h > max_h:
                scale_f = max_h / h
                new_h, new_w = int(h * scale_f), int(w * scale_f)
                panels = [p.resize((new_w, new_h), Image.BICUBIC) for p in panels]
                ph, pw = new_h, new_w
            else:
                ph, pw = h, w

            for label, panel in zip(labels, panels):
                panel.save(out_dir / f"img{idx:02d}_{label}.png")

            strip = Image.new("RGB", (pw * 4, ph))
            for i, panel in enumerate(panels):
                strip.paste(panel, (pw * i, 0))
            strip.save(out_dir / f"img{idx:02d}_comparison.png")

    print(f"  [samples] Saved {num} comparisons → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SRResNet on surveillance domain")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from outputs/checkpoints_srresnet/last.pth")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(SEED)
    device = get_device()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    surv_cfg = get_domain_preset("surveillance")

    deg_fn_train = lambda img: degrade(img, config=surv_cfg)

    # Validation: fix seed before creating dataset so degradation is reproducible
    torch.manual_seed(SEED)
    random.seed(SEED)

    try:
        train_dataset = DIV2KDataset(
            hr_dir=HR_TRAIN_DIR,
            degradation_fn=deg_fn_train,
            patch_size=PATCH_SIZE,
            scale=SCALE,
            upsample_lr=False,   # SRResNet takes raw LR
        )
        val_dataset = DIV2KDataset(
            hr_dir=HR_VALID_DIR,
            degradation_fn=deg_fn_train,
            patch_size=None,       # full images for validation
            scale=SCALE,
            upsample_lr=False,
        )
    except FileNotFoundError as e:
        print(f"\n[error] {e}")
        print("  Place HR images in:")
        print(f"    {HR_TRAIN_DIR}")
        print(f"    {HR_VALID_DIR}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1,
                              shuffle=False, num_workers=0)

    print(f"  Training images  : {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = SRResNet(scale=SCALE, num_res_blocks=8, num_features=64).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5
    )  # halve LR every 25 epochs: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → …

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SRResNet ({num_params:,} parameters)")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_psnr   = 0.0
    last_ckpt   = CKPT_DIR / "last.pth"

    if args.resume and last_ckpt.exists():
        resumed_epoch, best_psnr, _ = load_checkpoint(
            model, optimizer, last_ckpt, device)
        start_epoch = resumed_epoch + 1
    elif last_ckpt.exists() and not args.resume:
        answer = input(
            f"\nCheckpoint found at {last_ckpt}. Resume? [y/N]: "
        ).strip().lower()
        if answer == "y":
            resumed_epoch, best_psnr, _ = load_checkpoint(
                model, optimizer, last_ckpt, device)
            start_epoch = resumed_epoch + 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SRResNet Training — Surveillance Domain")
    print(f"  Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Loss: L1  |  Scale: {SCALE}×  |  Seed: {SEED}")
    print(f"  Checkpoints → {CKPT_DIR}")
    print(f"  CSV log     → {CSV_LOG_PATH}")
    print(f"{'='*60}\n")

    run_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{NUM_EPOCHS} ──")

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, NUM_EPOCHS,
        )

        avg_psnr, avg_ssim = validate(model, val_loader, device)

        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        print(f"\n  Train Loss : {avg_loss:.6f}")
        print(f"  Val PSNR   : {avg_psnr:.2f} dB  (best: {best_psnr:.2f} dB)"
              + ("  <- new best!" if is_best else ""))
        print(f"  Val SSIM   : {avg_ssim:.4f}")

        # ── LR schedule step ───────────────────────────────────────────────
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        if epoch % 25 == 0:
            print(f"  [scheduler] LR → {cur_lr:.2e}")

        # ── Checkpoints ────────────────────────────────────────────────────
        save_checkpoint(CKPT_DIR / "last.pth",
                        model, optimizer, epoch, avg_psnr, avg_ssim)
        if is_best:
            save_checkpoint(CKPT_DIR / "best.pth",
                            model, optimizer, epoch, avg_psnr, avg_ssim)
            print(f"  [checkpoint] best.pth updated (PSNR {best_psnr:.2f} dB)")

        # ── CSV log ────────────────────────────────────────────────────────
        append_csv_row(CSV_LOG_PATH, [
            epoch,
            round(avg_loss, 6),
            round(avg_psnr, 4),
            round(avg_ssim, 6),
        ])

        # ── Sample images ──────────────────────────────────────────────────
        if epoch % SAVE_SAMPLES_EVERY == 0 or epoch == NUM_EPOCHS:
            save_sample_images(model, val_dataset, device,
                               epoch, num=NUM_SAMPLE_IMGS)

    runtime = round(time.time() - run_start, 1)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best PSNR  : {best_psnr:.2f} dB")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  CSV log    : {CSV_LOG_PATH}")
    print(f"  Runtime    : {runtime:.1f}s")
    print(f"{'='*60}")
    print("\nNext: run evaluate.py to compare all three models.")
    print("  python scripts/evaluate.py")


if __name__ == "__main__":
    main()
