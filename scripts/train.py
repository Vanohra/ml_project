"""
train.py
--------
Trains SimpleSRCNN on your DIV2K image pairs.

What this script does, step by step:
  1. Loads training image patches from HR_train / LR_train
  2. Loads full validation images from HR_val / LR_val
  3. For each epoch:
       a. Feeds batches of LR patches through the model
       b. Compares output to HR patches using MSE loss
       c. Adjusts model weights to reduce that loss (backpropagation)
       d. After training, evaluates on validation set → prints PSNR
       e. Saves a checkpoint (the model weights + training state)
  4. Keeps the best-ever checkpoint saved separately

Key concepts:
  epoch     : one full pass through the entire training dataset
  batch     : a small group of images processed together (e.g., 16 at once)
  loss      : a number measuring how wrong the model is (lower = better)
  MSE loss  : Mean Squared Error — average of (predicted - target)^2
  PSNR      : image quality score in dB (higher = better, goal: beat bicubic ~28-30 dB)
  backprop  : the algorithm that figures out how to nudge weights to reduce loss
  optimizer : the algorithm that applies those nudges (we use Adam)
  checkpoint: a saved snapshot of model weights so you can resume later
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import our own modules (files in the same folder)
from dataset import DIV2KDataset
from degradation import degrade
from model import SimpleSRCNN


# ── Settings — edit these to control training ─────────────────────────────────

NUM_EPOCHS    = 50       # How many full passes through the training data
                         # More epochs = more training, but can overfit if too many

BATCH_SIZE    = 16       # Images processed together per step
                         # Lower if you get CUDA out-of-memory errors (try 8 or 4)

LEARNING_RATE = 1e-4     # How large each weight update step is
                         # Too high → unstable training; too low → very slow

PATCH_SIZE    = 48       # LR crop size for training (HR crop = 48 * 4 = 192)
                         # Smaller = faster per batch; larger = more context per sample

SCALE         = 4        # Upscale factor — must match prepare_data.py

# Stage 2: set True to generate LR images live using degradation.py
# Set False to fall back to loading pre-saved LR files from disk (Stage 1)
USE_ONLINE_DEGRADATION = True

LOG_EVERY     = 10       # Print loss every N batches

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data" / "DIV2K"
CKPT_DIR      = BASE_DIR / "outputs" / "checkpoints"

# ──────────────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    """
    Returns the best available device for training.

    CUDA = NVIDIA GPU (much faster for deep learning)
    MPS  = Apple Silicon GPU (Mac M1/M2)
    CPU  = fallback — works, but slower

    PyTorch automatically uses whichever you have.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU  (no GPU found — training will be slower)")
    return device


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


def save_checkpoint(model, optimizer, epoch, psnr, is_best, ckpt_dir):
    """
    Saves a checkpoint file containing everything needed to resume training.

    What we save:
      model weights  → model.state_dict()
      optimizer state → optimizer.state_dict()  (includes learning rate, momentum)
      epoch number   → so we know where to resume from
      psnr           → so we can track whether this epoch was the best

    Two files are always maintained:
      last.pth  → the most recent epoch (always overwritten)
      best.pth  → the epoch with the highest validation PSNR (updated only if improved)
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":     epoch,
        "psnr":      psnr,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # Always save the latest checkpoint
    torch.save(state, ckpt_dir / "last.pth")

    # Also save it as best.pth if this is the highest PSNR so far
    if is_best:
        torch.save(state, ckpt_dir / "best.pth")
        print(f"  *** New best model saved (PSNR {psnr:.2f} dB) ***")


def load_checkpoint(model, optimizer, ckpt_path, device):
    """
    Loads a previously saved checkpoint so training can resume.

    Returns the epoch to start from (so epoch count continues correctly).
    """
    print(f"Resuming from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    start_epoch = state["epoch"] + 1
    print(f"  Resuming from epoch {start_epoch}  (previous PSNR: {state['psnr']:.2f} dB)")
    return start_epoch


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one full pass through the training data.

    For each batch:
      1. Move data to GPU/CPU
      2. Bicubic-upsample LR to HR size  → this is the model's input
      3. Forward pass: model predicts SR
      4. Calculate MSE loss vs HR target
      5. Backward pass: compute gradients
      6. Optimizer step: update weights
      7. Zero gradients for next batch (important! gradients accumulate otherwise)

    Returns the average loss across all batches.
    """
    model.train()   # tell the model it's in training mode (enables dropout etc.)
    total_loss = 0.0

    for batch_idx, (lr_batch, hr_batch) in enumerate(loader):
        # 1. Move tensors to the training device
        lr_batch = lr_batch.to(device)   # shape: (B, 3, patch, patch)
        hr_batch = hr_batch.to(device)   # shape: (B, 3, patch*scale, patch*scale)

        # 2. Bicubic upsample LR to HR size
        #    F.interpolate resizes tensors. mode='bicubic' matches what
        #    prepare_data.py used to create LR images.
        #    align_corners=False is the standard setting for image resizing.
        lr_up = F.interpolate(lr_batch,
                              scale_factor=SCALE,
                              mode="bicubic",
                              align_corners=False)  # shape: (B, 3, HR_h, HR_w)

        # 3. Forward pass through model
        sr_batch = model(lr_up)           # shape: (B, 3, HR_h, HR_w)

        # 4. Calculate loss
        loss = criterion(sr_batch, hr_batch)

        # 5-7. Standard PyTorch training step (always these three lines in order)
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        total_loss += loss.item()

        if batch_idx % LOG_EVERY == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader)} "
                  f"| Loss {loss.item():.6f}")

    return total_loss / len(loader)   # average loss for this epoch


@torch.no_grad()   # disables gradient tracking — saves memory during evaluation
def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    During validation:
      - We use full images (not crops) for a fair PSNR measurement
      - We do NOT update weights (no backward pass)
      - We measure PSNR on each image and return the average

    Returns the average PSNR across all validation images.
    """
    model.eval()   # switches model to evaluation mode (disables dropout etc.)
    psnr_scores = []

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        # Upscale LR to HR size
        lr_up = F.interpolate(lr,
                              scale_factor=SCALE,
                              mode="bicubic",
                              align_corners=False)

        # Get model prediction
        sr = model(lr_up)   # already clamped to [0,1] by model.forward

        psnr_scores.append(calculate_psnr(sr, hr))

    return sum(psnr_scores) / len(psnr_scores)


def main():
    print("=" * 60)
    print("SimpleSRCNN Training")
    print("=" * 60)

    device = get_device()

    # ── Datasets ──────────────────────────────────────────────────────────────
    #
    # Training: use patch_size so batches are small fixed-size crops
    # Validation: use patch_size=None to evaluate on full images
    #
    print("\nLoading datasets...")

    if USE_ONLINE_DEGRADATION:
        # Stage 2: generate LR images live from HR using randomised degradation.
        # degradation_fn is called once per image per epoch with fresh random params.
        # Validation still uses pre-saved LR so PSNR is comparable across epochs.
        print("  Mode: online degradation (Stage 2)")
        deg_fn = lambda img: degrade(img, scale=SCALE)
        train_dataset = DIV2KDataset(
            hr_dir=DATA_DIR / "HR_train",
            degradation_fn=deg_fn,
            patch_size=PATCH_SIZE,
            scale=SCALE,
        )
    else:
        # Stage 1: load pre-saved LR files from disk (requires prepare_data.py)
        print("  Mode: pre-saved LR files (Stage 1)")
        train_dataset = DIV2KDataset(
            hr_dir=DATA_DIR / "HR_train",
            lr_dir=DATA_DIR / "LR_train",
            patch_size=PATCH_SIZE,
            scale=SCALE,
        )

    val_dataset = DIV2KDataset(
        hr_dir=DATA_DIR / "HR_valid",
        lr_dir=DATA_DIR / "LR_valid",
        patch_size=None,          # full images for validation
        scale=SCALE,
    )
    print(f"  Training images : {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    #
    # DataLoader feeds batches to the model automatically.
    #   shuffle=True  : randomises order each epoch (important for training)
    #   batch_size=1  : process one full image at a time during validation
    #   num_workers=0 : safest on Windows; set to 2-4 on Linux/Mac for speed
    #
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=0)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = SimpleSRCNN().to(device)
    criterion = nn.MSELoss()
    # nn.MSELoss = Mean Squared Error: average of (predicted_pixel - true_pixel)^2
    # Common loss for regression/image tasks. Lower is better.

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Adam is an adaptive learning-rate optimiser. It's the standard choice
    # for image tasks — it usually converges faster than plain SGD.

    # Count and display model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SimpleSRCNN  ({num_params:,} parameters)")

    # ── Resume from checkpoint? ───────────────────────────────────────────────
    start_epoch  = 0
    best_psnr    = 0.0
    last_ckpt    = CKPT_DIR / "last.pth"

    if last_ckpt.exists():
        answer = input("\nCheckpoint found. Resume training? [y/N]: ").strip().lower()
        if answer == "y":
            start_epoch = load_checkpoint(model, optimizer, last_ckpt, device)
            # Try to recover best_psnr from best checkpoint
            best_ckpt = CKPT_DIR / "best.pth"
            if best_ckpt.exists():
                best_psnr = torch.load(best_ckpt, map_location="cpu")["psnr"]

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training for {NUM_EPOCHS} epochs "
          f"(batch={BATCH_SIZE}, lr={LEARNING_RATE}, patch={PATCH_SIZE})\n")

    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── Train ──────────────────────────────────────────────────────────
        avg_loss = train_one_epoch(model, train_loader, optimizer,
                                   criterion, device, epoch)

        # ── Validate ───────────────────────────────────────────────────────
        avg_psnr = validate(model, val_loader, device)

        # ── Log epoch summary ──────────────────────────────────────────────
        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS - 1} | "
              f"Train Loss: {avg_loss:.6f} | "
              f"Val PSNR: {avg_psnr:.2f} dB | "
              f"Best: {best_psnr:.2f} dB"
              + ("  ← new best" if is_best else ""))

        # ── Save checkpoint ────────────────────────────────────────────────
        save_checkpoint(model, optimizer, epoch, avg_psnr, is_best, CKPT_DIR)
        print()

    print("=" * 60)
    print(f"Training complete!  Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Best model saved to: {CKPT_DIR / 'best.pth'}")
    print("Compare this PSNR to the bicubic baseline: python scripts/baseline.py")


if __name__ == "__main__":
    # The  if __name__ == "__main__":  guard is required on Windows when using
    # DataLoader with num_workers > 0. It's good practice to always include it.
    main()
