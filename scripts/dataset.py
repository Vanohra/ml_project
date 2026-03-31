"""
dataset.py
----------
Loads HR/LR image pairs for training and validation.

Supports two modes controlled by whether you pass lr_dir or degradation_fn:

  MODE A — Pre-saved LR (Stage 1)
    DIV2KDataset(hr_dir=..., lr_dir=..., patch_size=48)
    Loads matching HR and LR files from disk.
    Use this when you already ran prepare_data.py.

  MODE B — Online degradation (Stage 2+)
    DIV2KDataset(hr_dir=..., degradation_fn=degrade, patch_size=48)
    Loads HR only. Generates LR live by calling degradation_fn on each crop.
    Every epoch sees different random degradations → effectively unlimited data.

Think of it as a vending machine:
  Mode A: you pre-packaged the snacks (LR files on disk)
  Mode B: the machine makes each snack fresh every time you press the button
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class DIV2KDataset(Dataset):
    """
    Loads (LR, HR) image pairs for super-resolution training and validation.

    Arguments:
      hr_dir         : folder with high-resolution images (always required)
      lr_dir         : folder with pre-saved LR images (Mode A only)
      degradation_fn : callable that turns a HR PIL image into a LR PIL image
                       (Mode B only). Example: from degradation import degrade
      patch_size     : LR patch size for training crops. Set None for full images.
      scale          : upscale factor (default 4 — must match your pipeline)

    patch_size controls training vs validation behaviour:
      patch_size=48   → crops 48×48 LR and 192×192 HR patches for training
      patch_size=None → returns full images for validation
    """

    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, hr_dir, lr_dir=None, degradation_fn=None,
                 patch_size=None, scale=4):

        if lr_dir is None and degradation_fn is None:
            raise ValueError(
                "Provide either lr_dir (Mode A: pre-saved LR files) "
                "or degradation_fn (Mode B: online degradation)."
            )

        self.hr_dir         = Path(hr_dir)
        self.lr_dir         = Path(lr_dir) if lr_dir else None
        self.degradation_fn = degradation_fn
        self.patch_size     = patch_size
        self.scale          = scale

        # Collect all HR filenames, sorted so order is consistent across runs
        self.filenames = sorted([
            p.name for p in self.hr_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED
        ])

        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.hr_dir}")

    def __len__(self):
        return len(self.filenames)

    # ── Mode A helpers ────────────────────────────────────────────────────────

    def _find_lr_path(self, hr_filename: str) -> Path:
        """
        Locates the LR file matching an HR filename.

        prepare_data.py always saves LR as .png even if HR is .jpg,
        so we check both the exact name and a .png fallback.
        """
        exact = self.lr_dir / hr_filename
        if exact.exists():
            return exact

        fallback = self.lr_dir / (Path(hr_filename).stem + ".png")
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"No LR file found for {hr_filename} in {self.lr_dir}\n"
            "Run prepare_data.py first, or switch to online degradation mode."
        )

    def _paired_crop(self, lr_img: Image.Image,
                     hr_img: Image.Image):
        """
        (Mode A) Cuts matching crops from pre-aligned LR and HR images.

        The crop origin is picked in LR space, then multiplied by scale
        to get the matching position in HR space.
        """
        lr_w, lr_h = lr_img.size
        max_x = lr_w - self.patch_size
        max_y = lr_h - self.patch_size

        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"Image too small for patch_size={self.patch_size}. "
                f"LR size: {lr_img.size}"
            )

        lr_x = random.randint(0, max_x)
        lr_y = random.randint(0, max_y)
        hr_x, hr_y = lr_x * self.scale, lr_y * self.scale
        hr_p = self.patch_size * self.scale

        lr_crop = lr_img.crop((lr_x, lr_y,
                                lr_x + self.patch_size,
                                lr_y + self.patch_size))
        hr_crop = hr_img.crop((hr_x, hr_y,
                                hr_x + hr_p,
                                hr_y + hr_p))
        return lr_crop, hr_crop

    # ── Mode B helpers ────────────────────────────────────────────────────────

    def _hr_crop(self, hr_img: Image.Image) -> Image.Image:
        """
        (Mode B) Cuts a random HR-sized patch from the HR image.

        The patch size in HR pixels is patch_size * scale.
        Example: patch_size=48, scale=4 → crop 192×192 from HR.

        We crop from HR only (no pre-saved LR), then call degradation_fn
        to produce the matching LR patch.
        """
        hr_patch = self.patch_size * self.scale
        hr_w, hr_h = hr_img.size
        max_x = hr_w - hr_patch
        max_y = hr_h - hr_patch

        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"Image too small for patch_size={self.patch_size} × scale={self.scale}. "
                f"HR size: {hr_img.size}. Use a smaller patch_size."
            )

        hr_x = random.randint(0, max_x)
        hr_y = random.randint(0, max_y)
        return hr_img.crop((hr_x, hr_y, hr_x + hr_patch, hr_y + hr_patch))

    # ── Core data loading ─────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        """
        Returns one (lr_tensor, hr_tensor) pair.

        Mode A flow:
          load HR file + load LR file → crop matching patches → return tensors

        Mode B flow:
          load HR file → crop HR patch → degrade to make LR → return tensors

        Both modes return:
          lr_tensor : (3, patch_size, patch_size)          values in [0, 1]
          hr_tensor : (3, patch_size*scale, patch_size*scale) values in [0, 1]
          (or full image sizes if patch_size is None)
        """
        filename = self.filenames[index]
        hr_img = Image.open(self.hr_dir / filename).convert("RGB")

        if self.degradation_fn is not None:
            # ── Mode B: online degradation ────────────────────────────────
            if self.patch_size is not None:
                # Cut HR patch first, then degrade it to get LR
                hr_img = self._hr_crop(hr_img)

            lr_img = self.degradation_fn(hr_img)

        else:
            # ── Mode A: load pre-saved LR from disk ───────────────────────
            lr_img = Image.open(self._find_lr_path(filename)).convert("RGB")

            if self.patch_size is not None:
                lr_img, hr_img = self._paired_crop(lr_img, hr_img)

        # to_tensor: PIL image (H × W × C, 0-255) → tensor (C × H × W, 0.0-1.0)
        return to_tensor(lr_img), to_tensor(hr_img)


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run: python scripts/dataset.py

if __name__ == "__main__":
    from degradation import degrade

    base = Path(__file__).parent.parent / "data" / "DIV2K"

    # ── Test Mode A (pre-saved LR) ────────────────────────────────────────────
    print("Mode A — pre-saved LR files:")
    try:
        val_ds = DIV2KDataset(hr_dir=base / "HR_valid",
                              lr_dir=base / "LR_valid")
        lr, hr = val_ds[0]
        print(f"  Full image  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")

        train_ds = DIV2KDataset(hr_dir=base / "HR_train",
                                lr_dir=base / "LR_train",
                                patch_size=48, scale=4)
        lr, hr = train_ds[0]
        print(f"  Patch mode  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
        assert lr.shape == torch.Size([3, 48, 48])
        assert hr.shape == torch.Size([3, 192, 192])
        print("  Shape check passed.")
    except FileNotFoundError as e:
        print(f"  Skipped (LR files not found — run prepare_data.py first):\n  {e}")

    # ── Test Mode B (online degradation) ──────────────────────────────────────
    print("\nMode B — online degradation:")
    try:
        def deg_fn(img):
            return degrade(img, scale=4)

        train_ds_online = DIV2KDataset(
            hr_dir=base / "HR_train",
            degradation_fn=deg_fn,
            patch_size=48, scale=4,
        )
        print(f"  {len(train_ds_online)} HR image(s) found.")
        lr, hr = train_ds_online[0]
        print(f"  Patch mode  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
        assert lr.shape == torch.Size([3, 48, 48])
        assert hr.shape == torch.Size([3, 192, 192])
        print("  Shape check passed.")

        # Check that two calls to the same index give different LR
        import numpy as np
        lr_a, _ = train_ds_online[0]
        lr_b, _ = train_ds_online[0]
        assert not torch.equal(lr_a, lr_b), \
            "Two calls should produce different random degradations!"
        print("  Randomness check passed.")
    except FileNotFoundError as e:
        print(f"  Skipped (HR images not found): {e}")

    print("\ndataset.py is working correctly.")
