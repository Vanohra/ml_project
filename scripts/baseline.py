"""
baseline.py
-----------
What this script does:
  This is your FIRST "model" — but it doesn't actually learn anything.
  It takes an LR (low-resolution) image and simply blows it back up to HR size
  using BICUBIC interpolation (a standard image-resizing algorithm).

  This is called the BICUBIC BASELINE.
  It's the floor — any real super-resolution model should beat this.

Why do this first?
  - It proves your pipeline works end-to-end
  - It gives you a visual reference point: "my model must look better than this"
  - You can measure it with PSNR (a number that scores image quality)

PSNR = Peak Signal-to-Noise Ratio
  Higher is better. Typical bicubic baseline: ~28-30 dB on DIV2K.
  A good deep learning model might reach 32+ dB.

Output: saved to  outputs/bicubic_results/
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

# ── Settings ──────────────────────────────────────────────────────────────────
SCALE    = 4    # Must match what you used in prepare_data.py
MAX_IMGS = 10   # How many validation images to process (keep small for testing)

BASE_DIR    = Path(__file__).parent.parent
LR_VAL_DIR  = BASE_DIR / "data" / "DIV2K" / "LR_valid"
HR_VAL_DIR  = BASE_DIR / "data" / "DIV2K" / "HR_valid"
OUTPUT_DIR  = BASE_DIR / "outputs" / "bicubic_results"
# ──────────────────────────────────────────────────────────────────────────────


def calculate_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Calculates PSNR between two images (both as numpy arrays, values 0-255).

    PSNR formula:  10 * log10(255^2 / MSE)
    MSE = Mean Squared Error (average pixel difference squared)

    Returns infinity if the images are identical (perfect match).
    """
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0 ** 2 / mse)


def bicubic_upsample(lr_image: Image.Image, scale: int) -> Image.Image:
    """
    Enlarge lr_image by `scale` times using BICUBIC interpolation.
    This is PIL's built-in resize — no neural network involved.
    """
    new_width  = lr_image.width  * scale
    new_height = lr_image.height * scale
    return lr_image.resize((new_width, new_height), Image.BICUBIC)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find LR validation images
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    lr_paths = sorted([
        p for p in LR_VAL_DIR.iterdir()
        if p.suffix.lower() in supported
    ])

    if not lr_paths:
        print(f"[!] No LR images found in {LR_VAL_DIR}")
        print("    Run prepare_data.py first.")
        return

    lr_paths = lr_paths[:MAX_IMGS]  # limit to MAX_IMGS for speed
    psnr_scores = []

    print(f"Running bicubic baseline on {len(lr_paths)} image(s)...")
    print(f"Output folder: {OUTPUT_DIR}\n")

    for i, lr_path in enumerate(lr_paths, start=1):
        # 1. Load the LR image
        lr_image = Image.open(lr_path).convert("RGB")

        # 2. Upscale it with bicubic (our "model")
        sr_image = bicubic_upsample(lr_image, SCALE)

        # 3. Load the matching HR image (the ground truth)
        hr_path = HR_VAL_DIR / lr_path.name
        if not hr_path.exists():
            print(f"  [{i}] WARNING: No HR match for {lr_path.name}, skipping.")
            continue
        hr_image = Image.open(hr_path).convert("RGB")

        # 4. Crop SR to HR size (in case sizes are slightly off due to integer division)
        sr_image = sr_image.crop((0, 0, hr_image.width, hr_image.height))

        # 5. Calculate PSNR
        psnr = calculate_psnr(np.array(sr_image), np.array(hr_image))
        psnr_scores.append(psnr)

        # 6. Save the upscaled result
        out_path = OUTPUT_DIR / f"bicubic_{lr_path.name}"
        sr_image.save(out_path)

        print(f"  [{i}/{len(lr_paths)}] {lr_path.name}  "
              f"LR {lr_image.size} -> SR {sr_image.size}  "
              f"PSNR = {psnr:.2f} dB")

    # 7. Print average PSNR
    if psnr_scores:
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        print(f"\nAverage PSNR : {avg_psnr:.2f} dB  (higher is better)")
        print(f"This is your baseline score to beat with a real model.")
    else:
        print("\nNo scores calculated. Check that HR and LR filenames match.")

    print(f"\nSaved upscaled images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
