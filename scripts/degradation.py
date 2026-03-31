"""
degradation.py
--------------
Stage 2: realistic on-the-fly degradation pipeline.

Instead of loading pre-saved LR images, we generate LR images LIVE during
training by degrading HR images with randomly sampled parameters.

Why is this better than pre-saved LR images?
  Pre-saved: the model sees the exact same 800 degraded images every epoch.
             It can start to "memorise" them rather than actually learning.
  On-the-fly: every image gets different blur strength, noise level, and
               JPEG quality each epoch → effectively unlimited training data.

The degradation pipeline (applied in order):
  HR image
    → 1. Gaussian blur      (simulates lens blur or camera shake)
    → 2. Bicubic downsample (makes the image LR — this is the core step)
    → 3. Gaussian noise     (simulates sensor noise)
    → 4. JPEG compression   (simulates codec artefacts)
    → LR image

Each step's strength is randomly sampled from a range each time.
You can also skip noise or JPEG by setting their probability to 0.

This is a simplified single-order version.
Real-ESRGAN uses second-order (the full pipeline is applied twice).
That will be added in Stage 3.
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter

# ── Parameter ranges ──────────────────────────────────────────────────────────
# These define the min/max for each degradation step.
# The actual value is randomly chosen within the range for each image.

BLUR_SIGMA_MIN  = 0.2   # very light blur
BLUR_SIGMA_MAX  = 3.0   # noticeable blur (like a slightly out-of-focus shot)

# Interpolation methods used during downsampling
# Using a random method each time makes the model robust to different
# kinds of aliasing patterns, not just bicubic.
RESIZE_METHODS = [
    Image.BICUBIC,   # smooth — most common in SR research
    Image.BILINEAR,  # slightly sharper than bicubic
    Image.LANCZOS,   # sharpest, can cause slight ringing
    Image.NEAREST,   # blocky — rare but worth seeing occasionally
]

NOISE_SIGMA_MIN   = 0     # no noise
NOISE_SIGMA_MAX   = 25    # noticeable but not extreme noise (0-255 scale)
NOISE_PROBABILITY = 0.8   # 80% of images get noise; 20% are noise-free

JPEG_QUALITY_MIN   = 30   # heavy compression — lots of artefacts
JPEG_QUALITY_MAX   = 95   # nearly lossless
JPEG_PROBABILITY   = 0.8  # 80% of images get JPEG compression

# ──────────────────────────────────────────────────────────────────────────────


def sample_params() -> dict:
    """
    Randomly draws degradation parameters from the ranges above.

    Returns a dict that fully describes one degradation:
      {
        "blur_sigma"   : float   — how strong the blur is
        "resize_method": int     — which PIL interpolation to use
        "noise_sigma"  : float   — noise strength (0 = no noise)
        "jpeg_quality" : int     — JPEG compression quality (None = no JPEG)
      }

    Calling this once per image gives each image its own degradation fingerprint.
    """
    # Decide whether to apply noise and JPEG this time
    apply_noise = random.random() < NOISE_PROBABILITY
    apply_jpeg  = random.random() < JPEG_PROBABILITY

    return {
        "blur_sigma"   : random.uniform(BLUR_SIGMA_MIN, BLUR_SIGMA_MAX),
        "resize_method": random.choice(RESIZE_METHODS),
        "noise_sigma"  : random.uniform(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX)
                         if apply_noise else 0.0,
        "jpeg_quality" : random.randint(JPEG_QUALITY_MIN, JPEG_QUALITY_MAX)
                         if apply_jpeg else None,
    }


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """
    Blurs the image using a Gaussian filter with the given sigma.

    sigma = 0.2 : barely noticeable (like a perfectly sharp image)
    sigma = 1.0 : softens fine details
    sigma = 3.0 : noticeably blurry (like an out-of-focus photo)

    We blur BEFORE downsampling because that's how real cameras work:
    the lens blurs the scene, then the sensor samples it at lower resolution.
    """
    if sigma <= 0.1:
        return image  # skip if sigma is negligibly small
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(image: Image.Image, scale: int,
                 method: int = Image.BICUBIC) -> Image.Image:
    """
    Downsamples the image by `scale` using the given PIL interpolation method.

    This is the core step that makes the image "low-resolution".
    A 480x320 image with scale=4 becomes 120x80.

    Different methods create different aliasing patterns:
      BICUBIC  → smooth, standard for SR research
      BILINEAR → slightly less smooth
      LANCZOS  → very sharp, slight ringing at edges
      NEAREST  → pixelated/blocky
    """
    new_w = image.width  // scale
    new_h = image.height // scale
    return image.resize((new_w, new_h), resample=method)


def apply_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """
    Adds random Gaussian noise to the image.

    Gaussian noise = each pixel gets a small random offset drawn from
    a normal distribution with standard deviation `sigma`.

    sigma = 0   : no change
    sigma = 10  : light grain (like a slightly noisy camera)
    sigma = 25  : clearly visible noise (like a dark environment photo)

    We apply noise AFTER downsampling because sensor noise is introduced
    at the capture (LR) stage, not before.
    """
    if sigma <= 0:
        return image

    # Convert PIL image to numpy array for pixel arithmetic
    arr = np.array(image, dtype=np.float32)          # shape: (H, W, 3)

    # Generate noise with the same shape as the image
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)

    # Add noise and clip to valid 0-255 range
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """
    Simulates JPEG compression artefacts using an in-memory round-trip.

    Steps:
      1. Encode to JPEG bytes in RAM (no temp file written)
      2. Decode back into a PIL image

    quality = 95 : near-lossless, minimal artefacts
    quality = 75 : standard web quality, visible 8×8 block pattern in smooth areas
    quality = 30 : heavy compression, severe blocking artefacts

    io.BytesIO is an in-memory buffer — it behaves like a file but lives in RAM.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def degrade(hr_image: Image.Image, scale: int,
            params: dict = None) -> Image.Image:
    """
    Applies the full degradation pipeline to produce an LR image from HR.

    hr_image : PIL Image — the clean high-resolution source
    scale    : int       — how many times smaller the LR image will be
    params   : dict      — degradation parameters (from sample_params()).
                           If None, parameters are randomly sampled.

    Returns a PIL Image at (hr_width // scale) × (hr_height // scale).

    Pipeline order:
      blur → downsample → noise → jpeg

    Why this order?
      Blur before downsample: matches real lens physics
      Noise after downsample: matches real sensor physics (noise is per-pixel
                               at the capture resolution, not at HR)
      JPEG last: compression is the final step in a camera's image pipeline
    """
    if params is None:
        params = sample_params()

    image = hr_image.copy()

    # Step 1 — Gaussian blur (at HR resolution)
    image = apply_gaussian_blur(image, sigma=params["blur_sigma"])

    # Step 2 — Downsample to LR size
    image = apply_resize(image, scale=scale, method=params["resize_method"])

    # Step 3 — Gaussian noise (at LR resolution)
    image = apply_gaussian_noise(image, sigma=params["noise_sigma"])

    # Step 4 — JPEG compression (at LR resolution, if enabled)
    if params["jpeg_quality"] is not None:
        image = apply_jpeg_compression(image, quality=params["jpeg_quality"])

    return image


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run: python scripts/degradation.py

if __name__ == "__main__":
    from pathlib import Path

    # Create a fake 480x320 RGB image (solid colour, good enough to test shapes)
    fake_hr = Image.new("RGB", (480, 320), color=(128, 64, 200))
    scale = 4

    print("Testing degradation pipeline...")

    params = sample_params()
    print(f"\nSampled params:")
    print(f"  blur_sigma   : {params['blur_sigma']:.3f}")
    print(f"  resize_method: {params['resize_method']}")
    print(f"  noise_sigma  : {params['noise_sigma']:.1f}")
    print(f"  jpeg_quality : {params['jpeg_quality']}")

    lr = degrade(fake_hr, scale=scale, params=params)

    expected_w = fake_hr.width  // scale
    expected_h = fake_hr.height // scale
    assert lr.size == (expected_w, expected_h), \
        f"Expected {(expected_w, expected_h)}, got {lr.size}"

    print(f"\nHR size : {fake_hr.size}")
    print(f"LR size : {lr.size}  (expected {expected_w}x{expected_h})")
    print("Size check passed.")

    # Test that calling degrade multiple times gives different LR images
    lr_a = degrade(fake_hr, scale=scale)
    lr_b = degrade(fake_hr, scale=scale)
    arr_a = np.array(lr_a)
    arr_b = np.array(lr_b)
    assert not np.array_equal(arr_a, arr_b), \
        "Two random degradations should differ — something is wrong."
    print("Randomness check passed (two runs give different results).")

    # Optionally save a visual comparison if HR images are available
    base = Path(__file__).parent.parent / "data" / "DIV2K" / "HR_valid"
    sample_imgs = list(base.glob("*.png"))[:1] + list(base.glob("*.jpg"))[:1]

    if sample_imgs:
        hr_real = Image.open(sample_imgs[0]).convert("RGB")
        lr_real = degrade(hr_real, scale=scale)
        out_dir = Path(__file__).parent.parent / "outputs" / "degradation_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        hr_real.save(out_dir / "hr_sample.png")
        lr_real.save(out_dir / "lr_degraded_sample.png")
        print(f"\nSaved visual sample to: {out_dir}")
    else:
        print("\n(No HR_val images found for visual test — that's fine.)")

    print("\ndegradation.py is working correctly.")
