"""
domain_degradation.py
---------------------
Stage 3: domain-conditioned, high-order degradation pipeline.

Instead of a single generic degradation setting, this module defines
presets that match the real-world degradations of specific capture contexts:

  mobile      — smartphone photos, typically shared through social media
  surveillance — CCTV / IP camera footage at distance, often low-light
  dashcam     — vehicle-mounted camera with motion blur and video codec artefacts

What "high-order" means here:
  The standard pipeline (Stage 2) applies one sequence of operations:
    blur → downsample → noise → jpeg
  High-order adds an optional second pass AFTER downsampling:
    → additional blur → additional noise → additional jpeg
  This second pass simulates re-encoding steps that images commonly go through:
  a surveillance stream re-encoded for DVR storage, a dashcam clip re-encoded
  by YouTube, a phone photo recompressed by WhatsApp before delivery.

Relationship to degradation.py:
  The four low-level operations (apply_gaussian_blur, apply_resize,
  apply_gaussian_noise, apply_jpeg_compression) live in degradation.py
  and are imported here unchanged. No code in degradation.py is modified.

Basic usage:
  from domain_degradation import degrade_domain

  lr_image, metadata = degrade_domain(hr_image, scale=4, domain="surveillance")
  # lr_image  — degraded PIL Image at 1/scale resolution
  # metadata  — dict with the exact parameters used (stage1 and stage2)

To plug into the training loop (replacing the generic degrade() call):
  deg_fn = lambda img: degrade_domain(img, scale=4, domain="surveillance")[0]

Reference:
  High-order degradation pipeline concept from:
  Wang et al. (2021) — Real-ESRGAN: Training Real-World Blind Super-Resolution
  with Pure Synthetic Data.  https://arxiv.org/abs/2107.10833
"""

import random
import sys
from pathlib import Path

from PIL import Image

# Both files are in scripts/ — this ensures the import works when running
# this file directly OR when it is imported by another script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from degradation import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize,
)


# ── Domain presets ─────────────────────────────────────────────────────────────
#
# Each key is a domain name. Each value is a dict of parameter ranges.
#
# Range format:
#   (min, max) tuples for continuous values → sampled with random.uniform / random.randint
#   list       for categorical values       → sampled with random.choice
#   float      for probabilities            → compared against random.random()
#
# To tune a domain, edit the numbers here — nowhere else needs to change.
# To add a new domain, add a new entry with the same set of keys.
#
# Key explanations:
#   blur_sigma       : Gaussian blur strength. 0.2 = barely visible; 4.0 = very soft.
#   resize_methods   : PIL resamplers to randomly choose from when downsampling.
#   noise_sigma      : Gaussian noise std on a 0–255 scale. 0–10 subtle; 30+ severe.
#   noise_probability: Fraction of images that receive noise at all.
#   jpeg_quality     : JPEG quality. 20 = heavy blocking; 95 = near-lossless.
#   jpeg_probability : Fraction of images that receive JPEG compression.
#
# Second-stage keys (high-order degradation — applied at LR resolution, no resize):
#   second_stage_probability : Fraction of images that go through a second pass.
#   second_blur_sigma        : Extra blur range for the second pass.
#   second_noise_sigma       : Extra noise range for the second pass.
#   second_noise_probability : Fraction of second-pass images that get extra noise.
#   second_jpeg_quality      : JPEG quality range for the second pass.
#   second_jpeg_probability  : Fraction of second-pass images that get re-compressed.

DOMAIN_PRESETS = {

    # ── Mobile ────────────────────────────────────────────────────────────────
    # Source: modern smartphone (iPhone, Pixel, Galaxy), shared via messaging/social.
    #
    # Why these values?
    #   • Modern phone lenses are sharp → light blur only (0.2–1.5)
    #   • BSI sensors perform well → low-moderate noise (0–15)
    #   • Good ISP chips → prefer smooth resamplers (no NEAREST)
    #   • Social apps recompress aggressively: WhatsApp caps at ~100KB,
    #     Instagram strips EXIF and re-encodes at ~80 quality → JPEG 60–95
    #   • Second stage: platform re-compression is very common (0.5 probability)
    #     and usually mild (JPEG 50–85, barely any extra noise)
    "mobile": {
        "blur_sigma":               (0.2, 1.5),
        "resize_methods":           [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS],
        "noise_sigma":              (0, 15),
        "noise_probability":        0.70,
        "jpeg_quality":             (60, 95),
        "jpeg_probability":         0.85,
        # Second stage: social media platform re-compression
        "second_stage_probability": 0.50,
        "second_blur_sigma":        (0.1, 0.5),    # almost no extra blur
        "second_noise_sigma":       (0, 5),
        "second_noise_probability": 0.30,
        "second_jpeg_quality":      (50, 85),
        "second_jpeg_probability":  0.80,
    },

    # ── Surveillance ──────────────────────────────────────────────────────────
    # Source: fixed CCTV or IP camera, subject often 5–20m away, often at night.
    #
    # Why these values?
    #   • Cheap wide-angle lenses, subject at distance → heavy blur (1.0–4.0)
    #   • Small sensors, low-light environments → heavy noise (10–45, 95% of frames)
    #   • Bandwidth-limited streaming: many systems encode at 200–500 kbps
    #     which is extreme for video → JPEG quality 20–60 is realistic
    #   • Cheap encoding chips use any resampler including NEAREST
    #   • Second stage: DVR systems re-encode for long-term storage, cloud
    #     services re-encode for remote viewing → 75% probability, aggressive
    "surveillance": {
        "blur_sigma":               (1.0, 4.0),
        "resize_methods":           [Image.BICUBIC, Image.BILINEAR,
                                     Image.LANCZOS, Image.NEAREST],
        "noise_sigma":              (10, 45),
        "noise_probability":        0.95,
        "jpeg_quality":             (20, 60),
        "jpeg_probability":         0.95,
        # Second stage: DVR / cloud streaming re-encoding
        "second_stage_probability": 0.75,
        "second_blur_sigma":        (0.2, 1.0),    # de-blocking filter smoothing
        "second_noise_sigma":       (5, 20),
        "second_noise_probability": 0.70,
        "second_jpeg_quality":      (20, 50),       # DVR storage compression
        "second_jpeg_probability":  0.90,
    },

    # ── Dashcam ───────────────────────────────────────────────────────────────
    # Source: vehicle-mounted camera (Nextbase, Vantrue, BlackVue) at 60–120 km/h.
    #
    # Why these values?
    #   • Wide-angle + windshield distortion + vibration → moderate blur (0.5–2.5)
    #     (true motion blur is directional, but Gaussian is a reasonable proxy)
    #   • Consumer sensors, often set to 1080p30 or 1080p60 → noise 5–30
    #   • Fixed-bitrate H.264/H.265 recording at 12–20 Mbps creates 8×8 block
    #     artifacts in motion areas — JPEG 30–75 approximates this well
    #   • All resamplers: cheap consumer chips vary widely
    #   • Second stage: uploading to YouTube or insurance portals re-encodes at
    #     lower bitrate → 60% probability, JPEG 30–65 for the re-encode pass
    "dashcam": {
        "blur_sigma":               (0.5, 2.5),
        "resize_methods":           [Image.BICUBIC, Image.BILINEAR,
                                     Image.LANCZOS, Image.NEAREST],
        "noise_sigma":              (5, 30),
        "noise_probability":        0.85,
        "jpeg_quality":             (30, 75),
        "jpeg_probability":         0.90,
        # Second stage: platform re-encoding (YouTube, insurance portal, cloud DVR)
        "second_stage_probability": 0.60,
        "second_blur_sigma":        (0.1, 0.8),
        "second_noise_sigma":       (0, 10),
        "second_noise_probability": 0.50,
        "second_jpeg_quality":      (30, 65),
        "second_jpeg_probability":  0.85,
    },
}

SUPPORTED_DOMAINS = list(DOMAIN_PRESETS.keys())   # ["mobile", "surveillance", "dashcam"]


# ── Sampling ───────────────────────────────────────────────────────────────────

def sample_domain_params(domain: str) -> dict:
    """
    Randomly samples degradation parameters from the given domain's preset ranges.

    Arguments:
      domain : str — one of "mobile", "surveillance", "dashcam"

    Returns a dict with the structure:
      {
        "domain": "surveillance",
        "stage1": {
          "blur_sigma":    2.73,
          "resize_method": <PIL resampler>,
          "noise_sigma":   28.4,
          "jpeg_quality":  35,        # or None if JPEG skipped
        },
        "stage2": {                   # or None if second stage skipped
          "blur_sigma":    0.45,
          "noise_sigma":   12.1,
          "jpeg_quality":  30,        # or None if JPEG skipped
        }
      }

    Calling this once per training image gives each image its own unique
    degradation fingerprint that is reproducible if you save the dict.
    """
    if domain not in DOMAIN_PRESETS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Supported domains: {SUPPORTED_DOMAINS}"
        )

    p = DOMAIN_PRESETS[domain]

    # ── Stage 1 parameters ────────────────────────────────────────────────────
    apply_noise = random.random() < p["noise_probability"]
    apply_jpeg  = random.random() < p["jpeg_probability"]

    stage1 = {
        "blur_sigma":    random.uniform(*p["blur_sigma"]),
        "resize_method": random.choice(p["resize_methods"]),
        "noise_sigma":   random.uniform(*p["noise_sigma"]) if apply_noise else 0.0,
        "jpeg_quality":  random.randint(*p["jpeg_quality"]) if apply_jpeg else None,
    }

    # ── Stage 2 parameters ────────────────────────────────────────────────────
    stage2 = None
    if random.random() < p["second_stage_probability"]:
        apply_noise2 = random.random() < p["second_noise_probability"]
        apply_jpeg2  = random.random() < p["second_jpeg_probability"]

        stage2 = {
            "blur_sigma":   random.uniform(*p["second_blur_sigma"]),
            "noise_sigma":  random.uniform(*p["second_noise_sigma"]) if apply_noise2 else 0.0,
            "jpeg_quality": random.randint(*p["second_jpeg_quality"]) if apply_jpeg2 else None,
        }

    return {
        "domain": domain,
        "stage1": stage1,
        "stage2": stage2,
    }


# ── Core function ─────────────────────────────────────────────────────────────

def degrade_domain(
    hr_image: Image.Image,
    scale: int,
    domain: str,
    params: dict = None,
) -> tuple[Image.Image, dict]:
    """
    Applies domain-conditioned, high-order degradation to produce an LR image.

    Arguments:
      hr_image : PIL Image — clean high-resolution source
      scale    : int       — downscale factor (4 means output is 4× smaller)
      domain   : str       — one of "mobile", "surveillance", "dashcam"
      params   : dict      — pre-sampled parameters from sample_domain_params().
                             If None, parameters are sampled automatically.

    Returns:
      (lr_image, metadata)
        lr_image : PIL Image — degraded image at (width//scale) × (height//scale)
        metadata : dict      — the exact parameters used (pass back to this
                               function to reproduce the exact same degradation)

    Pipeline — Stage 1 (always applied):
      1. Gaussian blur         at HR resolution  (simulates lens / optical blur)
      2. Downsample by scale                     (the core LR step)
      3. Gaussian noise        at LR resolution  (simulates sensor noise)
      4. JPEG compression      at LR resolution  (simulates codec artefacts)

    Pipeline — Stage 2 (applied with domain-specific probability):
      5. Additional Gaussian blur  at LR resolution  (re-encode smoothing)
      6. Additional Gaussian noise at LR resolution  (transmission noise)
      7. Additional JPEG           at LR resolution  (DVR / platform re-encode)

    Why no second downsample in Stage 2?
      Our model is a fixed 4× SR model. Adding a second downsample would change
      the target scale factor and break the training setup. The second stage is
      purely about compression / encoding artefacts applied to the already-LR image.
    """
    if params is None:
        params = sample_domain_params(domain)

    image = hr_image.copy()
    s1 = params["stage1"]

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    # Step 1 — blur at HR resolution (lens softness, camera shake, distance)
    image = apply_gaussian_blur(image, sigma=s1["blur_sigma"])

    # Step 2 — downsample to LR size (the defining step for super-resolution)
    image = apply_resize(image, scale=scale, method=s1["resize_method"])

    # Step 3 — noise at LR resolution (sensor noise is per-capture-pixel)
    image = apply_gaussian_noise(image, sigma=s1["noise_sigma"])

    # Step 4 — JPEG / codec compression at LR resolution
    if s1["jpeg_quality"] is not None:
        image = apply_jpeg_compression(image, quality=s1["jpeg_quality"])

    # ── Stage 2 (optional re-degradation at LR resolution) ───────────────────

    if params["stage2"] is not None:
        s2 = params["stage2"]

        # Extra blur — de-blocking filter smoothing or motion from re-capture
        image = apply_gaussian_blur(image, sigma=s2["blur_sigma"])

        # Extra noise — transmission or re-encoding noise
        image = apply_gaussian_noise(image, sigma=s2["noise_sigma"])

        # Re-compression — DVR, social platform, streaming service
        if s2["jpeg_quality"] is not None:
            image = apply_jpeg_compression(image, quality=s2["jpeg_quality"])

    return image, params


# ── Self-test ─────────────────────────────────────────────────────────────────
# Run:  python scripts/domain_degradation.py
# from the ml_project/ directory.

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("  Domain Degradation Self-Test")
    print("=" * 60)

    SCALE = 4
    out_dir = Path(__file__).parent.parent / "outputs" / "domain_degradation_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load a real HR image if available ─────────────────────────────────────
    hr_valid_dir = Path(__file__).parent.parent / "data" / "DIV2K" / "HR_valid"
    candidates = sorted(hr_valid_dir.glob("*.png"))[:1] + \
                 sorted(hr_valid_dir.glob("*.jpg"))[:1]

    if candidates:
        hr_image = Image.open(candidates[0]).convert("RGB")
        print(f"\nSource HR image : {candidates[0].name}  "
              f"({hr_image.width}×{hr_image.height})")
    else:
        # Fall back to a synthetic image so the test still runs without data
        hr_image = Image.new("RGB", (480, 320), color=(100, 149, 237))
        print("\nNo HR images found — using a synthetic 480×320 image.")
        print("(Place HR images in data/DIV2K/HR_valid/ for a real visual test.)")

    expected_lr_size = (hr_image.width // SCALE, hr_image.height // SCALE)
    hr_image.save(out_dir / "hr_original.png")

    # ── Run one degradation per domain ────────────────────────────────────────
    for domain in SUPPORTED_DOMAINS:
        print(f"\n{'─' * 50}")
        print(f"  Domain: {domain.upper()}")
        print(f"{'─' * 50}")

        lr_image, metadata = degrade_domain(hr_image, scale=SCALE, domain=domain)

        # Print the sampled parameters so you can see what was applied
        s1 = metadata["stage1"]
        print(f"  Stage 1  (blur → downsample → noise → jpeg):")
        print(f"    blur_sigma   : {s1['blur_sigma']:.3f}")
        print(f"    resize_method: {s1['resize_method']}")
        print(f"    noise_sigma  : {s1['noise_sigma']:.1f}")
        print(f"    jpeg_quality : {s1['jpeg_quality']}")

        if metadata["stage2"] is not None:
            s2 = metadata["stage2"]
            print(f"  Stage 2  (re-degradation at LR resolution):")
            print(f"    blur_sigma   : {s2['blur_sigma']:.3f}")
            print(f"    noise_sigma  : {s2['noise_sigma']:.1f}")
            print(f"    jpeg_quality : {s2['jpeg_quality']}")
        else:
            print(f"  Stage 2: skipped this time")

        # Verify output size is exactly (width//scale, height//scale)
        assert lr_image.size == expected_lr_size, (
            f"Size mismatch: expected {expected_lr_size}, got {lr_image.size}"
        )
        print(f"  Output LR size: {lr_image.size}  ✓")

        # Save individual LR result
        lr_image.save(out_dir / f"lr_{domain}.png")

    # Also save a plain bicubic LR for comparison
    bicubic_lr = hr_image.resize(expected_lr_size, Image.BICUBIC)
    bicubic_lr.save(out_dir / "lr_bicubic_reference.png")

    # ── Verify randomness ─────────────────────────────────────────────────────
    lr_a, _ = degrade_domain(hr_image, scale=SCALE, domain="mobile")
    lr_b, _ = degrade_domain(hr_image, scale=SCALE, domain="mobile")
    arr_a = np.array(lr_a)
    arr_b = np.array(lr_b)
    assert not np.array_equal(arr_a, arr_b), \
        "Two calls should produce different results — check random seeding."
    print(f"\n  Randomness check passed (two mobile runs differ).  ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  All tests passed.")
    print(f"  Outputs saved to: {out_dir}")
    print(f"\n  hr_original.png           — clean HR source")
    print(f"  lr_bicubic_reference.png  — plain bicubic downsample (no degradation)")
    for domain in SUPPORTED_DOMAINS:
        print(f"  lr_{domain}.png{'':>{18 - len(domain)}}— degraded LR ({domain} preset)")
    print("=" * 60)
