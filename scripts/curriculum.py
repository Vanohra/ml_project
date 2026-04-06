"""
curriculum.py
-------------
Curriculum training: progressively increases degradation difficulty over epochs.

Instead of training on the full range of blur/noise/JPEG severity from epoch 1,
curriculum training starts with easy (mild) degradations and gradually introduces
harder ones as training progresses.  This helps the model:

  1. Learn the SR task on clean, easy examples first (stable early training)
  2. Generalise to harder degradations once it has a strong base (better PSNR)

How it works:
  The config defines a list of curriculum stages, each with a start_epoch and
  parameter ranges.  Before each training epoch, get_active_stage() selects the
  stage whose start_epoch is the highest value still <= the current epoch.

  Example with 3 stages over 100 epochs:
    epochs 1–24  → easy   (blur 0.2–1.0, noise 0–8,  JPEG 80–95)
    epochs 25–49 → medium (blur 0.3–2.5, noise 5–25,  JPEG 50–85)
    epochs 50+   → hard   (blur 0.5–4.0, noise 10–45, JPEG 20–70)

Relationship to domain_degradation.py:
  sample_curriculum_params() returns the same dict structure as
  domain_degradation.sample_domain_params().  The two are interchangeable —
  dataset.py uses curriculum sampling when curriculum_stage is set, and
  domain-preset sampling otherwise.  No changes to domain_degradation.py needed.

Validation:
  Curriculum only affects the TRAINING dataset.  The validation dataset always
  uses the full domain preset (pre-sampled at construction time) so PSNR numbers
  remain comparable across epochs and across different curriculum configs.
"""

import random
from PIL import Image

# Resize methods used when sampling — same 4 options available in all stages.
# Domain-specific resize preferences (e.g. surveillance uses NEAREST more) are
# intentionally not carried over — curriculum is domain-agnostic for resize.
_RESIZE_METHODS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.NEAREST]


# ── Stage selection ────────────────────────────────────────────────────────────

def get_active_stage(epoch: int, curriculum_cfg: dict) -> dict:
    """
    Returns the curriculum stage dict that should be active for the given epoch.

    Stages must be listed in the config in ascending start_epoch order.
    The function returns the last stage whose start_epoch <= epoch, so the model
    always starts at the first stage and advances automatically.

    Arguments
    ---------
    epoch          : current training epoch (1-indexed)
    curriculum_cfg : the "curriculum" dict from the YAML config

    Returns
    -------
    dict — one stage entry from curriculum_cfg["stages"]

    Raises ValueError if stages list is empty.

    Example
    -------
    stages = [{start_epoch:1, name:"easy"}, {start_epoch:25, name:"medium"}]
    get_active_stage(10, cfg)  →  easy
    get_active_stage(25, cfg)  →  medium
    get_active_stage(30, cfg)  →  medium
    """
    stages = curriculum_cfg.get("stages", [])
    if not stages:
        raise ValueError(
            "curriculum.stages is empty — define at least one stage in the config."
        )

    active = stages[0]   # fallback: always use the first stage
    for stage in stages:
        if epoch >= stage["start_epoch"]:
            active = stage
        # Stages must be sorted by start_epoch; stop early for efficiency
        # (but correctness does not depend on this — last match wins)
    return active


# ── Parameter sampling ─────────────────────────────────────────────────────────

def sample_curriculum_params(domain: str, stage: dict) -> dict:
    """
    Samples degradation parameters from a curriculum stage's ranges.

    Returns the same dict structure as domain_degradation.sample_domain_params()
    so the two are interchangeable in dataset.py's __getitem__ logic.

    Arguments
    ---------
    domain : str  — domain name ("mobile", "surveillance", "dashcam")
                    Used only to populate the "domain" key in the returned dict.
                    (The conditioning vector still encodes the correct domain.)
    stage  : dict — one stage entry from curriculum_cfg["stages"]

    Stage dict keys
    ---------------
    Required:
      blur_sigma               : [min, max]  — Stage 1 Gaussian blur range
      noise_sigma              : [min, max]  — Stage 1 noise range
      jpeg_quality             : [min, max]  — Stage 1 JPEG quality range

    Optional (sensible defaults shown):
      noise_probability        : 0.8   — fraction of images that get noise
      jpeg_probability         : 0.8   — fraction of images that get JPEG
      second_stage_probability : 0.0   — probability of applying a second pass
      second_blur_sigma        : [0.1, 0.5]
      second_noise_sigma       : [0, 10]
      second_noise_probability : 0.5
      second_jpeg_quality      : [50, 85]
      second_jpeg_probability  : 0.75

    Returns
    -------
    dict with keys "domain", "stage1", "stage2"
    (same structure as domain_degradation.sample_domain_params())
    """
    # ── Stage 1 ───────────────────────────────────────────────────────────────
    blur_lo, blur_hi   = stage["blur_sigma"]
    noise_lo, noise_hi = stage["noise_sigma"]
    jpeg_lo,  jpeg_hi  = stage["jpeg_quality"]

    apply_noise = random.random() < stage.get("noise_probability", 0.8)
    apply_jpeg  = random.random() < stage.get("jpeg_probability", 0.8)

    stage1 = {
        "blur_sigma":    random.uniform(blur_lo, blur_hi),
        "resize_method": random.choice(_RESIZE_METHODS),
        "noise_sigma":   random.uniform(noise_lo, noise_hi) if apply_noise else 0.0,
        "jpeg_quality":  random.randint(int(jpeg_lo), int(jpeg_hi)) if apply_jpeg else None,
    }

    # ── Stage 2 (optional second degradation pass) ────────────────────────────
    stage2 = None
    s2_prob = stage.get("second_stage_probability", 0.0)
    if s2_prob > 0 and random.random() < s2_prob:
        s2_blur  = stage.get("second_blur_sigma",        [0.1, 0.5])
        s2_noise = stage.get("second_noise_sigma",        [0, 10])
        s2_jpeg  = stage.get("second_jpeg_quality",       [50, 85])
        s2_n_p   = stage.get("second_noise_probability",  0.5)
        s2_j_p   = stage.get("second_jpeg_probability",   0.75)

        apply_noise2 = random.random() < s2_n_p
        apply_jpeg2  = random.random() < s2_j_p

        stage2 = {
            "blur_sigma":   random.uniform(*s2_blur),
            "noise_sigma":  random.uniform(*s2_noise) if apply_noise2 else 0.0,
            "jpeg_quality": random.randint(int(s2_jpeg[0]), int(s2_jpeg[1]))
                            if apply_jpeg2 else None,
        }

    return {
        "domain": domain,
        "stage1": stage1,
        "stage2": stage2,
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
# Run: python scripts/curriculum.py

if __name__ == "__main__":
    print("curriculum.py self-test")
    print("=" * 50)

    # Minimal curriculum config for testing
    test_cfg = {
        "enabled": True,
        "stages": [
            {
                "name": "easy", "start_epoch": 1,
                "blur_sigma": [0.2, 1.0], "noise_sigma": [0, 8], "jpeg_quality": [80, 95],
                "noise_probability": 0.5, "jpeg_probability": 0.7,
                "second_stage_probability": 0.0,
            },
            {
                "name": "medium", "start_epoch": 25,
                "blur_sigma": [0.3, 2.5], "noise_sigma": [5, 25], "jpeg_quality": [50, 85],
                "noise_probability": 0.75, "jpeg_probability": 0.85,
                "second_stage_probability": 0.30,
            },
            {
                "name": "hard", "start_epoch": 50,
                "blur_sigma": [0.5, 4.0], "noise_sigma": [10, 45], "jpeg_quality": [20, 70],
                "noise_probability": 0.90, "jpeg_probability": 0.95,
                "second_stage_probability": 0.70,
            },
        ],
    }

    # ── Stage selection ────────────────────────────────────────────────────────
    cases = [(1, "easy"), (24, "easy"), (25, "medium"), (49, "medium"),
             (50, "hard"), (100, "hard")]
    print("\nStage selection:")
    for epoch, expected in cases:
        stage = get_active_stage(epoch, test_cfg)
        assert stage["name"] == expected, \
            f"epoch={epoch}: expected '{expected}', got '{stage['name']}'"
        print(f"  epoch {epoch:3d}  ->  {stage['name']}  OK")

    # ── Parameter sampling ─────────────────────────────────────────────────────
    print("\nParameter sampling (hard stage):")
    hard = test_cfg["stages"][2]
    for _ in range(5):
        params = sample_curriculum_params("surveillance", hard)
        s1 = params["stage1"]
        blur_ok  = 0.5 <= s1["blur_sigma"] <= 4.0
        noise_ok = s1["noise_sigma"] == 0.0 or 10 <= s1["noise_sigma"] <= 45
        assert blur_ok,  f"blur out of range: {s1['blur_sigma']}"
        assert noise_ok, f"noise out of range: {s1['noise_sigma']}"
        assert params["domain"] == "surveillance"
        s2_info = "stage2=yes" if params["stage2"] is not None else "stage2=no"
        print(f"  blur={s1['blur_sigma']:.2f}  noise={s1['noise_sigma']:.1f}  "
              f"jpeg={s1['jpeg_quality']}  {s2_info}")

    # ── Verify structure matches sample_domain_params output ───────────────────
    params = sample_curriculum_params("mobile", test_cfg["stages"][0])
    assert "domain"  in params
    assert "stage1"  in params
    assert "stage2"  in params
    assert "blur_sigma"    in params["stage1"]
    assert "resize_method" in params["stage1"]
    assert "noise_sigma"   in params["stage1"]
    assert "jpeg_quality"  in params["stage1"]
    print("\nOutput structure matches sample_domain_params()  OK")

    print("\ncurriculum.py is working correctly.")
