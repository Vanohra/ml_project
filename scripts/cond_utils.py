"""
cond_utils.py
-------------
Converts degradation metadata dicts into a fixed-length conditioning vector
for use by ConditionedSRResNet (scripts/models/conditioned_sr.py).

Both dataset.py (to return cond vectors from the DataLoader) and
conditioned_sr.py (to document the format) import from this file.

Conditioning vector layout   COND_DIM = 13
──────────────────────────────────────────────────────────────────────────────
 Index │ Name                  │ Normalization │ Notes
───────┼───────────────────────┼───────────────┼──────────────────────────────
   0   │ blur_sigma_s1         │ / 4.0         │ Stage 1 Gaussian blur strength
   1   │ resize_method_s1      │ / 3.0         │ PIL resampler mapped to index 0–3
   2   │ noise_sigma_s1        │ / 45.0        │ 0.0 = noise was not applied
   3   │ jpeg_quality_s1       │ / 95.0        │ 0.0 = JPEG was not applied
   4   │ jpeg_applied_s1       │ {0, 1}        │ explicit JPEG flag (avoids ambiguity)
   5   │ stage2_active         │ {0, 1}        │ 1 = second degradation stage ran
   6   │ blur_sigma_s2         │ / 1.0         │ 0 if stage 2 inactive
   7   │ noise_sigma_s2        │ / 20.0        │ 0 if stage 2 inactive
   8   │ jpeg_quality_s2       │ / 85.0        │ 0 if stage 2 JPEG skipped
   9   │ jpeg_applied_s2       │ {0, 1}        │ 0 if stage 2 inactive
  10   │ domain = mobile       │ {0, 1}        │ one-hot encoding over the 3 domains
  11   │ domain = surveillance │ {0, 1}        │
  12   │ domain = dashcam      │ {0, 1}        │
──────────────────────────────────────────────────────────────────────────────

Normalization constants are set so every element stays in [0, 1] given the
parameter ranges in DOMAIN_PRESETS (scripts/domain_degradation.py).

Why explicit flags for JPEG?
  jpeg_quality = None means JPEG was skipped entirely.
  Encoding None as 0.0 would be ambiguous — 0.0 could also mean quality=0
  (extreme compression).  The separate jpeg_applied flag removes this ambiguity.

Why a one-hot domain encoding?
  Domain identity is categorical.  Treating "surveillance" as numeric (e.g. 2)
  would imply it is "twice as much as mobile" (1), which is meaningless.
  One-hot gives the model a clean binary signal for each domain.
"""

import torch
from PIL import Image

# ── Public constants ──────────────────────────────────────────────────────────

COND_DIM = 13   # length of every conditioning vector

DOMAIN_ORDER = ["mobile", "surveillance", "dashcam"]   # one-hot order

# ── Normalization denominators ────────────────────────────────────────────────
# Chosen from the maximum values in DOMAIN_PRESETS so every element ∈ [0, 1].

_BLUR_S1_MAX   = 4.0    # surveillance max blur sigma
_NOISE_S1_MAX  = 45.0   # surveillance max noise sigma
_JPEG_S1_MAX   = 95.0   # mobile max JPEG quality
_BLUR_S2_MAX   = 1.0    # max second-stage blur across all domains
_NOISE_S2_MAX  = 20.0   # max second-stage noise (surveillance)
_JPEG_S2_MAX   = 85.0   # max second-stage JPEG quality (mobile)

# Map PIL resampler objects/ints to a 0–3 index for normalization.
# PIL enum values can differ across Pillow versions; explicit mapping avoids surprises.
_RESIZE_INDEX = {
    Image.NEAREST:  0,
    Image.BILINEAR: 1,
    Image.BICUBIC:  2,
    Image.LANCZOS:  3,
}
_RESIZE_MAX = 3.0

# Int version of _RESIZE_INDEX for loading from JSON.
# extract_virat_frames.py saves resize_method as int via serialisable_params()
# (PIL enum → int).  When build_cond_vector() receives that JSON-loaded dict,
# resize_method is an int whose value is the PIL Resampling enum's integer value
# (NEAREST=0, LANCZOS=1, BILINEAR=2, BICUBIC=3 in Pillow ≥ 9).
# We build this at module load time so it adapts to whatever Pillow version
# is installed rather than hard-coding constants.
_RESIZE_INDEX_INT: dict[int, int] = {int(k): v for k, v in _RESIZE_INDEX.items()}


# ── Main function ─────────────────────────────────────────────────────────────

def build_cond_vector(metadata: dict) -> "torch.Tensor":
    """
    Converts a degradation metadata dict into a normalized conditioning vector.

    Arguments
    ---------
    metadata : dict as returned by domain_degradation.sample_domain_params()
               Must have keys: "domain" (str), "stage1" (dict), "stage2" (dict or None)

    Returns
    -------
    torch.Tensor  shape (COND_DIM,) = (13,)  dtype float32  values in [0, 1]

    This tensor is DataLoader-safe (fixed shape, numeric dtype) and can be
    collated into batches of shape (batch_size, 13) automatically.
    """
    s1 = metadata["stage1"]
    s2 = metadata["stage2"]   # None if the second degradation stage was skipped

    # ── Stage 1 fields ────────────────────────────────────────────────────────
    blur_s1 = float(s1["blur_sigma"]) / _BLUR_S1_MAX
    # resize_method may be a PIL Resampling enum (from live degradation) or an int
    # (from JSON-loaded manifests where serialisable_params() converted it).
    _rm = s1["resize_method"]
    resize = float(
        _RESIZE_INDEX_INT.get(_rm, _RESIZE_INDEX.get(_rm, 2))
        if isinstance(_rm, int)
        else _RESIZE_INDEX.get(_rm, 2)
    ) / _RESIZE_MAX
    noise_s1 = float(s1["noise_sigma"]) / _NOISE_S1_MAX

    jq1 = s1["jpeg_quality"]                        # int or None
    jpeg_q_s1  = float(jq1) / _JPEG_S1_MAX if jq1 is not None else 0.0
    jpeg_f_s1  = 1.0 if jq1 is not None else 0.0   # applied flag

    # ── Stage 2 fields ────────────────────────────────────────────────────────
    s2_active = 1.0 if s2 is not None else 0.0

    if s2 is not None:
        blur_s2  = float(s2["blur_sigma"]) / _BLUR_S2_MAX
        noise_s2 = float(s2["noise_sigma"]) / _NOISE_S2_MAX
        jq2      = s2["jpeg_quality"]
        jpeg_q_s2 = float(jq2) / _JPEG_S2_MAX if jq2 is not None else 0.0
        jpeg_f_s2 = 1.0 if jq2 is not None else 0.0
    else:
        blur_s2 = noise_s2 = jpeg_q_s2 = jpeg_f_s2 = 0.0

    # ── Domain one-hot ────────────────────────────────────────────────────────
    domain     = metadata.get("domain", "")
    domain_vec = [1.0 if d == domain else 0.0 for d in DOMAIN_ORDER]

    # ── Assemble ──────────────────────────────────────────────────────────────
    vec = [
        blur_s1,   resize,    noise_s1,  jpeg_q_s1, jpeg_f_s1,  # indices 0–4
        s2_active,                                                 # index  5
        blur_s2,   noise_s2,  jpeg_q_s2, jpeg_f_s2,              # indices 6–9
        *domain_vec,                                               # indices 10–12
    ]

    assert len(vec) == COND_DIM, \
        f"[cond_utils] Vector length is {len(vec)}, expected {COND_DIM}"

    return torch.tensor(vec, dtype=torch.float32)


# ── Self-test ──────────────────────────────────────────────────────────────────
# Run: python scripts/cond_utils.py

if __name__ == "__main__":
    # Simulate a full metadata dict (surveillance domain, both stages active)
    from PIL import Image as _PIL

    meta_full = {
        "domain": "surveillance",
        "stage1": {
            "blur_sigma":    2.73,
            "resize_method": _PIL.NEAREST,
            "noise_sigma":   28.4,
            "jpeg_quality":  35,
        },
        "stage2": {
            "blur_sigma":   0.45,
            "noise_sigma":  12.1,
            "jpeg_quality": 30,
        },
    }

    # Simulate mobile domain, no stage 2, JPEG skipped
    meta_minimal = {
        "domain": "mobile",
        "stage1": {
            "blur_sigma":    0.5,
            "resize_method": _PIL.BICUBIC,
            "noise_sigma":   0.0,   # noise not applied
            "jpeg_quality":  None,  # JPEG not applied
        },
        "stage2": None,
    }

    v_full    = build_cond_vector(meta_full)
    v_minimal = build_cond_vector(meta_minimal)

    print(f"Vector length : {len(v_full)}  (expected {COND_DIM})")
    print(f"\nSurveillance (full):")
    for i, (name, val) in enumerate(zip([
        "blur_s1", "resize", "noise_s1", "jpeg_q_s1", "jpeg_f_s1",
        "stage2_active",
        "blur_s2", "noise_s2", "jpeg_q_s2", "jpeg_f_s2",
        "domain_mobile", "domain_surv", "domain_dash",
    ], v_full.tolist())):
        print(f"  [{i:2d}] {name:<20s} = {val:.4f}")

    print(f"\nMobile (minimal — noise+JPEG skipped, no stage 2):")
    for i, val in enumerate(v_minimal.tolist()):
        print(f"  [{i:2d}] = {val:.4f}")

    assert v_full.shape == (COND_DIM,), "Shape check failed"
    assert v_full.dtype == torch.float32, "Dtype check failed"
    assert 0.0 <= v_full.min() and v_full.max() <= 1.0, "Values out of [0,1]!"

    # Two different domains should produce different vectors
    meta_dash = dict(meta_full)
    meta_dash["domain"] = "dashcam"
    assert not torch.equal(build_cond_vector(meta_full), build_cond_vector(meta_dash)), \
        "Different domains should give different vectors"

    print("\nAll checks passed.")
    print(f"cond_utils.py is working correctly.  COND_DIM = {COND_DIM}")