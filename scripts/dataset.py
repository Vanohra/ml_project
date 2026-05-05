"""
dataset.py
----------
Loads HR/LR image pairs for training and validation.

Supports three modes controlled by which constructor arguments you pass:

  MODE A — Pre-saved LR  (Stage 1)
    DIV2KDataset(hr_dir=..., lr_dir=..., patch_size=48)
    Loads matching HR and LR files from disk.
    Use this when you have already run prepare_data.py.

  MODE B — Generic online degradation  (Stage 2)
    DIV2KDataset(hr_dir=..., degradation_fn=degrade, patch_size=48)
    Loads HR only. Generates LR live by calling degradation_fn on each crop.
    Every epoch sees different random degradations → effectively unlimited data.

  MODE C — Domain-conditioned degradation  (Stage 3)
    DIV2KDataset(hr_dir=..., domain="surveillance", patch_size=48)
    Like Mode B, but uses the domain-specific presets from domain_degradation.py.
    Supports "mobile", "surveillance", and "dashcam".

    Training (patch_size is not None):
      Fresh degradation parameters are sampled each __getitem__ call,
      so every epoch sees different random degradations. [OK]

    Validation (patch_size=None):
      Degradation parameters are pre-sampled ONCE at dataset creation time.
      The same validation image always gets the same degradation every epoch,
      so PSNR numbers are comparable across epochs and across runs. [OK]
      (This works because train.py calls set_seed() before creating datasets.)

Think of it as a vending machine:
  Mode A: you pre-packaged the snacks (LR files on disk)
  Mode B: the machine makes each snack fresh every time (generic recipe)
  Mode C: the machine makes each snack fresh using a domain-specific recipe
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Domain names supported by domain_degradation.py
SUPPORTED_DOMAINS = ["mobile", "surveillance", "dashcam"]


class DIV2KDataset(Dataset):
    """
    Loads (LR, HR) image pairs for super-resolution training and validation.

    Arguments
    ---------
    hr_dir         : str or Path — folder with high-resolution images (always required)
    lr_dir         : str or Path — folder with pre-saved LR images (Mode A only)
    degradation_fn : callable   — function that turns a HR PIL image into a LR PIL image
                                  (Mode B only). Example: lambda img: degrade(img, scale=4)
    patch_size     : int or None — LR crop size for training. None = use full images.
    scale          : int         — upscale factor (default 4 — must match your pipeline)
    domain         : str or None — one of "mobile", "surveillance", "dashcam" (Mode C).
                                   When set, domain_degradation.degrade_domain() is used
                                   and degradation_fn is ignored.
    return_metadata: bool        — if True, __getitem__ returns (lr, hr, metadata) instead
                                   of (lr, hr). metadata is the dict of sampled degradation
                                   parameters for that sample.
                                   Only populated in Mode C (domain degradation).
                                   NOT compatible with PyTorch DataLoader — use direct
                                   indexing (dataset[i]) for inspection and debugging.
    return_cond_vector: bool     — if True, __getitem__ returns (lr, hr, cond_tensor) where
                                   cond_tensor is a float32 tensor of shape (COND_DIM,)
                                   built by cond_utils.build_cond_vector().
                                   Only valid in Mode C (domain degradation).
                                   IS compatible with PyTorch DataLoader — collates into
                                   (lr_batch, hr_batch, cond_batch) of shape (B, COND_DIM).

    patch_size controls training vs. validation behaviour:
      patch_size=48   → returns 48×48 LR and 192×192 HR patches  (training)
      patch_size=None → returns full images at their original size (validation)
    """

    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(
        self,
        hr_dir,
        lr_dir=None,
        degradation_fn=None,
        patch_size=None,
        scale=4,
        domain=None,
        return_metadata=False,
        return_cond_vector=False,
        curriculum_stage=None,
        upsample_lr=True,
    ):
        # ── Validate arguments ────────────────────────────────────────────────
        if lr_dir is None and degradation_fn is None and domain is None:
            raise ValueError(
                "You must provide one LR source:\n"
                "  lr_dir         — Mode A: load pre-saved LR files from disk\n"
                "  degradation_fn — Mode B: generic online degradation function\n"
                "  domain         — Mode C: domain-conditioned degradation preset"
            )

        if domain is not None and domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Supported: {SUPPORTED_DOMAINS}"
            )

        if return_cond_vector and domain is None:
            raise ValueError(
                "return_cond_vector=True requires domain to be set (Mode C). "
                "Conditioning vectors are only available when domain degradation is used."
            )

        # ── Store settings ────────────────────────────────────────────────────
        self.hr_dir             = Path(hr_dir)
        self.lr_dir             = Path(lr_dir) if lr_dir else None
        self.patch_size         = patch_size
        self.scale              = scale
        self.domain             = domain
        self.return_metadata    = return_metadata
        self.return_cond_vector = return_cond_vector
        # upsample_lr=True  → default/legacy behaviour (raw LR returned; the
        #                       training loop handles bicubic pre-upsampling for
        #                       models like SimpleSRCNN that need it).
        # upsample_lr=False → explicitly request raw LR at LR resolution;
        #                       used with SRResNet which upsamples internally.
        # Both values currently produce the same raw LR tensor — the flag is
        # semantic documentation for the training script's intent.
        self.upsample_lr        = upsample_lr

        # curriculum_stage is intentionally mutable so the training loop can
        # update it each epoch without recreating the dataset.
        # Only used for TRAINING (patch_size is not None).
        # Validation always uses pre-sampled params regardless of this field.
        # Set by train.py: train_dataset.curriculum_stage = get_active_stage(epoch, cfg)
        self.curriculum_stage = curriculum_stage

        # Lazy-import build_cond_vector only when conditioning vectors are needed
        if return_cond_vector:
            from cond_utils import build_cond_vector as _bcv
            self._build_cond_vector = _bcv
        else:
            self._build_cond_vector = None

        # domain takes precedence over degradation_fn when both are supplied.
        # We import lazily so the rest of the project doesn't depend on
        # domain_degradation.py unless Mode C is actually used.
        if domain is not None:
            from domain_degradation import degrade_domain, sample_domain_params
            self._degrade_domain       = degrade_domain
            self._sample_domain_params = sample_domain_params
            self.degradation_fn        = None   # domain overrides this
        else:
            self._degrade_domain       = None
            self._sample_domain_params = None
            self.degradation_fn        = degradation_fn

        # ── Collect image filenames ───────────────────────────────────────────
        # Sorted so the order is deterministic across runs and operating systems.
        self.filenames = sorted([
            p.name for p in self.hr_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED
        ])

        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.hr_dir}")

        # ── Pre-sample validation parameters (Mode C, validation only) ────────
        #
        # For validation datasets (patch_size=None) in domain mode, we sample
        # one set of degradation parameters per image right now, at construction
        # time. From then on, dataset[i] always applies the same degradation to
        # image i, so PSNR is meaningful and comparable across epochs.
        #
        # This works because train.py calls set_seed() before creating datasets,
        # so the random state here is deterministic — same seed → same val params.
        #
        # For training datasets (patch_size is not None), we leave _val_params
        # as None and sample fresh parameters every __getitem__ call instead.
        self._val_params = None
        if domain is not None and patch_size is None:
            self._val_params = [
                self._sample_domain_params(domain)
                for _ in range(len(self.filenames))
            ]

    def __len__(self):
        return len(self.filenames)

    # ── Mode A helpers ────────────────────────────────────────────────────────

    def _find_lr_path(self, hr_filename: str) -> Path:
        """
        Locates the LR file on disk that matches the given HR filename.

        prepare_data.py always saves LR images as .png (even when the HR is .jpg),
        so we check both the exact name and a .png fallback.
        """
        exact = self.lr_dir / hr_filename
        if exact.exists():
            return exact

        fallback = self.lr_dir / (Path(hr_filename).stem + ".png")
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"No LR file found for '{hr_filename}' in {self.lr_dir}\n"
            "Run prepare_data.py first, or switch to online degradation mode."
        )

    def _paired_crop(
        self, lr_img: Image.Image, hr_img: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """
        (Mode A) Cuts matching crops from pre-aligned LR and HR images.

        The crop origin is chosen in LR pixel space, then multiplied by scale
        to find the exact matching position in HR pixel space.
        """
        lr_w, lr_h = lr_img.size
        max_x = lr_w - self.patch_size
        max_y = lr_h - self.patch_size

        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"Image too small for patch_size={self.patch_size}. "
                f"LR image size: {lr_img.size}"
            )

        lr_x = random.randint(0, max_x)
        lr_y = random.randint(0, max_y)
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale
        hr_p = self.patch_size * self.scale

        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + self.patch_size, lr_y + self.patch_size))
        hr_crop = hr_img.crop((hr_x, hr_y, hr_x + hr_p,            hr_y + hr_p))
        return lr_crop, hr_crop

    # ── Mode B / C helper ─────────────────────────────────────────────────────

    def _hr_crop(self, hr_img: Image.Image) -> Image.Image:
        """
        (Mode B / C) Cuts a random patch from the HR image.

        The patch covers (patch_size × scale) pixels in HR space.
        Example: patch_size=48, scale=4 → cut a 192×192 crop from the HR image.

        We crop from HR only (there is no pre-saved LR), then the caller
        passes the crop to the degradation function to produce the LR patch.
        """
        hr_patch = self.patch_size * self.scale
        hr_w, hr_h = hr_img.size
        max_x = hr_w - hr_patch
        max_y = hr_h - hr_patch

        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"Image too small for patch_size={self.patch_size} × scale={self.scale}. "
                f"HR size: {hr_img.size}. Reduce patch_size or use larger images."
            )

        hr_x = random.randint(0, max_x)
        hr_y = random.randint(0, max_y)
        return hr_img.crop((hr_x, hr_y, hr_x + hr_patch, hr_y + hr_patch))

    # ── Core data loading ─────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        """
        Returns one training or validation sample.

        Default return  (return_metadata=False):
          (lr_tensor, hr_tensor)
            lr_tensor : FloatTensor shape (3, lrH, lrW)  values in [0, 1]
            hr_tensor : FloatTensor shape (3, hrH, hrW)  values in [0, 1]

        With return_metadata=True and domain mode:
          (lr_tensor, hr_tensor, metadata)
            metadata  : dict — the exact degradation parameters used.
                        Use for logging, debugging, or reproducibility.
                        Not suitable for use with DataLoader batching.

        Mode A (pre-saved LR):
          load HR + LR from disk → crop matching patches → tensors

        Mode B (generic degradation_fn):
          load HR → crop → call degradation_fn(hr_crop) → tensors

        Mode C training (domain, patch_size is not None):
          load HR → crop → sample fresh domain params → degrade → tensors [+ metadata]
          (different params every call → diverse training data)

        Mode C validation (domain, patch_size is None):
          load HR → degrade with pre-sampled params[index] → tensors [+ metadata]
          (same params every call for the same index → reproducible PSNR)
        """
        filename = self.filenames[index]
        hr_img   = Image.open(self.hr_dir / filename).convert("RGB")
        metadata = {}   # only populated in Mode C

        if self.domain is not None:
            # ── Mode C: domain-conditioned degradation ────────────────────────
            if self.patch_size is not None:
                # Training: crop a fresh HR patch, then sample degradation params.
                # If a curriculum stage is set, use curriculum parameter ranges
                # instead of the domain preset so difficulty increases over epochs.
                hr_img = self._hr_crop(hr_img)
                if self.curriculum_stage is not None:
                    # Lazy import — only used when curriculum is active
                    from curriculum import sample_curriculum_params
                    train_params = sample_curriculum_params(self.domain,
                                                            self.curriculum_stage)
                    lr_img, metadata = self._degrade_domain(
                        hr_img, scale=self.scale, domain=self.domain,
                        params=train_params,
                    )
                else:
                    lr_img, metadata = self._degrade_domain(
                        hr_img, scale=self.scale, domain=self.domain,
                    )
            else:
                # Validation: use pre-sampled params AND a fixed numpy seed so
                # the noise pattern is identical across epochs (truly reproducible).
                _np_state = np.random.get_state()
                np.random.seed(index)
                lr_img, metadata = self._degrade_domain(
                    hr_img,
                    scale=self.scale,
                    domain=self.domain,
                    params=self._val_params[index],
                )
                np.random.set_state(_np_state)

        elif self.degradation_fn is not None:
            # ── Mode B: generic online degradation ────────────────────────────
            if self.patch_size is not None:
                hr_img = self._hr_crop(hr_img)
            lr_img = self.degradation_fn(hr_img)

        else:
            # ── Mode A: load pre-saved LR from disk ───────────────────────────
            lr_img = Image.open(self._find_lr_path(filename)).convert("RGB")
            if self.patch_size is not None:
                lr_img, hr_img = self._paired_crop(lr_img, hr_img)

        # to_tensor converts a PIL image (H × W × C, uint8 0-255)
        #                           to a tensor (C × H × W, float32 0.0-1.0)
        lr_tensor = to_tensor(lr_img)
        hr_tensor = to_tensor(hr_img)

        # upsample_lr=True: bicubic-upsample LR to HR spatial size before
        # returning, so models like SimpleSRCNN that operate at HR resolution
        # can receive correctly-sized input from the dataset directly.
        # upsample_lr=False: return raw LR at native (smaller) resolution;
        # SRResNet and the training loop handle upsampling internally.
        if self.upsample_lr:
            hr_h, hr_w = hr_tensor.shape[1], hr_tensor.shape[2]
            lr_tensor = F.interpolate(
                lr_tensor.unsqueeze(0),
                size=(hr_h, hr_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)

        if self.return_cond_vector:
            # Build a fixed-shape float tensor from the metadata dict.
            # This IS DataLoader-safe — shape (COND_DIM,) collates cleanly.
            cond = self._build_cond_vector(metadata)
            return lr_tensor, hr_tensor, cond

        if self.return_metadata:
            return lr_tensor, hr_tensor, metadata

        return lr_tensor, hr_tensor


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run:  python scripts/dataset.py
# from the ml_project/ directory.

if __name__ == "__main__":
    from degradation import degrade

    base = Path(__file__).parent.parent / "data" / "DIV2K"
    SCALE = 4
    PATCH = 48

    def section(title: str):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)

    # ── upsample_lr parameter test ────────────────────────────────────────────
    section("upsample_lr: Mode B patch (True vs False) and Mode B full (True)")
    try:
        deg_fn = lambda img: degrade(img, scale=SCALE)

        # Mode B, upsample_lr=True, patch — LR should be at HR size (192×192)
        ds_up_patch = DIV2KDataset(hr_dir=base / "HR_train",
                                   degradation_fn=deg_fn,
                                   patch_size=PATCH, scale=SCALE, upsample_lr=True)
        lr_up, hr_up = ds_up_patch[0]
        assert lr_up.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE]), \
            f"upsample_lr=True patch LR shape wrong: {lr_up.shape}"
        assert hr_up.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE])
        print(f"  Mode B, upsample_lr=True,  patch — LR: {tuple(lr_up.shape)}  "
              f"HR: {tuple(hr_up.shape)}  [OK]")

        # Mode B, upsample_lr=False, patch — LR should be raw (48×48)
        ds_raw_patch = DIV2KDataset(hr_dir=base / "HR_train",
                                    degradation_fn=deg_fn,
                                    patch_size=PATCH, scale=SCALE, upsample_lr=False)
        lr_raw, hr_raw = ds_raw_patch[0]
        assert lr_raw.shape == torch.Size([3, PATCH, PATCH]), \
            f"upsample_lr=False patch LR shape wrong: {lr_raw.shape}"
        assert hr_raw.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE])
        print(f"  Mode B, upsample_lr=False, patch — LR: {tuple(lr_raw.shape)}  "
              f"HR: {tuple(hr_raw.shape)}  [OK]")

        # Mode B, upsample_lr=True, full image — LR should be upsampled to HR size
        ds_up_full = DIV2KDataset(hr_dir=base / "HR_train",
                                   degradation_fn=deg_fn,
                                   patch_size=None, scale=SCALE, upsample_lr=True)
        lr_full, hr_full = ds_up_full[0]
        assert lr_full.shape == hr_full.shape, \
            f"upsample_lr=True full: LR {lr_full.shape} != HR {hr_full.shape}"
        print(f"  Mode B, upsample_lr=True,  full  — LR: {tuple(lr_full.shape)}  "
              f"HR: {tuple(hr_full.shape)}  [OK]")

        print("  upsample_lr=True  → LR bicubic-upscaled to HR spatial size")
        print("  upsample_lr=False → LR returned at native LR resolution")
    except FileNotFoundError as e:
        print(f"  Skipped — HR images not found:\n  {e}")

    # ── Mode A: pre-saved LR ──────────────────────────────────────────────────
    section("Mode A — pre-saved LR files (Stage 1)")
    try:
        val_ds = DIV2KDataset(hr_dir=base / "HR_valid", lr_dir=base / "LR_valid")
        lr, hr = val_ds[0]
        print(f"  Full image  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")

        train_ds = DIV2KDataset(hr_dir=base / "HR_train",lr_dir=base / "LR_train",patch_size=PATCH,
                                scale=SCALE,
                                upsample_lr=False
                                )
        lr, hr = train_ds[0]
        assert lr.shape == torch.Size([3, PATCH, PATCH])
        assert hr.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE])
        print(f"  Patch mode  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
        print("  Shape check passed.  [OK]")
    except FileNotFoundError as e:
        print(f"  Skipped — LR files not found (run prepare_data.py first):\n  {e}")

    # ── Mode B: generic degradation_fn ────────────────────────────────────────
    section("Mode B — generic online degradation (Stage 2, unchanged)")
    try:
        ds_b = DIV2KDataset(
            hr_dir=base / "HR_train",
            degradation_fn=lambda img: degrade(img, scale=SCALE),
            patch_size=PATCH, scale=SCALE,
        )
        lr, hr = ds_b[0]
        assert lr.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE])
        assert hr.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE])
        print(f"  Patch mode  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
        print("  Shape check passed.  [OK]")

        lr_a, _ = ds_b[0]
        lr_b, _ = ds_b[0]
        assert not torch.equal(lr_a, lr_b)
        print("  Randomness check passed (two calls differ).  [OK]")
    except FileNotFoundError as e:
        print(f"  Skipped — HR images not found:\n  {e}")

    # ── Mode C: domain-conditioned training ───────────────────────────────────
    for domain in SUPPORTED_DOMAINS:
        section(f"Mode C — '{domain}' domain, training (patch_size={PATCH})")
        try:
            ds_c = DIV2KDataset(
                hr_dir=base / "HR_train",
                patch_size=PATCH, scale=SCALE,
                domain=domain,
            )
            print(f"  {len(ds_c)} HR image(s) found.")

            lr, hr = ds_c[0]
            assert lr.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE]), \
                f"LR shape wrong: {lr.shape}"
            assert hr.shape == torch.Size([3, PATCH * SCALE, PATCH * SCALE]), \
                f"HR shape wrong: {hr.shape}"
            print(f"  Patch mode  — LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
            print("  Shape check passed.  [OK]")

            # Training calls should produce different LR for the same index
            lr_a, _ = ds_c[0]
            lr_b, _ = ds_c[0]
            assert not torch.equal(lr_a, lr_b), \
                "Training should give fresh degradation every call!"
            print("  Training randomness check passed.  [OK]")
        except FileNotFoundError as e:
            print(f"  Skipped — HR images not found:\n  {e}")

    # ── Mode C + return_metadata ──────────────────────────────────────────────
    section("Mode C — return_metadata=True  (surveillance domain)")
    try:
        ds_meta = DIV2KDataset(
            hr_dir=base / "HR_train",
            patch_size=PATCH, scale=SCALE,
            domain="surveillance",
            return_metadata=True,
        )
        result = ds_meta[0]
        assert len(result) == 3, "Expected (lr, hr, metadata) tuple of length 3"
        lr, hr, metadata = result

        print(f"  LR: {tuple(lr.shape)}  HR: {tuple(hr.shape)}")
        print(f"  metadata keys : {list(metadata.keys())}")

        s1 = metadata["stage1"]
        print(f"  stage1:")
        print(f"    blur_sigma   : {s1['blur_sigma']:.3f}")
        print(f"    resize_method: {s1['resize_method']}")
        print(f"    noise_sigma  : {s1['noise_sigma']:.1f}")
        print(f"    jpeg_quality : {s1['jpeg_quality']}")

        if metadata["stage2"] is not None:
            s2 = metadata["stage2"]
            print(f"  stage2 (applied):")
            print(f"    blur_sigma   : {s2['blur_sigma']:.3f}")
            print(f"    noise_sigma  : {s2['noise_sigma']:.1f}")
            print(f"    jpeg_quality : {s2['jpeg_quality']}")
        else:
            print("  stage2: skipped this sample")

        print("  Metadata shape + keys check passed.  [OK]")
    except FileNotFoundError as e:
        print(f"  Skipped — HR images not found:\n  {e}")

    # ── Mode C validation: determinism check ──────────────────────────────────
    section("Mode C — deterministic validation (dashcam, patch_size=None)")
    try:
        val_domain = DIV2KDataset(
            hr_dir=base / "HR_valid",
            patch_size=None, scale=SCALE,
            domain="dashcam",
        )
        print(f"  {len(val_domain)} validation image(s) found.")
        print(f"  {len(val_domain._val_params)} pre-sampled parameter sets.")

        # Same index must give identical LR every call
        lr_a, _ = val_domain[0]
        lr_b, _ = val_domain[0]
        assert torch.equal(lr_a, lr_b), \
            "Validation should return identical LR for repeated calls on the same index!"
        print("  Reproducibility check passed (same index -> same LR).  [OK]")

        # Different indices must give different LR
        lr_0, _ = val_domain[0]
        lr_1, _ = val_domain[1]
        assert not torch.equal(lr_0, lr_1), \
            "Different indices should give different LR tensors!"
        print("  Diversity check passed (different indices -> different LR).  [OK]")

    except FileNotFoundError as e:
        print(f"  Skipped — HR_valid images not found:\n  {e}")

    print("\n" + "=" * 60)
    print("  dataset.py self-test complete.")
    print("=" * 60)
