"""
metrics.py
----------
Image quality metrics for super-resolution evaluation.

Functions
---------
compute_psnr(img1, img2)
    Peak Signal-to-Noise Ratio (dB).  Higher is better.

compute_ssim(img1, img2)
    Structural Similarity Index (Wang et al., 2004).  Higher is better.
    Implemented in pure PyTorch — no scipy or scikit-image required.

Both functions accept tensors of shape (B, C, H, W) with values in [0, 1].

SSIM formula (per-pixel):
    ssim(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                 ─────────────────────────────────────────
                 (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

where μ, σ are local means and variances computed with a Gaussian kernel,
C1 = (k1·L)² = 0.0001, C2 = (k2·L)² = 0.0009 for L=1 (float range).
"""

import math

import torch
import torch.nn.functional as F


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    PSNR between two image tensors (values in [0, 1]).

    Formula:  PSNR = 10 · log10(1 / MSE)

    Returns infinity when the images are identical (MSE = 0).
    """
    with torch.no_grad():
        mse = F.mse_loss(img1, img2).item()
        if mse == 0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)


def _gaussian_window(window_size: int, sigma: float,
                     channels: int, device: torch.device) -> torch.Tensor:
    """
    Creates a normalised Gaussian kernel expanded for depthwise convolution.

    Returns tensor of shape (channels, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) * g.unsqueeze(0)   # (ws, ws)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Mean SSIM between two image tensors.

    img1, img2 : FloatTensor  (B, C, H, W)  in [0, 1]
    window_size: size of the Gaussian window (default 11)
    sigma      : Gaussian standard deviation (default 1.5)
    k1, k2     : stability constants (default 0.01 and 0.03)

    Returns mean SSIM as a float in approximately [0, 1].
    Higher is better; 1.0 means the images are identical.

    Implementation follows the original SSIM paper:
        Wang et al. (2004) "Image quality assessment: from error visibility
        to structural similarity." IEEE TIP.
    """
    C1 = k1 ** 2   # = 0.0001
    C2 = k2 ** 2   # = 0.0009

    with torch.no_grad():
        channels = img1.shape[1]
        window   = _gaussian_window(window_size, sigma, channels, img1.device)
        pad      = window_size // 2

        def _filt(x: torch.Tensor) -> torch.Tensor:
            """Depthwise Gaussian filter: each channel filtered independently."""
            return F.conv2d(x, window, padding=pad, groups=channels)

        mu1    = _filt(img1)
        mu2    = _filt(img2)

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # Local variance and covariance via Var(X) = E[X²] - E[X]²
        sigma1_sq = _filt(img1 * img1) - mu1_sq
        sigma2_sq = _filt(img2 * img2) - mu2_sq
        sigma12   = _filt(img1 * img2) - mu1_mu2

        numerator   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12   + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map.mean().item()


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run: python scripts/metrics.py

if __name__ == "__main__":
    print("=" * 50)
    print("  metrics.py — self-test")
    print("=" * 50)

    torch.manual_seed(0)
    B, C, H, W = 2, 3, 192, 192

    # ── Identical images → PSNR = inf, SSIM = 1.0 ────────────────────────────
    img = torch.rand(B, C, H, W)
    psnr_id = compute_psnr(img, img)
    ssim_id = compute_ssim(img, img)
    print(f"\nIdentical images:")
    print(f"  PSNR : {psnr_id}  (expected: inf)")
    print(f"  SSIM : {ssim_id:.6f}  (expected ~1.0)")
    assert psnr_id == float("inf"), "PSNR should be inf for identical images"
    assert abs(ssim_id - 1.0) < 1e-4, f"SSIM should be ≈1.0, got {ssim_id}"
    print("  PASSED")

    # ── Noisy pair → PSNR and SSIM should both be in reasonable range ─────────
    noise = torch.rand(B, C, H, W) * 0.1   # ~10 % noise
    noisy = (img + noise).clamp(0, 1)
    psnr_n = compute_psnr(img, noisy)
    ssim_n = compute_ssim(img, noisy)
    print(f"\nNoisy pair (noise sigma~0.05):")
    print(f"  PSNR : {psnr_n:.2f} dB  (expected roughly 20–30 dB)")
    print(f"  SSIM : {ssim_n:.4f}  (expected 0 < SSIM < 1)")
    assert 15.0 < psnr_n < 50.0, f"PSNR out of expected range: {psnr_n}"
    assert 0.0 < ssim_n < 1.0,   f"SSIM out of [0,1]: {ssim_n}"
    print("  PASSED")

    # ── Completely different images → SSIM near 0 ─────────────────────────────
    img_b = torch.rand(B, C, H, W)
    ssim_diff = compute_ssim(img, img_b)
    print(f"\nRandom pair:")
    print(f"  SSIM : {ssim_diff:.4f}  (expected near 0, possibly slightly negative)")
    print("  PASSED")

    print("\n" + "=" * 50)
    print("  metrics.py is working correctly.")
    print("=" * 50)
