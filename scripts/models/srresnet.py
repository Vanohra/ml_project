"""
srresnet.py
-----------
SRResNet — the generator backbone from SRGAN (Ledig et al., CVPR 2017).

Adapted for a student project:
  8 residual blocks (original used 16)   — still much stronger than SRCNN
  64 feature channels                    — same as the original
  PixelShuffle sub-pixel upsampling      — same as the original
  ~957K parameters                       — ~16× more capacity than SimpleSRCNN

Architecture overview:
  Input : LR image         shape (batch, 3, H, W)          values in [0, 1]
  ──────────────────────────────────────────────────────────────────────────
  Head   Conv 9×9 + PReLU                 64 feature maps
  Body   N × ResidualBlock               repeated feature refinement
         Conv 3×3 + BN                   merge residual outputs
         [global skip: add head output]  prevents gradient vanishing
  Up     log2(scale) × UpsampleBlock     sub-pixel convolution (PixelShuffle)
  Tail   Conv 9×9                        reconstruct 3-channel RGB
  ──────────────────────────────────────────────────────────────────────────
  Output: SR image         shape (batch, 3, H×scale, W×scale)  clamped [0, 1]

Why SRResNet is stronger than SimpleSRCNN:
  1. Deeper — residual blocks let the model learn more complex mappings without
     the gradient-vanishing problems that plague plain deep networks.
  2. Efficient — convolutions run at LR resolution (quarter of the pixels),
     so more layers fit within the same compute budget as shallow SRCNN.
  3. Learned upsampling — PixelShuffle decides how to expand features into
     pixels, rather than using a fixed bicubic formula.
  4. More capacity — 16× more parameters to store and combine edge / texture
     patterns that SRCNN simply cannot represent.

Reference: Ledig et al., "Photo-Realistic Single Image Super-Resolution Using
a Generative Adversarial Network", CVPR 2017. arXiv:1609.04802
"""

import math

import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    One residual block: two 3×3 convolutions connected by a skip connection.

    Skip connections ( out = f(x) + x ) let gradients flow directly from
    later layers back to earlier ones during backpropagation.  Without them,
    very deep networks suffer from vanishing gradients and fail to train.

    Block structure:
      x → Conv3×3 → BN → PReLU → Conv3×3 → BN → (+ x) → out
                                                   ↑ skip

    PReLU vs ReLU: PReLU has a small learnable negative slope instead of a
    hard zero.  This helps preserve low-amplitude signal in deep networks.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)   # residual: learned correction added to input


class UpsampleBlock(nn.Module):
    """
    Doubles spatial resolution using sub-pixel convolution (PixelShuffle).

    How PixelShuffle works:
      A convolution produces r² times more channels than needed.
      PixelShuffle then rearranges those extra channels into spatial pixels,
      effectively "shuffling" channels → pixels without any fixed interpolation.

      Tensor shape change for scale_factor=2:
        (B, C×4, H, W)  →  (B, C, 2H, 2W)

    This is sharper than bilinear/bicubic because the network learns exactly
    how to distribute its features into the expanded spatial grid.

    For a ×4 SR model: apply this block twice (2× then 2× = 4×).
    """

    def __init__(self, num_features: int, scale_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            # Expand channels by scale_factor² for PixelShuffle to consume
            nn.Conv2d(num_features, num_features * scale_factor ** 2,
                      kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),   # rearrange channels → pixels
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class SRResNet(nn.Module):
    """
    SRResNet super-resolution model.

    Takes a raw LR image and outputs a super-resolved HR image.
    Unlike SRCNN, this model performs upsampling internally — the training loop
    does NOT bicubic-upsample the LR input before feeding it here.

    Args:
        scale          : upscale factor — 2, 4, or 8 (must be a power of 2)
        num_res_blocks : residual blocks in the body (default 8;
                         original paper used 16, which is slower but stronger)
        num_features   : feature channels throughout the network (default 64)

    Attribute:
        expects_upsampled_input = False
            Tells train.py that this model receives raw LR, not bicubic-upsampled.
    """

    expects_upsampled_input = False   # train.py reads this to choose model input

    def __init__(self, scale: int = 4, num_res_blocks: int = 8,
                 num_features: int = 64):
        super().__init__()

        if scale not in (2, 4, 8):
            raise ValueError(
                f"scale must be 2, 4, or 8 (power of two); got {scale}"
            )

        # ── Head: initial feature extraction from the LR image ────────────────
        # Large 9×9 kernel here gives the model a wide receptive field up front.
        self.head = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # ── Body: deep residual feature refinement ────────────────────────────
        # All convolutions here run at LR resolution — cheap, so we can stack many.
        res_blocks = [ResidualBlock(num_features) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(
            *res_blocks,
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            # Note: no activation here — the global skip (head + body) is the
            # full feature before upsampling. Activating here would distort that.
        )

        # ── Upsample: sub-pixel convolution blocks ────────────────────────────
        # One UpsampleBlock per factor-of-2 step:
        #   scale=4 → 2 blocks (×2 then ×2)
        #   scale=2 → 1 block
        #   scale=8 → 3 blocks
        num_up_blocks = int(math.log2(scale))
        self.upsample = nn.Sequential(
            *[UpsampleBlock(num_features, scale_factor=2)
              for _ in range(num_up_blocks)]
        )

        # ── Tail: final RGB reconstruction ────────────────────────────────────
        # 9×9 kernel mirrors the head — wide receptive field for final synthesis.
        self.tail = nn.Conv2d(num_features, 3, kernel_size=9, padding=4)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv2d, ones/zeros for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x       : LR image tensor  (B, 3, H, W),          values in [0, 1]
        returns : SR image tensor  (B, 3, H×scale, W×scale),  clamped [0, 1]
        """
        head_out = self.head(x)          # (B, C, H, W)  — initial features
        body_out = self.body(head_out)   # (B, C, H, W)  — refined features

        # Global residual: add the head output to the body output.
        # This skip connection stabilises training of the full deep network,
        # analogous to how each ResidualBlock has a local skip inside it.
        merged = head_out + body_out     # (B, C, H, W)

        up  = self.upsample(merged)      # (B, C, H×scale, W×scale)
        out = self.tail(up)              # (B, 3, H×scale, W×scale)

        return torch.clamp(out, 0.0, 1.0)


# ── Self-test ──────────────────────────────────────────────────────────────────
# Run: python scripts/models/srresnet.py

if __name__ == "__main__":
    print("=" * 50)
    print("SRResNet shape tests")
    print("=" * 50)

    # ── ×4 model (the main use case) ──────────────────────────────────────────
    model_x4 = SRResNet(scale=4, num_res_blocks=8, num_features=64)
    num_params = sum(p.numel() for p in model_x4.parameters())
    print(f"\nSRResNet (scale=4, blocks=8, features=64)")
    print(f"  Parameters : {num_params:,}")

    lr = torch.rand(2, 3, 48, 48)   # batch of 2 LR patches (48×48)
    sr = model_x4(lr)
    expected = (2, 3, 192, 192)      # 48×4 = 192

    print(f"  Input  : {tuple(lr.shape)}  (LR patches)")
    print(f"  Output : {tuple(sr.shape)}  (SR patches, 4× upscaled)")
    assert tuple(sr.shape) == expected, \
        f"Shape mismatch! Expected {expected}, got {tuple(sr.shape)}"
    assert sr.min() >= 0.0 and sr.max() <= 1.0, "Output values outside [0, 1]!"
    print("  Shape check   PASSED")
    print("  Range [0,1]   PASSED")

    # ── ×2 model ──────────────────────────────────────────────────────────────
    model_x2 = SRResNet(scale=2, num_res_blocks=4, num_features=32)
    lr2 = torch.rand(1, 3, 64, 64)
    sr2 = model_x2(lr2)
    assert tuple(sr2.shape) == (1, 3, 128, 128), \
        f"x2 shape mismatch: {tuple(sr2.shape)}"
    print(f"\nSRResNet (scale=2)  shape check  PASSED")

    # ── Flag check ────────────────────────────────────────────────────────────
    assert SRResNet.expects_upsampled_input is False, \
        "expects_upsampled_input should be False for SRResNet"
    print("expects_upsampled_input = False  PASSED")

    print("\nsrresnet.py is working correctly.")
