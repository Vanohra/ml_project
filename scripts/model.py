"""
model.py
--------
Defines SimpleSRCNN — the smallest reasonable super-resolution CNN.

Architecture overview:
  The model takes a LOW-RESOLUTION image that has already been upscaled
  to HR size using bicubic interpolation (the same blurry resize from baseline.py).
  It then learns to SHARPEN and CORRECT that blurry image.

  Think of it as an AI sharpening filter — it doesn't invent new structure,
  it learns from thousands of examples what a sharp version should look like.

  Input : bicubic-upscaled LR   shape (batch, 3, H, W)   values in [0, 1]
  Output: sharpened SR image    shape (batch, 3, H, W)   values in [0, 1]

Layer structure (SRCNN — He et al. 2014):
  Conv 9x9 → ReLU   64 filters   extracts low-level features (edges, colours)
  Conv 5x5 → ReLU   32 filters   maps features to sharper representations
  Conv 5x5           3 filters   reconstructs the final RGB image

  Total parameters: ~57,000  (tiny! ResNet-50 has 25 million)

Residual connection:
  output = model(bicubic) + bicubic

  Instead of learning the full sharp image, the model only learns
  what to ADD to the bicubic image.  This is much easier — most pixels
  need very little change, so the model starts close to right from epoch 1.
"""

import torch
import torch.nn as nn


class SimpleSRCNN(nn.Module):
    """
    3-layer super-resolution CNN with a residual (skip) connection.

    nn.Module is the base class for all PyTorch models.
    Every model must define:
      __init__  : create the layers
      forward   : describe how data flows through those layers

    Attribute:
        expects_upsampled_input = True
            Tells train.py that this model receives a bicubic-upsampled LR image
            (already at HR spatial size), not a raw LR image.
    """

    expects_upsampled_input = True   # train.py reads this to prepare model input

    def __init__(self):
        super().__init__()   # always call this first

        # ── Layer definitions ──────────────────────────────────────────────────
        #
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        #
        # kernel_size : how large a patch of pixels each filter looks at
        #               9x9 = sees a wide neighbourhood → good for initial features
        #               5x5 = medium neighbourhood
        #
        # padding     : adds zeros around the image border so the output
        #               stays the same spatial size as the input
        #               Rule: padding = kernel_size // 2  keeps size constant
        #
        # in_channels / out_channels: number of "feature maps"
        #   - 3 in  = RGB image (Red, Green, Blue channels)
        #   - 64    = 64 internal feature maps (like 64 different detectors)
        #   - 3 out = back to RGB

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),   # ReLU: set negatives to 0 (adds non-linearity)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # No ReLU after the last layer — the output should be a full-range image
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=3,  kernel_size=5, padding=2)

        # Initialise weights using a standard method for ReLU networks
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming (He) initialisation: sets starting weights to sensible values
        so training converges faster.  Without this, training can be very slow
        or unstable in the first few epochs.
        """
        for layer in [self.layer1[0], self.layer2[0], self.layer3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: data flows from input x through all layers.

        x       : bicubic-upscaled LR image tensor, values in [0, 1]
        returns : super-resolved image tensor, values clamped to [0, 1]

        The residual connection  (out + x)  means:
          "predict what to ADD to the input, not the full output"
        This trick makes training ~3x faster on small datasets.
        """
        out = self.layer1(x)   # (B, 64, H, W)
        out = self.layer2(out) # (B, 32, H, W)
        out = self.layer3(out) # (B,  3, H, W)

        # Residual: add the original bicubic image back
        # clamp ensures pixel values stay in the valid [0, 1] range
        return torch.clamp(out + x, 0.0, 1.0)


import math


# ── SRResNet building blocks ───────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """Two 3×3 convs with a residual skip: out = f(x) + x."""

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
        return x + self.block(x)


class _UpsampleBlock(nn.Module):
    """Sub-pixel convolution block that doubles spatial resolution (PixelShuffle ×2)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRResNet(nn.Module):
    """
    SRResNet super-resolution model (Ledig et al., CVPR 2017).

    Takes a raw LR image at native resolution and outputs a 4× super-resolved
    image.  Unlike SimpleSRCNN, this model handles upsampling internally using
    learned sub-pixel convolution (PixelShuffle) instead of a fixed bicubic
    pre-upsample.

    Architecture:
      Head   : Conv(3→64, 9×9) + PReLU
      Body   : 8 × ResidualBlock(64)  +  Conv(64→64, 3×3) + BN
               [global skip: add head features before upsample]
      Upsample: 2 × UpsampleBlock (PixelShuffle ×2 each → total ×4)
      Tail   : Conv(64→3, 9×9)

    The model learns the full HR→LR mapping; there is NO residual skip from
    the input image to the output (unlike SimpleSRCNN).

    Parameters: ~957 K  (≈16× more capacity than SimpleSRCNN)
    Scale     : 4× (fixed — matches the surveillance degradation preset)

    Attribute:
        expects_upsampled_input = False
            Tells the training loop that this model receives raw LR, not a
            bicubic-pre-upsampled image.
    """

    expects_upsampled_input = False

    def __init__(self, scale: int = 4, num_res_blocks: int = 8,
                 num_features: int = 64):
        super().__init__()

        if scale not in (2, 4, 8):
            raise ValueError(f"scale must be 2, 4, or 8; got {scale}")

        # ── Head ─────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # ── Body ─────────────────────────────────────────────────────────────
        res_blocks = [_ResidualBlock(num_features) for _ in range(num_res_blocks)]
        self.body = nn.Sequential(
            *res_blocks,
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
        )

        # ── Upsample ──────────────────────────────────────────────────────────
        num_up = int(math.log2(scale))
        self.upsample = nn.Sequential(
            *[_UpsampleBlock(num_features) for _ in range(num_up)]
        )

        # ── Tail ──────────────────────────────────────────────────────────────
        self.tail = nn.Conv2d(num_features, 3, kernel_size=9, padding=4)

        self._init_weights()

    def _init_weights(self) -> None:
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
        x       : raw LR tensor  (B, 3, H, W)           values in [0, 1]
        returns : SR tensor      (B, 3, H×scale, W×scale)  clamped [0, 1]
        """
        head_out = self.head(x)
        body_out = self.body(head_out)
        merged   = head_out + body_out   # global residual stabilises deep training
        up       = self.upsample(merged)
        return torch.clamp(self.tail(up), 0.0, 1.0)


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run: python scripts/model.py

if __name__ == "__main__":
    print("=" * 55)
    print("  model.py — self-test")
    print("=" * 55)

    # ── SimpleSRCNN ───────────────────────────────────────────────────────────
    srcnn = SimpleSRCNN()
    n_srcnn = sum(p.numel() for p in srcnn.parameters())
    fake_lr_up = torch.rand(2, 3, 192, 192)   # already bicubic-upscaled
    fake_sr    = srcnn(fake_lr_up)

    print(f"\nSimpleSRCNN")
    print(f"  Parameters : {n_srcnn:,}")
    print(f"  Input      : {tuple(fake_lr_up.shape)}")
    print(f"  Output     : {tuple(fake_sr.shape)}")
    assert fake_lr_up.shape == fake_sr.shape, "SRCNN output shape mismatch!"
    assert fake_sr.min() >= 0.0 and fake_sr.max() <= 1.0, "SRCNN values out of [0,1]!"
    print("  Shape check   PASSED")
    print("  Range [0,1]   PASSED")

    # ── SRResNet ──────────────────────────────────────────────────────────────
    srresnet = SRResNet(scale=4, num_res_blocks=8, num_features=64)
    n_srresnet = sum(p.numel() for p in srresnet.parameters())
    fake_lr_raw = torch.rand(2, 3, 48, 48)    # raw LR at 1/4 size
    fake_sr_r   = srresnet(fake_lr_raw)

    print(f"\nSRResNet (scale=4, blocks=8, features=64)")
    print(f"  Parameters : {n_srresnet:,}")
    print(f"  Input      : {tuple(fake_lr_raw.shape)}")
    print(f"  Output     : {tuple(fake_sr_r.shape)}")
    assert tuple(fake_sr_r.shape) == (2, 3, 192, 192), \
        f"SRResNet output shape mismatch: {tuple(fake_sr_r.shape)}"
    assert fake_sr_r.min() >= 0.0 and fake_sr_r.max() <= 1.0, \
        "SRResNet values out of [0,1]!"
    print("  Shape check   PASSED  (48->192, x4 upscale)")
    print("  Range [0,1]   PASSED")

    print(f"\n  expects_upsampled_input: "
          f"SimpleSRCNN={SimpleSRCNN.expects_upsampled_input}, "
          f"SRResNet={SRResNet.expects_upsampled_input}")
    assert SimpleSRCNN.expects_upsampled_input is True
    assert SRResNet.expects_upsampled_input is False
    print("  Flag checks   PASSED")

    print("\n" + "=" * 55)
    print("  model.py is working correctly.")
    print("=" * 55)
