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
    """

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


# ── Quick self-test ────────────────────────────────────────────────────────────
# Run: python scripts/model.py

if __name__ == "__main__":
    model = SimpleSRCNN()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SimpleSRCNN")
    print(f"Total parameters: {num_params:,}")

    # Test with a fake batch: 2 images, 3 channels, 192x192 pixels
    fake_input = torch.rand(2, 3, 192, 192)
    fake_output = model(fake_input)

    print(f"Input  shape: {tuple(fake_input.shape)}")
    print(f"Output shape: {tuple(fake_output.shape)}")
    assert fake_input.shape == fake_output.shape, "Output shape mismatch!"
    print("Shape check passed.")
    print("model.py is working correctly.")
