"""
losses.py
---------
Configurable training losses for super-resolution.

All losses are enabled and weighted from the config (see configs/default.yaml).
The factory function build_loss(cfg, device) returns a CombinedLoss module that
can be used as a drop-in replacement for any single nn.Loss.

  loss_total, components = criterion(sr_batch, hr_batch)
  # components = {"pixel": 0.012, "perceptual": 0.003, "total": 0.015}

──────────────────────────────────────────────────────────────────────────────
Available loss terms
──────────────────────────────────────────────────────────────────────────────

  pixel       — L1 or MSE reconstruction loss (always recommended, default L1).

                L1 (mean absolute error): penalises each pixel proportionally
                to how wrong it is.  Tends to produce sharper results than MSE
                because very large errors are not squared, so the optimiser has
                less incentive to "play it safe" with a blurry average.

                MSE (mean squared error): squares each error before averaging.
                Large errors are penalised far more than small ones, which
                encourages conservative output — images look smooth but blurry.
                Still useful when PSNR is the primary evaluation metric because
                PSNR is defined in terms of MSE.

                Recommendation for this project: L1.

  perceptual  — Feature-space loss using VGG16 pretrained on ImageNet.

                Compares SR and HR images in terms of how a pretrained CNN
                represents them, rather than pixel-by-pixel.  Captures
                texture and structural similarity that pixel losses miss.

                Network used: VGG16 (torchvision.models.vgg16)
                              pretrained on ImageNet (ILSVRC-2012)
                              weights: VGG16_Weights.IMAGENET1K_V1
                              downloaded automatically from PyTorch Hub on first use.

                Layer:        relu2_2  (VGG16 feature index 0–8, the ReLU after
                              the second Conv of block 2, 128 feature maps at
                              1/2 spatial resolution).
                              relu2_2 captures mid-level textures and edges.
                              It is shallower than the relu3_4 used in SRGAN,
                              which makes it less likely to introduce hallucinated
                              high-frequency patterns — safer for a class project.

                Images are normalised to ImageNet statistics before being passed
                through VGG (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

                The perceptual loss weight is typically much smaller than the
                pixel weight (e.g. pixel_weight=1.0, perceptual_weight=0.1).
                Values above 0.5 tend to introduce texture artifacts.

                Requires torchvision (already in requirements.txt).
                No extra installation needed.

  adversarial — NOT implemented.

                GAN training requires a discriminator network, a two-player
                training schedule, careful balancing of generator and discriminator
                learning rates, and monitoring of mode collapse.  Getting this
                wrong produces outputs worse than bicubic.  The complexity is
                not appropriate for a class-project deadline.

                If you need perceptual quality beyond what pixel + perceptual
                losses provide, train a longer run with a higher perceptual weight.

──────────────────────────────────────────────────────────────────────────────
Config reference (see configs/default.yaml)
──────────────────────────────────────────────────────────────────────────────

  loss:
    pixel:
      type: l1        # "l1" or "mse"
      weight: 1.0
    perceptual:
      enabled: false  # requires torchvision (already installed)
      weight: 0.1
      layer: relu2_2  # "relu2_2" (default) or "relu3_3" (deeper, riskier)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── VGG layer name → last feature-index map ───────────────────────────────────
# VGG16.features is a Sequential of Conv/ReLU/MaxPool layers.
# Index 8  = ReLU after block2 conv2  (relu2_2)  — 128ch, H/2 × W/2
# Index 15 = ReLU after block3 conv3  (relu3_3)  — 256ch, H/4 × W/4
_VGG_LAYER_END = {
    "relu2_2": 8,
    "relu3_3": 15,
}

# ImageNet normalisation constants (VGG was trained on these)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Perceptual (VGG) loss ─────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a frozen VGG16 network.

    The network weights are fixed (requires_grad=False) — they are never
    updated during SR training.  Only the SR model learns.

    Inputs are expected in [0, 1]; this module applies ImageNet normalisation
    internally before passing them through VGG.
    """

    def __init__(self, layer: str = "relu2_2"):
        super().__init__()
        if layer not in _VGG_LAYER_END:
            raise ValueError(
                f"Unknown VGG layer '{layer}'. "
                f"Choose from: {list(_VGG_LAYER_END)}"
            )
        end_idx = _VGG_LAYER_END[layer]

        # Load pretrained VGG16 — auto-downloads weights on first run (~58 MB)
        try:
            from torchvision.models import vgg16, VGG16_Weights
            full_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except (TypeError, AttributeError):
            # Older torchvision (<0.13) uses the deprecated pretrained flag
            from torchvision.models import vgg16
            full_vgg = vgg16(pretrained=True)

        # Keep only layers up to (and including) the chosen ReLU
        self.features = nn.Sequential(*list(full_vgg.features.children())[:end_idx + 1])

        # Freeze completely — never update during SR training
        for p in self.features.parameters():
            p.requires_grad = False

        self.layer_name = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : (B, 3, H, W) float in [0, 1]
        Returns the feature map at the chosen VGG layer.
        """
        # Normalise to ImageNet statistics
        mean = _IMAGENET_MEAN.to(x.device)
        std  = _IMAGENET_STD.to(x.device)
        x    = (x - mean) / std
        return self.features(x)


# ── Combined loss module ──────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Weighted sum of enabled loss terms.

    forward(sr, hr) returns:
        total_loss  : scalar Tensor (backprop through this)
        components  : dict[str, float]  — detached per-term values for logging

    Keys in components always include "total".
    Additional keys ("pixel", "perceptual") are present only when those terms
    are active.
    """

    def __init__(self, pixel_type: str, pixel_weight: float,
                 perceptual_enabled: bool, perceptual_weight: float,
                 perceptual_layer: str):
        super().__init__()

        # ── Pixel loss ────────────────────────────────────────────────────────
        if pixel_type == "l1":
            self.pixel_fn = nn.L1Loss()
        elif pixel_type == "mse":
            self.pixel_fn = nn.MSELoss()
        else:
            raise ValueError(f"loss.pixel.type must be 'l1' or 'mse', got '{pixel_type}'")
        self.pixel_weight = pixel_weight

        # ── Perceptual loss ───────────────────────────────────────────────────
        self.use_perceptual    = perceptual_enabled
        self.perceptual_weight = perceptual_weight
        if perceptual_enabled:
            self.vgg = VGGFeatureExtractor(layer=perceptual_layer)
        else:
            self.vgg = None

    def forward(self, sr: torch.Tensor,
                hr: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        sr, hr : (B, 3, H, W) float in [0, 1].
        Returns (total_loss_tensor, components_dict).
        """
        components = {}
        total      = torch.tensor(0.0, device=sr.device)

        # Pixel term — always active
        pix = self.pixel_fn(sr, hr)
        total = total + self.pixel_weight * pix
        components["pixel"] = pix.item()

        # Perceptual term — optional
        if self.use_perceptual and self.vgg is not None:
            with torch.no_grad():
                hr_feat = self.vgg(hr.clamp(0, 1))
            sr_feat = self.vgg(sr.clamp(0, 1))
            perc    = F.l1_loss(sr_feat, hr_feat)
            total   = total + self.perceptual_weight * perc
            components["perceptual"] = perc.item()

        components["total"] = total.item()
        return total, components


# ── Factory function ──────────────────────────────────────────────────────────

def build_loss(cfg: dict, device: torch.device) -> CombinedLoss:
    """
    Builds a CombinedLoss from a config dict.

    Reads cfg["loss"] with these defaults if the key is absent:
      pixel.type     : "l1"
      pixel.weight   : 1.0
      perceptual.enabled : false
      perceptual.weight  : 0.1
      perceptual.layer   : "relu2_2"
    """
    loss_cfg = cfg.get("loss", {})

    pixel_cfg  = loss_cfg.get("pixel", {})
    pixel_type = pixel_cfg.get("type",   "l1")
    pixel_w    = float(pixel_cfg.get("weight", 1.0))

    perc_cfg   = loss_cfg.get("perceptual", {})
    perc_on    = bool(perc_cfg.get("enabled", False))
    perc_w     = float(perc_cfg.get("weight", 0.1))
    perc_layer = perc_cfg.get("layer", "relu2_2")

    loss_fn = CombinedLoss(
        pixel_type=pixel_type,
        pixel_weight=pixel_w,
        perceptual_enabled=perc_on,
        perceptual_weight=perc_w,
        perceptual_layer=perc_layer,
    ).to(device)

    # Print a one-line summary so the user knows what is active
    parts = [f"pixel={pixel_type} (w={pixel_w})"]
    if perc_on:
        parts.append(f"perceptual=VGG16-{perc_layer} (w={perc_w})")
    print(f"[loss] {' + '.join(parts)}")
    if perc_on:
        print("  [perceptual] VGG16 pretrained on ImageNet — "
              "downloading weights on first run if not cached.")

    return loss_fn
