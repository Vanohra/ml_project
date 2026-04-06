"""
conditioned_sr.py
-----------------
Degradation-aware super-resolution model: ConditionedSRResNet.

This model extends SRResNet (srresnet.py) with FiLM conditioning — it takes
a degradation metadata vector alongside the LR image and uses that information
to modulate its internal feature maps.

Why bother conditioning on degradation parameters?
  A standard SRResNet sees only the degraded LR pixels.  It must "guess" how
  badly the image was degraded from visual cues alone.  When we hand it the
  actual degradation parameters (blur strength, noise level, JPEG quality, etc.)
  it can adapt its restoration strategy per image instead of applying one
  average strategy to everything.

  This matters more as degradation severity varies.  A surveillance model will
  see images ranging from lightly blurred (clear day, close camera) to heavily
  noisy (night, long range).  Conditioning lets the model respond differently
  to each case rather than finding a single compromise.

FiLM conditioning (Feature-wise Linear Modulation — Perez et al., 2018):
  For every feature map channel c and every conditioned residual block:
    output_c = γ_c(z) · conv_output_c  +  β_c(z)
  where z is the degradation embedding and γ, β are predicted by small linear
  layers inside each block.

  Intuition: γ amplifies or suppresses channels (learned "what to emphasize"),
  β shifts the baseline value (learned "what to add unconditionally").

  Initialization to identity (γ=1, β=0 at start):
    At initialization the conditioning layers have zero weights, so γ=1 and β=0
    for any input z.  The model therefore starts as a plain SRResNet.
    As training proceeds it gradually learns to exploit the conditioning signal.
    This prevents the conditioning from destabilising early training.

Architecture:
  Input LR image  (B, 3, H, W)  +  conditioning vector  (B, 13)
  ────────────────────────────────────────────────────────────────────────────
  MetadataEncoder   MLP: 13 → 64 → embed_dim      encodes degradation info
  Head              Conv 9×9 + PReLU               initial feature extraction
  N × FiLMResidualBlock                            conditioned residual body
      Conv 3×3 → BN → PReLU → Conv 3×3 → BN       (standard residual path)
      γ_fc(emb), β_fc(emb) → FiLM modulation      (conditioning injection)
      + input skip                                  (residual connection)
  Post-body         Conv 3×3 + BN
  Global skip       add head output to body output  (stabilises training)
  2 × UpsampleBlock Conv 3×3 → PixelShuffle(2) → PReLU   (×4 total)
  Tail              Conv 9×9                        RGB reconstruction
  ────────────────────────────────────────────────────────────────────────────
  Output SR image  (B, 3, H×scale, W×scale)  clamped to [0, 1]

Reference:
  FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
  AAAI 2018.  arXiv:1709.07871

  SRResNet backbone: Ledig et al., "Photo-Realistic Single Image Super-Resolution
  Using a Generative Adversarial Network", CVPR 2017.  arXiv:1609.04802
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure scripts/ is on the path so we can import cond_utils when this file
# is run directly as a script (python scripts/models/conditioned_sr.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cond_utils import COND_DIM


# ── Sub-modules ───────────────────────────────────────────────────────────────

class MetadataEncoder(nn.Module):
    """
    Small MLP that encodes the degradation conditioning vector into an embedding.

    Two hidden layers with SiLU (Swish) activation — smoother gradient flow
    than ReLU, which works well for conditioning networks where the signal
    must propagate cleanly through the MLP into each residual block.

    Input  : (B, cond_dim)    normalized degradation vector
    Output : (B, embed_dim)   dense embedding
    """

    def __init__(self, cond_dim: int, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim, 64)   # at least 64 hidden units
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),                     # SiLU(x) = x * sigmoid(x)  (smooth, no dead neurons)
            nn.Linear(hidden, embed_dim),
            nn.SiLU(),
        )
        # Standard init is fine — this MLP sees normalized [0,1] inputs
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)   # (B, embed_dim)


class FiLMResidualBlock(nn.Module):
    """
    Residual block with FiLM (Feature-wise Linear Modulation) conditioning.

    The block applies a standard residual convolution path and then scales
    and shifts each feature channel based on the degradation embedding:

      out = γ(emb) * conv_block(x)  +  β(emb)   then   return x + out

    γ (gamma) : per-channel scale    — emphasises or suppresses channels
    β (beta)  : per-channel shift    — offsets the feature baseline
    Both are predicted by tiny linear layers inside this block.

    Initialization:
      gamma_fc has ZERO weights, bias ONE  → γ = 1 for any embedding at init
      beta_fc  has ZERO weights, bias ZERO → β = 0 for any embedding at init
      So at start: out = 1 * conv(x) + 0 = conv(x)  (identical to SRResNet).
      The network learns to use conditioning gradually via backpropagation.
    """

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()

        # Standard 2-conv residual body (identical to SRResNet's ResidualBlock)
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
        )

        # FiLM predictors: embedding → per-channel scale and shift
        # One Linear per block so each block can modulate independently.
        self.gamma_fc = nn.Linear(embed_dim, num_features)
        self.beta_fc  = nn.Linear(embed_dim, num_features)

        # Identity initialization: conditioning starts as a no-op
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)    # γ = 1  at init
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)    # β = 0  at init

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, H, W)  feature maps from the previous block
        emb : (B, embed_dim) degradation embedding from MetadataEncoder
        """
        # Residual path
        out = self.conv_block(x)   # (B, C, H, W)

        # FiLM modulation — reshape to (B, C, 1, 1) for broadcast over H×W
        gamma = self.gamma_fc(emb).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = self.beta_fc(emb).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        out = gamma * out + beta   # per-channel affine transform

        return x + out             # residual skip (add original input back)


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution block — doubles spatial resolution via PixelShuffle."""

    def __init__(self, num_features: int, scale_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale_factor ** 2,
                      kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main model ─────────────────────────────────────────────────────────────────

class ConditionedSRResNet(nn.Module):
    """
    Degradation-aware SRResNet conditioned on degradation metadata via FiLM.

    Takes TWO inputs:
      x    : LR image tensor        (B, 3, H, W)          values in [0, 1]
      cond : conditioning vector    (B, COND_DIM)          values in [0, 1]
             built by cond_utils.build_cond_vector()

    Returns:
      SR image tensor  (B, 3, H×scale, W×scale)  clamped to [0, 1]

    Attributes:
      expects_upsampled_input = False   — receives raw LR (not bicubic pre-upsampled)
      expects_cond_vector     = True    — train.py reads this to pass conditioning

    Args:
      scale          : upscale factor (2, 4, or 8)
      num_res_blocks : FiLM-conditioned residual blocks (default 8)
      num_features   : channels throughout the network (default 64)
      cond_dim       : conditioning vector length (default COND_DIM = 13)
      embed_dim      : metadata embedding size (default 64)
    """

    expects_upsampled_input = False   # receives raw LR; upsamples internally
    expects_cond_vector     = True    # train.py passes (lr, cond) instead of just lr

    def __init__(
        self,
        scale: int = 4,
        num_res_blocks: int = 8,
        num_features: int = 64,
        cond_dim: int = COND_DIM,
        embed_dim: int = 64,
    ):
        super().__init__()

        if scale not in (2, 4, 8):
            raise ValueError(f"scale must be 2, 4, or 8; got {scale}")

        # ── Metadata encoder ──────────────────────────────────────────────────
        # Converts the 13-d conditioning vector into a dense embedding.
        # This embedding is shared across all FiLMResidualBlocks.
        self.encoder = MetadataEncoder(cond_dim=cond_dim, embed_dim=embed_dim)

        # ── Head: initial feature extraction ──────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # ── Body: N FiLM-conditioned residual blocks ──────────────────────────
        # ModuleList instead of Sequential — each block needs the embedding arg.
        self.body_blocks = nn.ModuleList([
            FiLMResidualBlock(num_features=num_features, embed_dim=embed_dim)
            for _ in range(num_res_blocks)
        ])

        # Post-body conv merges all block outputs before the global skip.
        self.post_body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
        )

        # ── Upsample: sub-pixel convolution ───────────────────────────────────
        num_up = int(math.log2(scale))
        self.upsample = nn.Sequential(
            *[UpsampleBlock(num_features, scale_factor=2) for _ in range(num_up)]
        )

        # ── Tail: final RGB reconstruction ────────────────────────────────────
        self.tail = nn.Conv2d(num_features, 3, kernel_size=9, padding=4)

        # Apply standard weight init to everything EXCEPT the identity-init FiLM layers
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv2d; ones/zeros for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Note: FiLMResidualBlock.__init__ already sets identity init on
        # gamma_fc and beta_fc — _init_weights does NOT re-init those.

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x    : (B, 3, H, W)         raw LR image,             values in [0, 1]
        cond : (B, COND_DIM)        normalized degradation vector from build_cond_vector()

        Returns (B, 3, H×scale, W×scale) clamped to [0, 1].
        """
        # 1. Encode degradation metadata into a dense embedding
        emb = self.encoder(cond)             # (B, embed_dim)

        # 2. Extract initial features from LR image
        head_out = self.head(x)              # (B, C, H, W)

        # 3. Pass through FiLM-conditioned residual blocks
        feat = head_out
        for block in self.body_blocks:
            feat = block(feat, emb)          # each block modulates its features with emb

        # 4. Post-body merge + global residual skip (same as SRResNet)
        feat    = self.post_body(feat)       # (B, C, H, W)
        merged  = head_out + feat            # global skip: add head output back

        # 5. Upsample and reconstruct
        up  = self.upsample(merged)          # (B, C, H×scale, W×scale)
        out = self.tail(up)                  # (B, 3, H×scale, W×scale)

        return torch.clamp(out, 0.0, 1.0)


# ── Self-test ──────────────────────────────────────────────────────────────────
# Run: python scripts/models/conditioned_sr.py

if __name__ == "__main__":
    import torch.nn as nn

    print("=" * 60)
    print("ConditionedSRResNet self-test")
    print("=" * 60)

    scale     = 4
    B, H, W   = 2, 48, 48           # batch of 2, LR patch size 48×48
    H_hr, W_hr = H * scale, W * scale

    model = ConditionedSRResNet(
        scale=scale, num_res_blocks=8, num_features=64, embed_dim=64
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters : {num_params:,}")

    lr   = torch.rand(B, 3, H, W)                # LR image batch
    cond = torch.rand(B, COND_DIM)               # random conditioning vectors

    # ── Test 1: Shape ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sr = model(lr, cond)
    assert tuple(sr.shape) == (B, 3, H_hr, W_hr), \
        f"Shape mismatch! Expected {(B, 3, H_hr, W_hr)}, got {tuple(sr.shape)}"
    assert sr.min() >= 0.0 and sr.max() <= 1.0, "Output out of [0,1]"
    print(f"\nTest 1 - Shape:  {tuple(lr.shape)} + cond({COND_DIM},) -> {tuple(sr.shape)}  PASSED")
    print(f"Test 1 - Range:  [{sr.min():.4f}, {sr.max():.4f}]  PASSED")

    # ── Test 2: Conditioning is actually used ─────────────────────────────────
    # The identity initialization (γ=1, β=0) makes FiLM a no-op at init.
    # To test that conditioning CAN change the output, we break identity init
    # on one block by giving gamma_fc a non-zero weight, then compare two
    # different conditioning vectors.
    model2 = ConditionedSRResNet(scale=scale, num_res_blocks=8, num_features=64)
    nn.init.normal_(model2.body_blocks[0].gamma_fc.weight, std=0.5)

    cond_a = torch.zeros(1, COND_DIM)   # all zeros = no degradation
    cond_b = torch.ones(1, COND_DIM)    # all ones  = maximum degradation

    model2.eval()
    with torch.no_grad():
        lr_single = torch.rand(1, 3, H, W)
        out_a = model2(lr_single, cond_a)
        out_b = model2(lr_single, cond_b)

    diff = (out_a - out_b).abs().mean().item()
    assert diff > 1e-5, \
        f"Conditioning has no effect on output! (mean diff = {diff:.2e})"
    print(f"\nTest 2 - Conditioning effect:  mean |out_a - out_b| = {diff:.4f}  PASSED")

    # ── Test 3: Attributes ────────────────────────────────────────────────────
    assert ConditionedSRResNet.expects_upsampled_input is False
    assert ConditionedSRResNet.expects_cond_vector is True
    print(f"\nTest 3 - Attributes:  expects_upsampled_input=False  expects_cond_vector=True  PASSED")

    # ── Test 4: MetadataEncoder output shape ──────────────────────────────────
    with torch.no_grad():
        emb = model.encoder(cond)
    assert tuple(emb.shape) == (B, 64), f"Encoder output shape wrong: {emb.shape}"
    print(f"\nTest 4 - MetadataEncoder:  cond({B},{COND_DIM}) -> emb({B},64)  PASSED")

    print(f"\n{'='*60}")
    print("conditioned_sr.py is working correctly.")
    print(f"{'='*60}")