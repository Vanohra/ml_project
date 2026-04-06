"""
models/__init__.py
------------------
Model registry for the super-resolution project.

Supported models
  srcnn    — SimpleSRCNN   (~57K params)  fast, good for quick experiments
  srresnet — SRResNet      (~957K params)  deeper, higher PSNR ceiling

Usage in train.py:
  from models import build_model
  model = build_model(cfg)          # cfg = YAML config dict

Each model class exposes one extra attribute that the training loop reads:

  model.expects_upsampled_input
    True  → the model expects a bicubic-upsampled LR image at HR spatial size
             (SRCNN style: upsample first, then refine)
    False → the model expects a raw LR image; it handles upsampling internally
             (SRResNet style: process at LR resolution, then sub-pixel upsample)
"""

import sys
from pathlib import Path

# Ensure scripts/ is in sys.path so we can import the sibling model.py file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import SimpleSRCNN                    # scripts/model.py  (unchanged)
from .srresnet import SRResNet                   # scripts/models/srresnet.py
from .conditioned_sr import ConditionedSRResNet  # scripts/models/conditioned_sr.py

# Attach the flag to SimpleSRCNN without modifying model.py
SimpleSRCNN.expects_upsampled_input = True   # needs bicubic pre-upsample
# SRResNet and ConditionedSRResNet declare their flags as class attributes.

SUPPORTED_MODELS = ["srcnn", "srresnet", "conditioned_srresnet"]


def build_model(cfg: dict):
    """
    Instantiates and returns the model specified in the config dict.

    Reads:
      cfg["model"]["name"]            — "srcnn", "srresnet", or "conditioned_srresnet"
      cfg["model"]["num_res_blocks"]  — SRResNet / ConditionedSRResNet (default 8)
      cfg["model"]["num_features"]    — feature channels (default 64)
      cfg["model"]["embed_dim"]       — ConditionedSRResNet only (default 64)
      cfg["training"]["scale"]        — used by both ResNet models to build upsamplers

    Returns an nn.Module with .expects_upsampled_input and optionally
    .expects_cond_vector bool attributes read by train.py.
    Raises ValueError if the model name is not recognised.
    """
    model_cfg  = cfg.get("model", {})
    model_name = model_cfg.get("name", "srcnn").lower().strip()

    if model_name == "srcnn":
        return SimpleSRCNN()

    elif model_name == "srresnet":
        scale        = cfg["training"]["scale"]
        num_blocks   = model_cfg.get("num_res_blocks", 8)
        num_features = model_cfg.get("num_features", 64)
        return SRResNet(scale=scale, num_res_blocks=num_blocks,
                        num_features=num_features)

    elif model_name == "conditioned_srresnet":
        scale        = cfg["training"]["scale"]
        num_blocks   = model_cfg.get("num_res_blocks", 8)
        num_features = model_cfg.get("num_features", 64)
        embed_dim    = model_cfg.get("embed_dim", 64)
        return ConditionedSRResNet(scale=scale, num_res_blocks=num_blocks,
                                   num_features=num_features, embed_dim=embed_dim)

    else:
        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            f"Supported options: {SUPPORTED_MODELS}"
        )
