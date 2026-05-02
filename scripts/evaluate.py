"""
evaluate.py
-----------
Comprehensive evaluation of all three super-resolution methods:
  1. Bicubic baseline (no ML)
  2. SimpleSRCNN   (~57k params, trained with train.py)
  3. SRResNet      (~957k params, trained with train_srresnet.py)

For each method, computes PSNR and SSIM on the validation set, prints a
comparison table, saves 5 visual comparison grids (LR | Bicubic | SRCNN |
SRResNet | HR), and writes a CSV of results.

Usage (from ml_project/):
  python scripts/evaluate.py
  python scripts/evaluate.py \\
      --srcnn_ckpt   outputs/checkpoints/best.pth \\
      --srresnet_ckpt outputs/checkpoints_srresnet/best.pth \\
      --hr_dir        data/DIV2K/HR_valid \\
      --n_samples 5

Models are skipped gracefully if their checkpoint does not exist.
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from degradation import get_domain_preset, degrade
from metrics import compute_psnr, compute_ssim
from model import SimpleSRCNN, SRResNet

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_srcnn(ckpt_path: Path, device: torch.device):
    """Loads SimpleSRCNN from a checkpoint. Returns None if path missing."""
    if not ckpt_path.exists():
        print(f"  [skip] SimpleSRCNN checkpoint not found: {ckpt_path}")
        return None
    model = SimpleSRCNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Checkpoint may be a bare state_dict or a dict with a "model" key
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval()
    ep = state.get("epoch", "?")
    psnr = state.get("psnr", float("nan"))
    print(f"  [loaded] SimpleSRCNN from {ckpt_path.name}  "
          f"(epoch {ep}, saved PSNR {psnr:.2f} dB)")
    return model


def load_srresnet(ckpt_path: Path, device: torch.device, scale: int = 4):
    """Loads SRResNet from a checkpoint. Returns None if path missing."""
    if not ckpt_path.exists():
        print(f"  [skip] SRResNet checkpoint not found: {ckpt_path}")
        return None
    model = SRResNet(scale=scale).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model", state)
    model.load_state_dict(sd)
    model.eval()
    ep = state.get("epoch", "?")
    psnr = state.get("psnr", float("nan"))
    print(f"  [loaded] SRResNet from {ckpt_path.name}  "
          f"(epoch {ep}, saved PSNR {psnr:.2f} dB)")
    return model


# ── Image utilities ────────────────────────────────────────────────────────────

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image → FloatTensor (1, C, H, W) in [0, 1]."""
    return to_tensor(img).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, C, H, W) FloatTensor in [0, 1] → PIL Image."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate_all(
    hr_paths: list,
    srcnn_model,
    srresnet_model,
    device: torch.device,
    scale: int,
    surv_cfg: dict,
    seed: int = 42,
) -> dict:
    """
    Runs all three methods on every HR image and returns aggregated scores.

    LR images are generated on-the-fly using the surveillance preset so all
    methods see the same degraded inputs (same seed → same degradation each run).

    Returns a dict:
      {
        "bicubic":  {"psnr": [...], "ssim": [...]},
        "srcnn":    {"psnr": [...], "ssim": [...]},
        "srresnet": {"psnr": [...], "ssim": [...]},
      }
    Keys for srcnn / srresnet are only present if the model was loaded.
    """
    results = {
        "bicubic":  {"psnr": [], "ssim": []},
    }
    if srcnn_model is not None:
        results["srcnn"] = {"psnr": [], "ssim": []}
    if srresnet_model is not None:
        results["srresnet"] = {"psnr": [], "ssim": []}

    random.seed(seed)
    np.random.seed(seed)

    for i, hr_path in enumerate(hr_paths, 1):
        hr_pil = Image.open(hr_path).convert("RGB")

        # Crop to multiples of scale so sizes are exact
        w = (hr_pil.width  // scale) * scale
        h = (hr_pil.height // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))

        # Generate LR using surveillance preset
        lr_pil = degrade(hr_pil, config=surv_cfg)

        hr_t = pil_to_tensor(hr_pil).to(device)
        lr_t = pil_to_tensor(lr_pil).to(device)

        # ── Bicubic ──────────────────────────────────────────────────────────
        bicubic_t = F.interpolate(lr_t, scale_factor=scale, mode="bicubic",
                                  align_corners=False).clamp(0, 1)
        results["bicubic"]["psnr"].append(compute_psnr(bicubic_t, hr_t))
        results["bicubic"]["ssim"].append(compute_ssim(bicubic_t, hr_t))

        # ── SimpleSRCNN ───────────────────────────────────────────────────────
        if srcnn_model is not None:
            with torch.no_grad():
                lr_up = F.interpolate(lr_t, scale_factor=scale, mode="bicubic",
                                      align_corners=False)
                sr_srcnn = srcnn_model(lr_up).clamp(0, 1)
            results["srcnn"]["psnr"].append(compute_psnr(sr_srcnn, hr_t))
            results["srcnn"]["ssim"].append(compute_ssim(sr_srcnn, hr_t))

        # ── SRResNet ──────────────────────────────────────────────────────────
        if srresnet_model is not None:
            with torch.no_grad():
                sr_res = srresnet_model(lr_t).clamp(0, 1)
            results["srresnet"]["psnr"].append(compute_psnr(sr_res, hr_t))
            results["srresnet"]["ssim"].append(compute_ssim(sr_res, hr_t))

        print(f"  [{i:3d}/{len(hr_paths)}] {hr_path.name}")

    return results


# ── Visual grid saving ────────────────────────────────────────────────────────

def save_visual_grids(
    hr_paths: list,
    srcnn_model,
    srresnet_model,
    device: torch.device,
    scale: int,
    surv_cfg: dict,
    out_dir: Path,
    n_samples: int,
    seed: int = 42,
) -> None:
    """
    Saves n_samples comparison grids.

    Each grid shows (side by side, labelled):
        LR input | Bicubic | SimpleSRCNN | SRResNet | HR ground truth
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [warn] matplotlib not installed — visual grids skipped")
        print("         pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len(hr_paths)))
    if len(indices) > n_samples:
        # Evenly spaced sample indices
        step = len(indices) / n_samples
        indices = [int(i * step) for i in range(n_samples)]

    LABELS = ["LR input", "Bicubic", "SimpleSRCNN", "SRResNet", "HR (GT)"]
    MAX_HEIGHT = 256   # resize tall images for manageable file sizes

    for grid_idx, img_idx in enumerate(indices):
        hr_path = hr_paths[img_idx]
        hr_pil  = Image.open(hr_path).convert("RGB")

        w = (hr_pil.width  // scale) * scale
        h = (hr_pil.height // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))
        lr_pil = degrade(hr_pil, config=surv_cfg)

        hr_t = pil_to_tensor(hr_pil).to(device)
        lr_t = pil_to_tensor(lr_pil).to(device)

        bicubic_t = F.interpolate(lr_t, scale_factor=scale, mode="bicubic",
                                  align_corners=False).clamp(0, 1)
        lr_disp   = F.interpolate(lr_t, scale_factor=scale,
                                  mode="nearest").clamp(0, 1)

        panels_t = [lr_disp, bicubic_t]

        if srcnn_model is not None:
            with torch.no_grad():
                lr_up = F.interpolate(lr_t, scale_factor=scale, mode="bicubic",
                                      align_corners=False)
                sr_srcnn = srcnn_model(lr_up).clamp(0, 1)
            panels_t.append(sr_srcnn)
        else:
            panels_t.append(bicubic_t)   # placeholder = bicubic

        if srresnet_model is not None:
            with torch.no_grad():
                sr_res = srresnet_model(lr_t).clamp(0, 1)
            panels_t.append(sr_res)
        else:
            panels_t.append(bicubic_t)   # placeholder

        panels_t.append(hr_t)

        panels_pil = [tensor_to_pil(t) for t in panels_t]

        # Resize for display if image is tall
        if panels_pil[0].height > MAX_HEIGHT:
            ratio = MAX_HEIGHT / panels_pil[0].height
            new_w = int(panels_pil[0].width * ratio)
            new_h = MAX_HEIGHT
            panels_pil = [p.resize((new_w, new_h), Image.BICUBIC)
                          for p in panels_pil]

        n_panels = len(panels_pil)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(4 * n_panels, 4))
        for ax, img, label in zip(axes, panels_pil, LABELS):
            ax.imshow(np.array(img))
            ax.set_title(label, fontsize=9, pad=3)
            ax.axis("off")

        fig.suptitle(f"Sample {grid_idx + 1}: {hr_path.name}",
                     fontsize=10, y=1.01)
        plt.tight_layout()

        out_path = out_dir / f"sample_{grid_idx + 1:02d}_{hr_path.stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [grid] Saved: {out_path.name}")


# ── Results table and CSV ─────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    """Prints a formatted comparison table to stdout."""
    _DISPLAY = {
        "bicubic":  "Bicubic",
        "srcnn":    "SimpleSRCNN",
        "srresnet": "SRResNet",
    }
    order = ["bicubic", "srcnn", "srresnet"]

    print("\n" + "=" * 55)
    print("  === Super-Resolution Evaluation Results ===")
    print("=" * 55)
    print(f"  {'Method':<16} {'PSNR (dB)':>10}    {'SSIM':>8}")
    print("  " + "-" * 40)
    for key in order:
        if key not in results:
            continue
        psnrs = results[key]["psnr"]
        ssims = results[key]["ssim"]
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_ssim = sum(ssims) / len(ssims)
        label = _DISPLAY[key]
        tag = ""
        if key in ("srcnn", "srresnet") and psnrs:
            # Check if model actually improved over bicubic
            bic_psnr = sum(results["bicubic"]["psnr"]) / len(results["bicubic"]["psnr"])
            if avg_psnr < bic_psnr:
                tag = "  (not yet trained?)"
        print(f"  {label:<16} {avg_psnr:>10.2f}    {avg_ssim:>8.4f}{tag}")
    print("=" * 55)
    print("  (PSNR higher = better | SSIM higher = better)")
    print("=" * 55)


def save_csv(results: dict, out_path: Path) -> None:
    """Writes per-image and mean results to a CSV file."""
    _DISPLAY = {
        "bicubic":  "Bicubic",
        "srcnn":    "SimpleSRCNN",
        "srresnet": "SRResNet",
    }
    order = ["bicubic", "srcnn", "srresnet"]

    rows = []
    # Per-image rows (using image indices since we don't pass paths here)
    n_images = len(next(iter(results.values()))["psnr"])
    for i in range(n_images):
        row = {"image_index": i + 1}
        for key in order:
            if key not in results:
                continue
            row[f"{_DISPLAY[key]}_PSNR"] = round(results[key]["psnr"][i], 4)
            row[f"{_DISPLAY[key]}_SSIM"] = round(results[key]["ssim"][i], 6)
        rows.append(row)

    # Summary row
    summary = {"image_index": "MEAN"}
    for key in order:
        if key not in results:
            continue
        psnrs = results[key]["psnr"]
        ssims = results[key]["ssim"]
        summary[f"{_DISPLAY[key]}_PSNR"] = round(sum(psnrs) / len(psnrs), 4)
        summary[f"{_DISPLAY[key]}_SSIM"] = round(sum(ssims) / len(ssims), 6)
    rows.append(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results CSV: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bicubic / SimpleSRCNN / SRResNet on HR_valid."
    )
    parser.add_argument(
        "--srcnn_ckpt",
        default="outputs/checkpoints/best.pth",
        help="Path to SimpleSRCNN checkpoint (default: outputs/checkpoints/best.pth)",
    )
    parser.add_argument(
        "--srresnet_ckpt",
        default="outputs/checkpoints_srresnet/best.pth",
        help="Path to SRResNet checkpoint (default: outputs/checkpoints_srresnet/best.pth)",
    )
    parser.add_argument(
        "--hr_dir",
        default="data/DIV2K/HR_valid",
        help="Directory of HR validation images (default: data/DIV2K/HR_valid)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of visual comparison grids to save (default: 5)",
    )
    args = parser.parse_args()

    # ── Resolve paths relative to project root ────────────────────────────────
    def _resolve(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_DIR / p

    srcnn_ckpt   = _resolve(args.srcnn_ckpt)
    srresnet_ckpt = _resolve(args.srresnet_ckpt)
    hr_dir        = _resolve(args.hr_dir)

    out_dir      = PROJECT_DIR / "outputs" / "eval_samples"
    csv_out      = PROJECT_DIR / "outputs" / "eval_results.csv"
    SCALE        = 4
    SEED         = 42

    print("\n" + "=" * 55)
    print("  evaluate.py — Super-Resolution Comparison")
    print("=" * 55)

    # ── Collect HR images ─────────────────────────────────────────────────────
    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    if not hr_dir.exists():
        print(f"\n[error] HR directory not found: {hr_dir}")
        print("  Place validation HR images there first.")
        sys.exit(1)

    hr_paths = sorted([p for p in hr_dir.iterdir()
                       if p.suffix.lower() in SUPPORTED])
    if not hr_paths:
        print(f"\n[error] No images found in {hr_dir}")
        sys.exit(1)

    print(f"\n  HR images  : {len(hr_paths)} found in {hr_dir}")

    # ── Device and models ─────────────────────────────────────────────────────
    device = get_device()
    print(f"  Device     : {device}")

    print("\nLoading models...")
    srcnn_model    = load_srcnn(srcnn_ckpt, device)
    srresnet_model = load_srresnet(srresnet_ckpt, device, scale=SCALE)

    if srcnn_model is None and srresnet_model is None:
        print("\n  [note] Neither model checkpoint found. "
              "Evaluating bicubic baseline only.")

    # ── Surveillance degradation preset (same for all methods) ────────────────
    surv_cfg = get_domain_preset("surveillance")
    print(f"\n  Degradation: surveillance preset (seed={SEED})")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    print(f"\nEvaluating on {len(hr_paths)} image(s)...")
    results = evaluate_all(
        hr_paths, srcnn_model, srresnet_model,
        device, SCALE, surv_cfg, seed=SEED,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(results)

    # ── Save visual grids ─────────────────────────────────────────────────────
    print(f"\nSaving {args.n_samples} visual comparison grids -> {out_dir}")
    save_visual_grids(
        hr_paths, srcnn_model, srresnet_model,
        device, SCALE, surv_cfg, out_dir,
        n_samples=args.n_samples, seed=SEED,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(results, csv_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
