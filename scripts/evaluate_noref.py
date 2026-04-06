"""
evaluate_noref.py
-----------------
No-reference evaluation: assesses SR output quality WITHOUT ground truth.
Use this on real-world LR images (e.g. RealSRSet) where no HR exists.

Metrics computed
  NIQE    — Natural Image Quality Evaluator (lower = better).
            Measures distance from natural image statistics.
            Requires: piq   (pip install piq)
  BRISQUE — Blind/Referenceless Image Spatial QUality Evaluator (lower = better).
            Uses natural scene statistics on local patches.
            Requires: piq   (pip install piq)

The script evaluates the quality of:
  1. Bicubic upsampled SR  (always)
  2. Trained model SR      (if experiment and checkpoint are available)

Comparing both lets you measure how much quality the model adds over bicubic
on images you have never seen during training.

Outputs (saved to outputs/experiments/<name>/metrics/)
  eval_noref.csv            per-image scores for every method
  eval_noref_summary.json   mean ± std for every method

Optional: pass --save_sr to also save the SR images for visual inspection.
SR images are saved to outputs/experiments/<name>/eval_noref_sr/.

Usage examples
  # Evaluate bicubic upsampling quality only:
  python scripts/evaluate_noref.py \\
      --experiment baseline_srcnn \\
      --img_dir data/RealSRSet \\
      --bicubic_only

  # Evaluate trained model vs bicubic on real images:
  python scripts/evaluate_noref.py \\
      --experiment baseline_srcnn \\
      --img_dir data/RealSRSet

  # Save SR outputs for visual inspection:
  python scripts/evaluate_noref.py \\
      --experiment srresnet_run \\
      --img_dir data/RealSRSet \\
      --save_sr

Notes on metrics
  NIQE   : Requires images >= 64 × 64. Very small images are skipped.
  BRISQUE: May return NaN on atypical inputs (handled gracefully).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
NIQE_MIN_SIZE  = 64   # images smaller than this are skipped for NIQE

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import piq as _piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("[warn] piq not found — NIQE and BRISQUE will not be computed.")
    print("       Install with: pip install piq")


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_niqe(img: torch.Tensor) -> float | None:
    """
    NIQE score for a (1, 3, H, W) float tensor in [0, 1].
    Returns None if the image is too small or piq raises an error.
    Lower is better.
    """
    if not HAS_PIQ:
        return None
    H, W = img.shape[2], img.shape[3]
    if H < NIQE_MIN_SIZE or W < NIQE_MIN_SIZE:
        return None
    try:
        return float(_piq.niqe(img, data_range=1.0).item())
    except Exception as e:
        print(f"    [warn] NIQE failed: {e}")
        return None


def compute_brisque(img: torch.Tensor) -> float | None:
    """
    BRISQUE score for a (1, 3, H, W) float tensor in [0, 1].
    Returns None if piq is not available or the computation fails.
    Lower is better.
    """
    if not HAS_PIQ:
        return None
    try:
        val = float(_piq.brisque(img, data_range=1.0).item())
        return val if np.isfinite(val) else None
    except Exception as e:
        print(f"    [warn] BRISQUE failed: {e}")
        return None


# ── Model loading (shared with evaluate_paired.py) ─────────────────────────────

def load_model(exp_dir: Path, config_path, checkpoint_name: str,
               device: torch.device):
    """Loads model from experiment directory or explicit config."""
    from models import build_model

    if config_path is None:
        config_path = exp_dir / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}\n"
            "  Pass --config explicitly, or use --bicubic_only to skip the model."
        )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = exp_dir / "checkpoints" / f"{checkpoint_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    print(f"  Loaded {ckpt_path.name}"
          f"  (epoch {state.get('epoch', '?')}, "
          f"val PSNR {state.get('psnr', 0):.2f} dB)")
    return model, cfg


# ── Per-image evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_image(img_path: Path, scale: int, model, device: torch.device,
               has_cond: bool, save_dir: Path | None) -> dict:
    """
    Runs SR on one LR image and computes no-reference metrics on the result.

    Returns a dict of {metric_method: value} pairs.
    If save_dir is given, the SR images are saved there.
    """
    lr_img = Image.open(img_path).convert("RGB")
    lr = to_tensor(lr_img).unsqueeze(0).to(device)   # (1, 3, h, w)
    h, w = lr.shape[2], lr.shape[3]
    H, W = h * scale, w * scale

    # Bicubic SR
    bicubic = F.interpolate(lr, size=(H, W), mode="bicubic",
                            align_corners=False).clamp(0, 1)

    row = {}

    # ── Bicubic NR metrics ────────────────────────────────────────────────────
    row["niqe_bicubic"]    = compute_niqe(bicubic)
    row["brisque_bicubic"] = compute_brisque(bicubic)

    # ── Model SR and NR metrics ───────────────────────────────────────────────
    if model is not None:
        if model.expects_upsampled_input:
            model_input = bicubic
        else:
            model_input = lr

        if has_cond:
            from cond_utils import COND_DIM
            cond = torch.zeros(1, COND_DIM, device=device)
            sr = model(model_input, cond)
        else:
            sr = model(model_input)

        row["niqe_model"]    = compute_niqe(sr)
        row["brisque_model"] = compute_brisque(sr)

        # Save SR images if requested
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_tensor(sr,      save_dir / f"{img_path.stem}_model_sr.png")
            _save_tensor(bicubic, save_dir / f"{img_path.stem}_bicubic_sr.png")

    return row


def _save_tensor(t: torch.Tensor, path: Path) -> None:
    """Save a (1, 3, H, W) float tensor in [0, 1] as a PNG."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255).clip(0, 255).astype("uint8")).save(path)


# ── Aggregate statistics ──────────────────────────────────────────────────────

def aggregate(rows: list[dict], metric: str) -> dict | None:
    vals = [r[metric] for r in rows
            if metric in r and r[metric] is not None and np.isfinite(r[metric])]
    if not vals:
        return None
    a = np.array(vals)
    return {"mean": round(float(a.mean()), 4),
            "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4),
            "max":  round(float(a.max()),  4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="No-reference SR evaluation: NIQE / BRISQUE (no HR ground truth needed)"
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (folder under outputs/experiments/)")
    parser.add_argument("--img_dir",    required=True,
                        help="Directory of LR input images (no ground truth needed)")
    parser.add_argument("--config",     default=None,
                        help="YAML config for model instantiation "
                             "(auto-discovered from experiment dir if omitted)")
    parser.add_argument("--checkpoint", default="best",
                        choices=["best", "last"],
                        help="Which checkpoint to load (default: best)")
    parser.add_argument("--scale",      type=int, default=4,
                        help="SR upscale factor (default: 4)")
    parser.add_argument("--bicubic_only", action="store_true",
                        help="Evaluate bicubic SR only (no model loaded)")
    parser.add_argument("--save_sr",    action="store_true",
                        help="Save SR images to eval_noref_sr/ inside the experiment dir")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    img_dir     = Path(args.img_dir)
    exp_dir     = project_dir / "outputs" / "experiments" / args.experiment
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_dir    = (exp_dir / "eval_noref_sr") if args.save_sr else None

    if not img_dir.exists():
        print(f"[error] Image directory not found: {img_dir}");  return

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if (hasattr(torch.backends, "mps") and
                                      torch.backends.mps.is_available())
              else torch.device("cpu"))
    print(f"Device : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, has_cond, scale = None, False, args.scale
    if not args.bicubic_only:
        try:
            model, cfg = load_model(exp_dir, args.config, args.checkpoint, device)
            has_cond   = getattr(model, "expects_cond_vector", False)
            scale      = cfg.get("training", {}).get("scale", args.scale)
            if has_cond:
                print("  [note] ConditionedSRResNet: using zero conditioning vector.")
        except FileNotFoundError as e:
            print(f"[warn] Could not load model:\n  {e}\n  Falling back to bicubic only.")

    # ── Image list ────────────────────────────────────────────────────────────
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        print(f"[error] No images found in {img_dir}");  return
    print(f"\nEvaluating {len(images)} images  (scale x{scale})...")

    # ── Per-image loop ────────────────────────────────────────────────────────
    rows = []
    for i, img_p in enumerate(images, 1):
        row = eval_image(img_p, scale, model, device, has_cond, save_dir)
        row = {"filename": img_p.name, **row}
        rows.append(row)

        niqe_b = f"{row.get('niqe_bicubic', float('nan')):.3f}" if row.get('niqe_bicubic') is not None else "n/a"
        niqe_m = f"{row.get('niqe_model',   float('nan')):.3f}" if row.get('niqe_model')   is not None else "n/a"
        print(f"  [{i:3d}/{len(images)}] {img_p.name:<24s}  "
              f"NIQE  bicubic {niqe_b}  model {niqe_m}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = metrics_dir / "eval_noref.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else
                            ("" if v is None else v))
                        for k, v in row.items()})
    print(f"\nPer-image CSV : {csv_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    metrics  = ["niqe", "brisque"]
    methods  = ["bicubic"] + (["model"] if model is not None else [])

    summary = {
        "experiment":   args.experiment,
        "checkpoint":   f"{args.checkpoint}.pth" if not args.bicubic_only else None,
        "img_dir":      str(img_dir),
        "num_images":   len(rows),
        "scale":        scale,
        "note":         "lower NIQE and BRISQUE = better perceptual quality",
    }
    for method in methods:
        summary[method] = {}
        for m in metrics:
            key = f"{m}_{method}"
            agg = aggregate(rows, key)
            if agg is not None:
                summary[method][m] = agg

    json_path = metrics_dir / "eval_noref_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON  : {json_path}")
    if save_dir:
        print(f"SR images     : {save_dir}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Experiment : {args.experiment}  |  Images : {len(rows)}")
    print(f"  {'Method':<12}  {'NIQE':>10}  {'BRISQUE':>10}")
    print(f"  {'-'*42}")
    for method in methods:
        d = summary.get(method, {})
        niqe_s    = f"{d['niqe']['mean']:.3f} ± {d['niqe']['std']:.3f}" \
                    if "niqe" in d else "n/a (piq not installed)"
        brisque_s = f"{d['brisque']['mean']:.2f}"  if "brisque" in d else "n/a"
        print(f"  {method:<12}  {niqe_s:>10}  {brisque_s:>10}")
    print(f"  (lower = better perceptual quality)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
