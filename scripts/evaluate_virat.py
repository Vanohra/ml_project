"""
evaluate_virat.py
-----------------
Evaluates a trained SR model on VIRAT surveillance frame pairs produced by
extract_virat_frames.py.  Works exactly like evaluate_paired.py but is
pre-configured for the VIRAT folder layout and enriches the CSV output
with per-frame provenance (source video, frame index, timestamp).

Metrics computed
  PSNR   — Peak Signal-to-Noise Ratio (dB, higher = better).
            Always available; no extra packages needed.
  SSIM   — Structural Similarity Index (0–1, higher = better).
            Requires: scikit-image   (pip install scikit-image)
  LPIPS  — Learned Perceptual Image Patch Similarity (0–1, lower = better).
            Requires: lpips          (pip install lpips)

Both the bicubic baseline and the trained model are evaluated in one pass.

Inputs (produced by extract_virat_frames.py)
  data/VIRAT/frames/quantitative/hr/   — clean extracted frames
  data/VIRAT/frames/quantitative/lr/   — surveillance-degraded counterparts
  data/VIRAT/frames/manifest.json      — provenance for each frame pair

Outputs (saved to outputs/experiments/<name>/metrics/)
  eval_virat.csv              per-frame PSNR / SSIM / LPIPS  +  video/frame metadata
  eval_virat_summary.json     mean ± std / min / max for each metric and method

Usage examples
  # Evaluate trained model vs bicubic:
  python scripts/evaluate_virat.py \\
      --experiment srresnet_run

  # Bicubic baseline only:
  python scripts/evaluate_virat.py \\
      --experiment srresnet_run \\
      --bicubic_only

  # Evaluate the preference split instead of quantitative:
  python scripts/evaluate_virat.py \\
      --experiment srresnet_run \\
      --split preference

  # Explicit config and last checkpoint:
  python scripts/evaluate_virat.py \\
      --experiment srresnet_run \\
      --config configs/srresnet_run.yaml \\
      --checkpoint last
"""

import argparse
import csv
import json
import math
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

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    from skimage.metrics import structural_similarity as _ski_ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[warn] scikit-image not found — SSIM will not be computed.")
    print("       Install with: pip install scikit-image")

try:
    import lpips as _lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[warn] lpips not found — LPIPS will not be computed.")
    print("       Install with: pip install lpips")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr).item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return float(_ski_ssim(hr_np, sr_np, data_range=1.0, channel_axis=2))


def compute_lpips(sr: torch.Tensor, hr: torch.Tensor, fn) -> float:
    return fn(sr * 2 - 1, hr * 2 - 1).item()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(exp_dir: Path, config_path, checkpoint_name: str,
               device: torch.device):
    from models import build_model

    if config_path is None:
        config_path = exp_dir / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}\n"
            "  Pass --config explicitly, or use --bicubic_only to skip the model."
        )

    cfg   = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = exp_dir / "checkpoints" / f"{checkpoint_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Available: {[p.name for p in (exp_dir / 'checkpoints').glob('*.pth')]}"
        )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"  Loaded {ckpt_path.name}"
          f"  (epoch {state.get('epoch', '?')}, "
          f"val PSNR {state.get('psnr', 0):.2f} dB)")
    return model, cfg


# ── Per-image evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_pair(lr_path: Path, hr_path: Path, scale: int,
              model, lpips_fn, device: torch.device, has_cond: bool) -> dict:
    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    hr = to_tensor(hr_img).unsqueeze(0).to(device)
    lr = to_tensor(lr_img).unsqueeze(0).to(device)
    H, W = hr.shape[2], hr.shape[3]

    bicubic = F.interpolate(lr, size=(H, W), mode="bicubic",
                            align_corners=False).clamp(0, 1)
    row = {}

    row["psnr_bicubic"] = compute_psnr(bicubic, hr)
    if HAS_SSIM:
        row["ssim_bicubic"] = compute_ssim(bicubic, hr)
    if HAS_LPIPS and lpips_fn is not None:
        row["lpips_bicubic"] = compute_lpips(bicubic, hr, lpips_fn)

    if model is not None:
        model_input = bicubic if model.expects_upsampled_input else lr

        if has_cond:
            from cond_utils import COND_DIM
            cond = torch.zeros(1, COND_DIM, device=device)
            sr   = model(model_input, cond).clamp(0, 1)
        else:
            sr = model(model_input).clamp(0, 1)

        row["psnr_model"]  = compute_psnr(sr, hr)
        if HAS_SSIM:
            row["ssim_model"]  = compute_ssim(sr, hr)
        if HAS_LPIPS and lpips_fn is not None:
            row["lpips_model"] = compute_lpips(sr, hr, lpips_fn)

    return row


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(rows: list[dict], metric: str) -> dict | None:
    vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
    if not vals:
        return None
    a = np.array(vals)
    return {"mean": round(float(a.mean()), 4),
            "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4),
            "max":  round(float(a.max()),  4)}


# ── Manifest lookup ────────────────────────────────────────────────────────────

def load_manifest(frames_dir: Path) -> dict:
    """
    Return {filename -> manifest_entry} for quick lookup during CSV writing.
    Filename is the HR filename (e.g. '0042_hr.png').
    Returns empty dict if manifest.json is not found.
    """
    manifest_path = frames_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {Path(f["hr_path"]).name: f for f in data.get("frames", [])}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SR model on VIRAT surveillance frame pairs"
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (folder under outputs/experiments/)")
    parser.add_argument("--split",      default="quantitative",
                        choices=["quantitative", "preference"],
                        help="Which frame split to evaluate (default: quantitative)")
    parser.add_argument("--frames_dir", default="data/VIRAT/frames",
                        help="Root VIRAT frames directory (default: data/VIRAT/frames)")
    parser.add_argument("--config",     default=None,
                        help="YAML config path (auto-discovered from experiment dir if omitted)")
    parser.add_argument("--checkpoint", default="best",
                        choices=["best", "last"],
                        help="Which checkpoint to load (default: best)")
    parser.add_argument("--scale",      type=int, default=4,
                        help="SR upscale factor (default: 4; overridden by config if found)")
    parser.add_argument("--bicubic_only", action="store_true",
                        help="Evaluate bicubic baseline only (no model loaded)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    frames_dir  = (project_dir / args.frames_dir
                   if not Path(args.frames_dir).is_absolute()
                   else Path(args.frames_dir))
    hr_dir      = frames_dir / args.split / "hr"
    lr_dir      = frames_dir / args.split / "lr"
    exp_dir     = project_dir / "outputs" / "experiments" / args.experiment
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for d, label in [(hr_dir, "HR"), (lr_dir, "LR")]:
        if not d.exists():
            print(f"[error] {label} directory not found: {d}")
            print(f"  Run extract_virat_frames.py first.")
            return

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if (hasattr(torch.backends, "mps") and
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
                print("  [note] ConditionedSRResNet: zero conditioning vector for eval.")
        except FileNotFoundError as e:
            print(f"[warn] Could not load model:\n  {e}\n  Falling back to bicubic only.")

    # ── LPIPS network ─────────────────────────────────────────────────────────
    lpips_fn = None
    if HAS_LPIPS:
        print("Loading LPIPS network (AlexNet)...")
        lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    # ── Image pairs ───────────────────────────────────────────────────────────
    hr_files = sorted(p for p in hr_dir.iterdir()
                      if p.suffix.lower() in SUPPORTED_EXTS)
    if not hr_files:
        print(f"[error] No images found in {hr_dir}");  return

    # Manifest lookup — adds video / frame_idx / timestamp_s to CSV rows
    manifest_lookup = load_manifest(frames_dir)

    print(f"\nEvaluating {len(hr_files)} frame pairs  "
          f"[split: {args.split}]  (scale x{scale})...")

    # ── Per-image loop ─────────────────────────────────────────────────────────
    rows = []
    for i, hr_p in enumerate(hr_files, 1):
        lr_p = lr_dir / hr_p.name.replace("_hr.", "_lr.")
        if not lr_p.exists():
            print(f"  [warn] No LR match for {hr_p.name} — skipping.")
            continue

        metrics = eval_pair(lr_p, hr_p, scale, model, lpips_fn, device, has_cond)

        # Provenance from manifest
        meta = manifest_lookup.get(hr_p.name, {})
        row  = {
            "filename":    hr_p.name,
            "global_id":   meta.get("global_id", ""),
            "video":       meta.get("video",      ""),
            "frame_idx":   meta.get("frame_idx",  ""),
            "timestamp_s": meta.get("timestamp_s",""),
            **metrics,
        }
        rows.append(row)

        psnr_b = f"{metrics.get('psnr_bicubic', 0):.2f}"
        psnr_m = (f"{metrics.get('psnr_model', 0):.2f}"
                  if "psnr_model" in metrics else "n/a")
        print(f"  [{i:3d}/{len(hr_files)}] {hr_p.name:<22s}  "
              f"bicubic {psnr_b} dB  model {psnr_m} dB")

    if not rows:
        print("[error] No frame pairs evaluated.");  return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = metrics_dir / "eval_virat.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nPer-frame CSV  : {csv_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    metrics_list = ["psnr", "ssim", "lpips"]
    methods      = ["bicubic"] + (["model"] if model is not None else [])

    summary = {
        "experiment":  args.experiment,
        "checkpoint":  None if args.bicubic_only else f"{args.checkpoint}.pth",
        "split":       args.split,
        "num_frames":  len(rows),
        "scale":       scale,
        "frames_dir":  str(frames_dir),
    }
    for method in methods:
        summary[method] = {}
        for m in metrics_list:
            agg = aggregate(rows, f"{m}_{method}")
            if agg is not None:
                summary[method][m] = agg

    json_path = metrics_dir / "eval_virat_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON   : {json_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Experiment : {args.experiment}  |  Split : {args.split}  "
          f"|  Frames : {len(rows)}")
    print(f"  {'Method':<12}  {'PSNR (dB)':>12}  {'SSIM':>8}  {'LPIPS':>8}")
    print(f"  {'-'*54}")
    for method in methods:
        d      = summary.get(method, {})
        psnr_s = (f"{d['psnr']['mean']:.2f} ± {d['psnr']['std']:.2f}"
                  if "psnr" in d else "n/a")
        ssim_s = f"{d['ssim']['mean']:.4f}"  if "ssim"  in d else "n/a"
        lpi_s  = f"{d['lpips']['mean']:.4f}" if "lpips" in d else "n/a"
        print(f"  {method:<12}  {psnr_s:>12}  {ssim_s:>8}  {lpi_s:>8}")
    print(f"  (PSNR/SSIM higher = better   LPIPS lower = better)")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
