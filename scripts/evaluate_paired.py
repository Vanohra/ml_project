"""
evaluate_paired.py
------------------
Evaluates super-resolution quality against ground truth HR images.
Use this when you have matched LR / HR pairs (e.g. DIV2K validation set).

Metrics computed
  PSNR   — Peak Signal-to-Noise Ratio (dB, higher = better).
            Always available; requires no extra packages.
  SSIM   — Structural Similarity Index (0–1, higher = better).
            Requires: scikit-image   (pip install scikit-image)
  LPIPS  — Learned Perceptual Image Patch Similarity (0–1, lower = better).
            Correlates well with human judgement of perceptual quality.
            Requires: lpips          (pip install lpips)
            Note: downloads AlexNet weights (~9 MB) on first run.

Both bicubic baseline and trained-model outputs are evaluated in one pass
so the table is directly comparable.

Conditioning for ConditionedSRResNet
  --regen_lr  (recommended for conditioned models)
    Re-degrades each HR image using the domain from the training config.
    Captures the exact degradation parameters → passes the actual conditioning
    vector to the model.  This is the most faithful evaluation for the
    ConditionedSRResNet proposal.  --lr_dir is not required in this mode.

  Without --regen_lr (default)
    Loads LR images from --lr_dir.  For conditioned models a zero conditioning
    vector is used (fallback).  Results are still valid but the model cannot
    use its conditioning head.

Outputs (saved to outputs/experiments/<name>/metrics/)
  eval_paired.csv           per-image metrics for every method
  eval_paired_summary.json  mean ± std for every method

Usage examples
  # Evaluate bicubic baseline only (no model needed):
  python scripts/evaluate_paired.py \\
      --experiment baseline_srcnn \\
      --hr_dir data/DIV2K/HR_valid \\
      --lr_dir data/DIV2K/LR_valid \\
      --bicubic_only

  # Evaluate trained model (auto-discovers config from experiment dir):
  python scripts/evaluate_paired.py \\
      --experiment baseline_srcnn \\
      --hr_dir data/DIV2K/HR_valid \\
      --lr_dir data/DIV2K/LR_valid

  # ConditionedSRResNet — true conditioning via re-degradation from HR:
  python scripts/evaluate_paired.py \\
      --experiment exp3_final \\
      --hr_dir data/DIV2K/HR_valid \\
      --regen_lr

  # Explicit config and a specific checkpoint:
  python scripts/evaluate_paired.py \\
      --experiment srresnet_run \\
      --hr_dir data/DIV2K/HR_valid \\
      --lr_dir data/DIV2K/LR_valid \\
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

# Add scripts/ to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ── Optional package imports ──────────────────────────────────────────────────
# Each block prints a clear message if the package is missing so the script
# still runs and reports PSNR even without the optional dependencies.

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


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """PSNR in dB.  Both tensors: (1, 3, H, W) float in [0, 1]."""
    mse = F.mse_loss(sr, hr).item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """SSIM via scikit-image.  Averages across RGB channels."""
    sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()   # HWC
    hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return float(_ski_ssim(hr_np, sr_np, data_range=1.0, channel_axis=2))


def compute_lpips(sr: torch.Tensor, hr: torch.Tensor, fn) -> float:
    """LPIPS.  fn is a pre-loaded lpips.LPIPS network.  Lower = better."""
    # LPIPS expects values in [-1, 1]
    return fn(sr * 2 - 1, hr * 2 - 1).item()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(exp_dir: Path, config_path, checkpoint_name: str,
               device: torch.device):
    """
    Loads a trained model from the experiment directory.

    config_path  : explicit path to YAML, or None to auto-discover from exp_dir.
    checkpoint_name : "best" or "last" (no .pth extension needed).

    Returns (model, cfg).  The caller reads cfg to get scale, domain, etc.
    """
    from models import build_model

    # ── Config discovery ──────────────────────────────────────────────────────
    if config_path is None:
        config_path = exp_dir / "config.yaml"   # auto-saved by train.py
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}\n"
            "  Options:\n"
            "    1. Pass --config configs/your_config.yaml explicitly.\n"
            "    2. Re-run training with the current train.py (it now saves\n"
            "       config.yaml into the experiment directory automatically).\n"
            "    3. Use --bicubic_only to evaluate without a model."
        )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # resolve_paths is not called here — build_model only needs training.scale
    # and model.* keys, not file paths.
    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = exp_dir / "checkpoints" / f"{checkpoint_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Available checkpoints: "
            f"{[p.name for p in (exp_dir/'checkpoints').glob('*.pth')]}"
        )

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    print(f"  Loaded {ckpt_path.name}"
          f"  (epoch {state.get('epoch', '?')}, "
          f"val PSNR {state.get('psnr', 0):.2f} dB)")
    return model, cfg


# ── Image pair discovery ──────────────────────────────────────────────────────

def find_pairs(hr_dir: Path, lr_dir: Path):
    """
    Matches HR and LR files by filename (with .png fallback for LR).
    Returns sorted list of (lr_path, hr_path) tuples.
    """
    hr_files = sorted(p for p in hr_dir.iterdir()
                      if p.suffix.lower() in SUPPORTED_EXTS)
    pairs = []
    for hr_p in hr_files:
        lr_p = lr_dir / hr_p.name
        if not lr_p.exists():
            lr_p = lr_dir / (hr_p.stem + ".png")
        if not lr_p.exists():
            print(f"  [warn] No LR match for {hr_p.name} — skipping.")
            continue
        pairs.append((lr_p, hr_p))
    return pairs


# ── Per-image evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_image(lr_path, hr_path: Path, scale: int,
               model, lpips_fn, device: torch.device,
               has_cond: bool, domain: str | None = None,
               regen_lr: bool = False) -> tuple[dict, bool]:
    """
    Evaluates one HR image (and optionally its pre-saved LR counterpart).

    Two modes controlled by regen_lr:

      regen_lr=False (default)
        Loads LR from lr_path.  For conditioned models a zero vector is used
        (fallback conditioning) — the model still runs, but cannot use its
        conditioning head.

      regen_lr=True
        Ignores lr_path.  Re-degrades hr_path using `domain` (from the training
        config) and captures the exact degradation parameters. Passes the real
        conditioning vector to the model — true conditioning, faithful to the
        proposal.  `domain` must be set when regen_lr=True.

    Returns (metrics_dict, used_true_cond).
    metrics_dict keys: psnr_bicubic, ssim_bicubic, lpips_bicubic,
                       psnr_model, ssim_model, lpips_model (last three only
                       when model is not None).
    """
    hr_img = Image.open(hr_path).convert("RGB")

    # ── LR source ─────────────────────────────────────────────────────────────
    metadata = None
    if regen_lr and domain is not None:
        # Re-degrade from HR so we have the exact parameters for conditioning.
        from domain_degradation import degrade_domain
        lr_img, metadata = degrade_domain(hr_img, scale=scale, domain=domain)
    else:
        lr_img = Image.open(lr_path).convert("RGB")

    hr = to_tensor(hr_img).unsqueeze(0).to(device)   # (1, 3, H, W)
    lr = to_tensor(lr_img).unsqueeze(0).to(device)   # (1, 3, h, w)
    H, W = hr.shape[2], hr.shape[3]

    # Bicubic SR — the fixed baseline we always compare against
    bicubic = F.interpolate(lr, size=(H, W), mode="bicubic",
                            align_corners=False).clamp(0, 1)

    row = {}

    # ── Bicubic metrics ───────────────────────────────────────────────────────
    row["psnr_bicubic"] = compute_psnr(bicubic, hr)
    if HAS_SSIM:
        row["ssim_bicubic"] = compute_ssim(bicubic, hr)
    if HAS_LPIPS and lpips_fn is not None:
        row["lpips_bicubic"] = compute_lpips(bicubic, hr, lpips_fn)

    # ── Model metrics ─────────────────────────────────────────────────────────
    used_true_cond = False
    if model is not None:
        if model.expects_upsampled_input:
            model_input = bicubic              # SRCNN: needs HR-size input
        else:
            model_input = lr                   # SRResNet / ConditionedSRResNet

        if has_cond:
            if metadata is not None:
                # True conditioning: actual degradation parameters captured above.
                from cond_utils import build_cond_vector
                cond = build_cond_vector(metadata).unsqueeze(0).to(device)
                used_true_cond = True
            else:
                # Fallback: zero vector — conditioning head receives no information.
                from cond_utils import COND_DIM
                cond = torch.zeros(1, COND_DIM, device=device)
            sr = model(model_input, cond).clamp(0, 1)
        else:
            sr = model(model_input).clamp(0, 1)

        row["psnr_model"]  = compute_psnr(sr, hr)
        if HAS_SSIM:
            row["ssim_model"]  = compute_ssim(sr, hr)
        if HAS_LPIPS and lpips_fn is not None:
            row["lpips_model"] = compute_lpips(sr, hr, lpips_fn)

    return row, used_true_cond


# ── Aggregate statistics ──────────────────────────────────────────────────────

def aggregate(rows: list[dict], metric: str) -> dict | None:
    """Returns {mean, std, min, max} for a metric across all rows, or None."""
    vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
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
        description="Paired SR evaluation: PSNR / SSIM / LPIPS against HR ground truth"
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (folder under outputs/experiments/)")
    parser.add_argument("--hr_dir",     required=True,
                        help="Directory of HR ground-truth images")
    parser.add_argument("--lr_dir",     default=None,
                        help="Directory of LR input images (matched by filename). "
                             "Not required when --regen_lr is set.")
    parser.add_argument("--regen_lr",   action="store_true",
                        help="Re-degrade HR images using the training domain instead of "
                             "loading pre-saved LR files. Enables true conditioning for "
                             "ConditionedSRResNet (actual degradation params are captured "
                             "and passed as the conditioning vector). Recommended for "
                             "conditioned models. --lr_dir is ignored when this flag is set.")
    parser.add_argument("--config",     default=None,
                        help="YAML config for model instantiation "
                             "(auto-discovered from experiment dir if omitted)")
    parser.add_argument("--checkpoint", default="best",
                        choices=["best", "last"],
                        help="Which checkpoint to load (default: best)")
    parser.add_argument("--scale",      type=int, default=4,
                        help="Upscale factor (default: 4; overridden by config if found)")
    parser.add_argument("--bicubic_only", action="store_true",
                        help="Evaluate bicubic baseline only (no model loaded)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    hr_dir      = Path(args.hr_dir)
    lr_dir      = Path(args.lr_dir) if args.lr_dir else None
    exp_dir     = project_dir / "outputs" / "experiments" / args.experiment
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate LR source ────────────────────────────────────────────────────
    if not args.regen_lr and lr_dir is None and not args.bicubic_only:
        print("[error] Provide either --lr_dir (pre-saved LR images) or --regen_lr "
              "(re-degrade from HR, enables true conditioning).")
        return
    if not args.regen_lr and lr_dir is None and args.bicubic_only:
        # bicubic_only can re-degrade or use HR directly — require lr_dir for bicubic
        print("[error] --bicubic_only requires --lr_dir (pre-saved LR images).")
        return

    if not hr_dir.exists():
        print(f"[error] HR directory not found: {hr_dir}");  return
    if lr_dir is not None and not lr_dir.exists():
        print(f"[error] LR directory not found: {lr_dir}");  return

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if (hasattr(torch.backends, "mps") and
                                      torch.backends.mps.is_available())
              else torch.device("cpu"))
    print(f"Device : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, has_cond, scale, domain = None, False, args.scale, None
    if not args.bicubic_only:
        try:
            model, cfg = load_model(exp_dir, args.config, args.checkpoint, device)
            has_cond   = getattr(model, "expects_cond_vector", False)
            scale      = cfg.get("training", {}).get("scale", args.scale)
            domain     = cfg.get("data", {}).get("domain")   # e.g. "surveillance"
            if has_cond:
                if args.regen_lr and domain:
                    print(f"  [cond] ConditionedSRResNet: TRUE conditioning via --regen_lr.")
                    print(f"         HR images will be re-degraded using domain='{domain}'.")
                    print(f"         Actual degradation params → real conditioning vector per image.")
                else:
                    print(f"  [cond] ConditionedSRResNet: FALLBACK conditioning (zero vector).")
                    print(f"         Pass --regen_lr to enable true conditioning.")
        except FileNotFoundError as e:
            print(f"[warn] Could not load model:\n  {e}\n  Falling back to bicubic only.")

    # ── LPIPS network ─────────────────────────────────────────────────────────
    lpips_fn = None
    if HAS_LPIPS:
        print("Loading LPIPS network (AlexNet)...")
        lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    # ── Image list / pairs ────────────────────────────────────────────────────
    # regen_lr mode: iterate over HR files only (no pre-saved LR needed).
    # standard mode: use find_pairs() to match HR and LR files by filename.
    n_true_cond = 0
    n_fallback  = 0

    if args.regen_lr:
        hr_files = sorted(p for p in hr_dir.iterdir()
                          if p.suffix.lower() in SUPPORTED_EXTS)
        if not hr_files:
            print(f"[error] No images found in {hr_dir}");  return
        print(f"\nEvaluating {len(hr_files)} HR images  "
              f"[regen_lr — re-degrading with domain='{domain}']...")

        rows = []
        for i, hr_p in enumerate(hr_files, 1):
            row, used_true = eval_image(
                None, hr_p, scale, model, lpips_fn, device, has_cond,
                domain=domain, regen_lr=True,
            )
            if has_cond and model is not None:
                n_true_cond += int(used_true)
                n_fallback  += int(not used_true)
            row = {"filename": hr_p.name, **row}
            rows.append(row)

            psnr_b = f"{row.get('psnr_bicubic', 0):.2f}"
            psnr_m = f"{row.get('psnr_model',   0):.2f}" if "psnr_model" in row else "n/a"
            print(f"  [{i:3d}/{len(hr_files)}] {hr_p.name:<20s}  "
                  f"bicubic {psnr_b} dB  model {psnr_m} dB")
    else:
        pairs = find_pairs(hr_dir, lr_dir)
        if not pairs:
            print("[error] No matching LR/HR pairs found.");  return
        print(f"\nEvaluating {len(pairs)} image pairs...")

        rows = []
        for i, (lr_p, hr_p) in enumerate(pairs, 1):
            row, used_true = eval_image(
                lr_p, hr_p, scale, model, lpips_fn, device, has_cond,
                domain=domain, regen_lr=False,
            )
            if has_cond and model is not None:
                n_true_cond += int(used_true)
                n_fallback  += int(not used_true)
            row = {"filename": hr_p.name, **row}
            rows.append(row)

            psnr_b = f"{row.get('psnr_bicubic', 0):.2f}"
            psnr_m = f"{row.get('psnr_model',   0):.2f}" if "psnr_model" in row else "n/a"
            print(f"  [{i:3d}/{len(pairs)}] {hr_p.name:<20s}  "
                  f"bicubic {psnr_b} dB  model {psnr_m} dB")

    # ── Conditioning mode summary ─────────────────────────────────────────────
    if has_cond and model is not None:
        total_cond = n_true_cond + n_fallback
        if n_true_cond == total_cond:
            print(f"\n[cond] All {n_true_cond}/{total_cond} images used TRUE conditioning "
                  f"(actual degradation params from --regen_lr re-degradation).")
        else:
            print(f"\n[cond] {n_fallback}/{total_cond} images used FALLBACK conditioning "
                  f"(zero vector). Run with --regen_lr for true conditioning.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = metrics_dir / "eval_paired.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\nPer-image CSV : {csv_path}")

    # ── Build summary ─────────────────────────────────────────────────────────
    metrics = ["psnr", "ssim", "lpips"]
    methods = ["bicubic"] + (["model"] if model is not None else [])

    summary = {
        "experiment":    args.experiment,
        "checkpoint":    f"{args.checkpoint}.pth" if not args.bicubic_only else None,
        "num_images":    len(rows),
        "scale":         scale,
        "regen_lr":      args.regen_lr,
        "cond_true":     n_true_cond if has_cond and model else None,
        "cond_fallback": n_fallback  if has_cond and model else None,
    }
    for method in methods:
        summary[method] = {}
        for m in metrics:
            key  = f"{m}_{method}"
            agg  = aggregate(rows, key)
            if agg is not None:
                summary[method][m] = agg

    json_path = metrics_dir / "eval_paired_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON  : {json_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment : {args.experiment}  |  Images : {len(rows)}")
    print(f"  {'Method':<12}  {'PSNR (dB)':>12}  {'SSIM':>8}  {'LPIPS':>8}")
    print(f"  {'-'*52}")
    for method in methods:
        d = summary.get(method, {})
        psnr_s = f"{d['psnr']['mean']:.2f} ± {d['psnr']['std']:.2f}" if "psnr" in d else "n/a"
        ssim_s = f"{d['ssim']['mean']:.4f}"  if "ssim"  in d else "n/a"
        lpips_s= f"{d['lpips']['mean']:.4f}" if "lpips" in d else "n/a"
        print(f"  {method:<12}  {psnr_s:>12}  {ssim_s:>8}  {lpips_s:>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
