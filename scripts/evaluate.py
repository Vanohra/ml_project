"""
evaluate.py
-----------
Cross-condition evaluation of all three super-resolution models.

Two evaluation modes:

  MODE 1 — IN-DISTRIBUTION (default, no --cross_condition):
    Evaluate the surveillance-trained models (SRCNN-surv, SRResNet) and the
    bicubic baseline on surveillance-degraded validation images.
    Tests: "how good is each model at what it was trained for?"

  MODE 2 — CROSS-CONDITION (--cross_condition flag):
    Evaluate ALL models — including SimpleSRCNN trained on CLEAN bicubic LR —
    on the SAME surveillance-degraded inputs.
    This is the project's key scientific contribution: it quantifies the PSNR
    cost of training-distribution mismatch (training on clean LR, deploying on
    realistic surveillance degradation).

Usage (from ml_project/ directory):
  python scripts/evaluate.py \\
      --srcnn_clean_ckpt  outputs/checkpoints_srcnn_clean/best.pth \\
      --srcnn_surv_ckpt   outputs/checkpoints_srcnn_surv/best.pth \\
      --srresnet_ckpt     outputs/checkpoints_srresnet/best.pth \\
      --hr_dir            data/DIV2K/HR_valid \\
      --n_samples         5 \\
      --cross_condition

Models are skipped gracefully if their checkpoint does not exist.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from degradation import get_domain_preset, degrade
from metrics import compute_psnr, compute_ssim
from model import SimpleSRCNN, SRResNet

PROJECT_DIR = Path(__file__).resolve().parent.parent
SEED = 42


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_obj, ckpt_path: Path, device: torch.device, label: str):
    if ckpt_path is None or not ckpt_path.exists():
        print(f"  [skip] {label} checkpoint not found: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model", state)
    model_obj.load_state_dict(sd)
    model_obj.eval()
    ep   = state.get("epoch", "?")
    psnr = state.get("psnr", float("nan"))
    print(f"  [loaded] {label} from {ckpt_path.name}  "
          f"(epoch {ep}, saved PSNR {psnr:.2f} dB)")
    return model_obj


def load_srcnn(ckpt_path, device):
    if ckpt_path is None:
        return None
    return _load_model(SimpleSRCNN().to(device), Path(ckpt_path), device, "SimpleSRCNN")


def load_srresnet(ckpt_path, device, scale=4):
    if ckpt_path is None:
        return None
    return _load_model(SRResNet(scale=scale).to(device), Path(ckpt_path),
                       device, "SRResNet")


# ── Image utilities ────────────────────────────────────────────────────────────

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return to_tensor(img).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))


def center_crop_256(img: Image.Image) -> Image.Image:
    """Return a 256x256 centre patch (or smaller if image is tiny)."""
    w, h   = img.size
    cw, ch = min(256, w), min(256, h)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def _avg(lst):
    return sum(lst) / len(lst) if lst else float("nan")


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_all(hr_paths, srcnn_clean, srcnn_surv, srresnet,
                 device, scale, surv_cfg, cross_condition):
    """
    Evaluates all available methods on surveillance-degraded validation images.

    Returns a dict keyed by method name; each value has "psnr" and "ssim" lists.
    Keys present: "bicubic" always, plus "srcnn_clean" (if cross_condition and
    model loaded), "srcnn_surv", "srresnet" (if respective model loaded).
    """
    results = {"bicubic": {"psnr": [], "ssim": []}}
    if cross_condition and srcnn_clean is not None:
        results["srcnn_clean"] = {"psnr": [], "ssim": []}
    if srcnn_surv is not None:
        results["srcnn_surv"] = {"psnr": [], "ssim": []}
    if srresnet is not None:
        results["srresnet"] = {"psnr": [], "ssim": []}

    random.seed(SEED)
    np.random.seed(SEED)

    for i, hr_path in enumerate(hr_paths, 1):
        hr_pil = Image.open(hr_path).convert("RGB")
        w = (hr_pil.width  // scale) * scale
        h = (hr_pil.height // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))
        lr_pil = degrade(hr_pil, config=surv_cfg)

        hr_t = pil_to_tensor(hr_pil).to(device)
        lr_t = pil_to_tensor(lr_pil).to(device)

        # Bicubic
        bic_t = F.interpolate(lr_t, scale_factor=scale, mode="bicubic",
                              align_corners=False).clamp(0, 1)
        results["bicubic"]["psnr"].append(compute_psnr(bic_t, hr_t))
        results["bicubic"]["ssim"].append(compute_ssim(bic_t, hr_t))

        # SRCNN clean (cross-condition only)
        if cross_condition and srcnn_clean is not None:
            with torch.no_grad():
                lr_up = F.interpolate(lr_t, scale_factor=scale,
                                      mode="bicubic", align_corners=False)
                sr = srcnn_clean(lr_up).clamp(0, 1)
            results["srcnn_clean"]["psnr"].append(compute_psnr(sr, hr_t))
            results["srcnn_clean"]["ssim"].append(compute_ssim(sr, hr_t))

        # SRCNN surv
        if srcnn_surv is not None:
            with torch.no_grad():
                lr_up = F.interpolate(lr_t, scale_factor=scale,
                                      mode="bicubic", align_corners=False)
                sr = srcnn_surv(lr_up).clamp(0, 1)
            results["srcnn_surv"]["psnr"].append(compute_psnr(sr, hr_t))
            results["srcnn_surv"]["ssim"].append(compute_ssim(sr, hr_t))

        # SRResNet
        if srresnet is not None:
            with torch.no_grad():
                sr = srresnet(lr_t).clamp(0, 1)
            results["srresnet"]["psnr"].append(compute_psnr(sr, hr_t))
            results["srresnet"]["ssim"].append(compute_ssim(sr, hr_t))

        print(f"  [{i:3d}/{len(hr_paths)}] {hr_path.name}")

    return results


# ── Results table ─────────────────────────────────────────────────────────────

def print_table(results: dict, cross_condition: bool) -> None:
    bic_psnr = _avg(results["bicubic"]["psnr"])

    rows = []
    rows.append(("Bicubic baseline",
                 bic_psnr,
                 _avg(results["bicubic"]["ssim"]),
                 "No ML, fixed filter"))

    if "srcnn_clean" in results:
        p = _avg(results["srcnn_clean"]["psnr"])
        s = _avg(results["srcnn_clean"]["ssim"])
        delta = p - bic_psnr
        note  = (f"WORSE than bicubic — mismatch!"
                 if delta < 0 else f"+{delta:.2f} dB over bicubic")
        rows.append(("SRCNN (trained on clean LR)", p, s, note))

    if "srcnn_surv" in results:
        p = _avg(results["srcnn_surv"]["psnr"])
        s = _avg(results["srcnn_surv"]["ssim"])
        rows.append(("SRCNN (trained on surv. preset)", p, s,
                     f"+{p - bic_psnr:.2f} dB over bicubic"))

    if "srresnet" in results:
        p = _avg(results["srresnet"]["psnr"])
        s = _avg(results["srresnet"]["ssim"])
        rows.append(("SRResNet (trained on surv. preset)", p, s,
                     f"+{p - bic_psnr:.2f} dB over bicubic"))

    title = ("EVALUATION RESULTS — Surveillance-Degraded Inputs"
             if cross_condition
             else "EVALUATION RESULTS — In-Distribution (Surveillance)")

    W = 62
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)
    print(f"  {'Method':<35} {'PSNR':>8}  {'SSIM':>6}  Notes")
    print("  " + "-" * (W - 2))
    for method, psnr, ssim, note in rows:
        print(f"  {method:<35} {psnr:>8.2f}  {ssim:>6.3f}  {note}")
    print("=" * W)

    if (cross_condition
            and "srcnn_clean" in results
            and "srcnn_surv"  in results
            and "srresnet"    in results):
        p_clean = _avg(results["srcnn_clean"]["psnr"])
        p_surv  = _avg(results["srcnn_surv"]["psnr"])
        p_res   = _avg(results["srresnet"]["psnr"])
        dg = p_surv - p_clean
        ag = p_res  - p_surv
        print(f"  KEY FINDING: Training on matched degradation adds ~{dg:.1f} dB.")
        print(f"               Deeper model adds another ~{ag:.1f} dB on top.")
        print("=" * W)


# ── CSV saving ────────────────────────────────────────────────────────────────

def save_csv(results: dict, out_path: Path, hr_paths: list) -> None:
    _DISPLAY = {
        "bicubic":     "Bicubic",
        "srcnn_clean": "SRCNN_clean",
        "srcnn_surv":  "SRCNN_surv",
        "srresnet":    "SRResNet",
    }
    order  = [k for k in _DISPLAY if k in results]
    n_imgs = len(next(iter(results.values()))["psnr"])

    rows = []
    for i in range(n_imgs):
        row = {"image": hr_paths[i].name}
        for key in order:
            row[f"{_DISPLAY[key]}_PSNR"] = round(results[key]["psnr"][i], 4)
            row[f"{_DISPLAY[key]}_SSIM"] = round(results[key]["ssim"][i], 6)
        rows.append(row)

    summary = {"image": "MEAN"}
    for key in order:
        summary[f"{_DISPLAY[key]}_PSNR"] = round(_avg(results[key]["psnr"]), 4)
        summary[f"{_DISPLAY[key]}_SSIM"] = round(_avg(results[key]["ssim"]), 6)
    rows.append(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results CSV: {out_path}")


# ── Visual comparison grids ───────────────────────────────────────────────────

_COL_LABELS = {
    "lr_disp":     "LR input",
    "bicubic":     "Bicubic",
    "srcnn_clean": "SRCNN\n(clean LR)",
    "srcnn_surv":  "SRCNN\n(surv. preset)",
    "srresnet":    "SRResNet\n(surv. preset)",
    "hr":          "HR ground truth",
}


def _run_models(lr_t, hr_t, srcnn_clean, srcnn_surv, srresnet,
                device, scale, cross_condition):
    """Return dict: key -> (tensor_cpu, psnr_float, ssim_float)."""
    h, w  = hr_t.shape[2], hr_t.shape[3]
    bic_t = F.interpolate(lr_t, size=(h, w), mode="bicubic",
                          align_corners=False).clamp(0, 1)
    out = {"bicubic": (bic_t.cpu(),
                       compute_psnr(bic_t, hr_t),
                       compute_ssim(bic_t, hr_t))}

    def _srcnn_fwd(model):
        with torch.no_grad():
            lr_up = F.interpolate(lr_t, size=(h, w), mode="bicubic",
                                  align_corners=False)
            return model(lr_up).clamp(0, 1)

    if cross_condition and srcnn_clean is not None:
        sr = _srcnn_fwd(srcnn_clean)
        out["srcnn_clean"] = (sr.cpu(), compute_psnr(sr, hr_t), compute_ssim(sr, hr_t))

    if srcnn_surv is not None:
        sr = _srcnn_fwd(srcnn_surv)
        out["srcnn_surv"] = (sr.cpu(), compute_psnr(sr, hr_t), compute_ssim(sr, hr_t))

    if srresnet is not None:
        with torch.no_grad():
            sr = srresnet(lr_t).clamp(0, 1)
        out["srresnet"] = (sr.cpu(), compute_psnr(sr, hr_t), compute_ssim(sr, hr_t))

    return out


def _save_grid(hr_path, lr_pil, model_outs, hr_t, out_path, cross_condition, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available — visual grids skipped")
        return

    hr_cpu = hr_t.cpu()
    lr_t2  = pil_to_tensor(lr_pil).cpu()
    h, w   = hr_cpu.shape[2], hr_cpu.shape[3]
    lr_disp = F.interpolate(lr_t2, size=(h, w), mode="nearest").clamp(0, 1)

    col_order = ["lr_disp", "bicubic", "srcnn_clean", "srcnn_surv", "srresnet", "hr"]
    panels = []
    for key in col_order:
        if key == "lr_disp":
            panels.append((key, tensor_to_pil(lr_disp), None, None))
        elif key == "hr":
            panels.append((key, tensor_to_pil(hr_cpu), None, None))
        elif key in model_outs:
            t, p, s = model_outs[key]
            panels.append((key, tensor_to_pil(t), p, s))
        # skip missing models silently

    # 256x256 centre crop for compact display
    panels = [(k, center_crop_256(img), p, s) for k, img, p, s in panels]

    n = len(panels)
    fig_h = 20 * panels[0][1].height / (panels[0][1].width * max(n, 1))
    fig, axes = plt.subplots(1, n, figsize=(20, max(fig_h, 3.0)))
    if n == 1:
        axes = [axes]

    for ax, (key, img, psnr, ssim) in zip(axes, panels):
        ax.imshow(np.array(img))
        ax.set_title(_COL_LABELS.get(key, key), fontsize=9, pad=3)
        if psnr is not None:
            ax.set_xlabel(f"PSNR {psnr:.2f} dB\nSSIM {ssim:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{tag}{hr_path.name}", fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grid] Saved: {out_path.name}")


def save_visual_grids(hr_paths, srcnn_clean, srcnn_surv, srresnet,
                      device, scale, surv_cfg, out_dir, n_samples, cross_condition):
    random.seed(SEED)
    np.random.seed(SEED)

    indices = list(range(len(hr_paths)))
    if len(indices) > n_samples:
        step    = len(indices) / n_samples
        indices = [int(i * step) for i in range(n_samples)]

    for grid_i, img_i in enumerate(indices):
        hr_path = hr_paths[img_i]
        hr_pil  = Image.open(hr_path).convert("RGB")
        w = (hr_pil.width  // scale) * scale
        h = (hr_pil.height // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))
        lr_pil = degrade(hr_pil, config=surv_cfg)

        hr_t = pil_to_tensor(hr_pil).to(device)
        lr_t = pil_to_tensor(lr_pil).to(device)

        outs = _run_models(lr_t, hr_t, srcnn_clean, srcnn_surv, srresnet,
                           device, scale, cross_condition)

        _save_grid(hr_path, lr_pil, outs, hr_t,
                   out_dir / f"sample_{grid_i:02d}.png",
                   cross_condition)


def save_mismatch_figure(hr_paths, srcnn_clean, srcnn_surv,
                         device, scale, surv_cfg, out_dir):
    """
    Saves a figure for the validation image where SRCNN-clean performs MOST
    differently from SRCNN-surv (largest PSNR gap on surveillance-degraded input).
    This is the 'money figure' demonstrating the cost of degradation mismatch.
    """
    if srcnn_clean is None or srcnn_surv is None:
        return

    random.seed(SEED)
    np.random.seed(SEED)

    gaps  = []
    cache = []

    for hr_path in hr_paths:
        hr_pil = Image.open(hr_path).convert("RGB")
        w = (hr_pil.width  // scale) * scale
        h = (hr_pil.height // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))
        lr_pil = degrade(hr_pil, config=surv_cfg)

        hr_t = pil_to_tensor(hr_pil).to(device)
        lr_t = pil_to_tensor(lr_pil).to(device)
        h2, w2 = hr_t.shape[2], hr_t.shape[3]

        with torch.no_grad():
            lr_up = F.interpolate(lr_t, size=(h2, w2),
                                  mode="bicubic", align_corners=False)
            sr_cl = srcnn_clean(lr_up).clamp(0, 1)
            sr_sv = srcnn_surv(lr_up).clamp(0, 1)

        gap = compute_psnr(sr_sv, hr_t) - compute_psnr(sr_cl, hr_t)
        gaps.append(gap)
        cache.append((hr_path, lr_pil, hr_t, lr_t))

    worst_i = max(range(len(gaps)), key=lambda i: gaps[i])
    hr_path, lr_pil, hr_t, lr_t = cache[worst_i]

    outs = _run_models(lr_t, hr_t, srcnn_clean, srcnn_surv, None,
                       device, scale, cross_condition=True)

    _save_grid(hr_path, lr_pil, outs, hr_t,
               out_dir / "mismatch_example.png",
               cross_condition=True,
               tag=f"Mismatch (gap={gaps[worst_i]:.2f} dB) — ")
    print(f"  [mismatch] Worst gap: {gaps[worst_i]:.2f} dB on {hr_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-condition SR evaluation: Bicubic / SRCNN-clean / "
            "SRCNN-surv / SRResNet on surveillance-degraded inputs."
        )
    )
    parser.add_argument("--srcnn_clean_ckpt", default=None,
                        help="SimpleSRCNN trained on clean bicubic LR (Exp 1)")
    parser.add_argument("--srcnn_surv_ckpt",  default=None,
                        help="SimpleSRCNN trained on surveillance preset (Exp 2/3)")
    parser.add_argument("--srresnet_ckpt",    default=None,
                        help="SRResNet trained on surveillance preset (Exp 3)")
    parser.add_argument("--hr_dir",
                        default="data/DIV2K/HR_valid",
                        help="HR validation image directory")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of visual comparison grids to save (default 5)")
    parser.add_argument("--cross_condition", action="store_true",
                        help=(
                            "Include SRCNN-clean evaluated on surveillance inputs "
                            "(key scientific result: cost of degradation mismatch)"
                        ))
    parser.add_argument("--output_dir", default="outputs/eval_results",
                        help="Output directory for results (default: outputs/eval_results/)")
    args = parser.parse_args()

    def _resolve(p):
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else PROJECT_DIR / p

    srcnn_clean_ckpt = _resolve(args.srcnn_clean_ckpt)
    srcnn_surv_ckpt  = _resolve(args.srcnn_surv_ckpt)
    srresnet_ckpt    = _resolve(args.srresnet_ckpt)
    hr_dir           = _resolve(args.hr_dir)
    out_dir          = _resolve(args.output_dir)
    SCALE = 4

    print("\n" + "=" * 62)
    print("  evaluate.py — Super-Resolution Cross-Condition Evaluation")
    print("=" * 62)

    # ── HR images ─────────────────────────────────────────────────────────────
    SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    if not hr_dir.exists():
        print(f"\n[error] HR directory not found: {hr_dir}")
        sys.exit(1)
    hr_paths = sorted([p for p in hr_dir.iterdir()
                       if p.suffix.lower() in SUPPORTED])
    if not hr_paths:
        print(f"\n[error] No images found in {hr_dir}")
        sys.exit(1)
    print(f"\n  HR images  : {len(hr_paths)} in {hr_dir}")

    # ── Device & models ───────────────────────────────────────────────────────
    device = get_device()
    print(f"  Device     : {device}")
    print(f"  Mode       : "
          f"{'cross-condition' if args.cross_condition else 'in-distribution'}")
    print("\nLoading models...")

    srcnn_clean = load_srcnn(srcnn_clean_ckpt, device)
    srcnn_surv  = load_srcnn(srcnn_surv_ckpt,  device)
    srresnet    = load_srresnet(srresnet_ckpt, device, scale=SCALE)

    surv_cfg = get_domain_preset("surveillance")
    print(f"\n  Degradation: surveillance preset (seed={SEED})")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(hr_paths)} image(s)...")
    results = evaluate_all(hr_paths, srcnn_clean, srcnn_surv, srresnet,
                           device, SCALE, surv_cfg, args.cross_condition)

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(results, args.cross_condition)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(results, out_dir / "eval_results.csv", hr_paths)

    # ── Visual grids ──────────────────────────────────────────────────────────
    print(f"\nSaving {args.n_samples} visual comparison grids -> {out_dir}")
    save_visual_grids(hr_paths, srcnn_clean, srcnn_surv, srresnet,
                      device, SCALE, surv_cfg, out_dir,
                      args.n_samples, args.cross_condition)

    # ── Mismatch figure ───────────────────────────────────────────────────────
    if (args.cross_condition
            and srcnn_clean is not None
            and srcnn_surv  is not None):
        print("\nFinding worst-case mismatch image...")
        save_mismatch_figure(hr_paths, srcnn_clean, srcnn_surv,
                             device, SCALE, surv_cfg, out_dir)

    print("\nDone.  Results in:", out_dir)


if __name__ == "__main__":
    main()
