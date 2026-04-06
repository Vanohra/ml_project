"""
infer.py
--------
Folder-based inference: run a trained SR model on every image in a directory
and save the restored outputs.  Designed for real-world LR images (e.g. RealSRSet)
where no HR ground truth exists.

What it does
  1. Loads the trained model from the experiment's checkpoints/ folder.
  2. Reads every image in --img_dir.
  3. Runs SR on each one and saves the result to:
       outputs/experiments/<name>/inference/<tag>/
  4. Optionally saves a gallery contact sheet (LR upscaled | SR side-by-side)
     as gallery.png in the same output folder — useful for a quick visual check.
  5. Writes run_info.json (provenance: model, checkpoint, scale, source dir, date).

Usage examples
  # Basic inference with best checkpoint:
  python scripts/infer.py \\
      --experiment srresnet_run \\
      --img_dir data/RealSRSet

  # Give the output a descriptive tag so it doesn't overwrite previous runs:
  python scripts/infer.py \\
      --experiment srresnet_run \\
      --img_dir data/RealSRSet \\
      --tag realSRSet_best

  # Use last checkpoint and save a gallery:
  python scripts/infer.py \\
      --experiment srresnet_run \\
      --img_dir data/RealSRSet \\
      --checkpoint last \\
      --save_gallery

  # Bicubic upsampling only (no model — useful as a visual baseline):
  python scripts/infer.py \\
      --experiment srresnet_run \\
      --img_dir data/RealSRSet \\
      --bicubic_only \\
      --tag bicubic_baseline \\
      --save_gallery

Output folder layout
  outputs/experiments/<name>/inference/<tag>/
    0001_sr.png          SR image for each input
    0001_bicubic.png     Bicubic upsampled (always saved alongside SR for comparison)
    gallery.png          Contact sheet  LR | Bicubic | SR  (with --save_gallery)
    run_info.json        Provenance record

Notes
  - ConditionedSRResNet is evaluated with a zero conditioning vector
    (same as evaluate_noref.py).
  - Images are processed one at a time (no batching) so arbitrarily large
    images do not cause out-of-memory errors.
  - SR images are saved as PNG regardless of input format to avoid quality loss.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Gallery layout constants
GALLERY_MAX_COLS  = 4    # images per row in the contact sheet
GALLERY_THUMB_W   = 320  # thumbnail width (px); height scales proportionally


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(exp_dir: Path, config_path, checkpoint_name: str,
               device: torch.device):
    """Load trained model from experiment directory."""
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


# ── Image SR ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_sr(img_path: Path, scale: int, model, device: torch.device,
           has_cond: bool):
    """
    Run super-resolution on a single LR image.

    Returns (lr_tensor, bicubic_tensor, sr_tensor) each (1,3,H,W) in [0,1].
    sr_tensor == bicubic_tensor when model is None (bicubic-only mode).
    """
    lr_img = Image.open(img_path).convert("RGB")
    lr = to_tensor(lr_img).unsqueeze(0).to(device)
    h, w = lr.shape[2], lr.shape[3]
    H, W = h * scale, w * scale

    bicubic = F.interpolate(lr, size=(H, W), mode="bicubic",
                            align_corners=False).clamp(0, 1)

    if model is None:
        return lr, bicubic, bicubic

    if model.expects_upsampled_input:
        model_input = bicubic
    else:
        model_input = lr

    if has_cond:
        from cond_utils import COND_DIM
        cond = torch.zeros(1, COND_DIM, device=device)
        sr = model(model_input, cond).clamp(0, 1)
    else:
        sr = model(model_input).clamp(0, 1)

    return lr, bicubic, sr


# ── Tensor -> PIL helpers ─────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (1,3,H,W) float tensor in [0,1] to a PIL RGB image."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))


def _thumb(img: Image.Image, target_w: int) -> Image.Image:
    """Resize to target_w while keeping aspect ratio."""
    w, h = img.size
    target_h = max(1, int(h * target_w / w))
    return img.resize((target_w, target_h), Image.LANCZOS)


# ── Gallery / contact sheet ───────────────────────────────────────────────────

def make_gallery(entries: list[dict], out_path: Path,
                 thumb_w: int = GALLERY_THUMB_W,
                 max_cols: int = GALLERY_MAX_COLS) -> None:
    """
    Build a contact sheet grid.

    entries  : list of {"name": str, "lr": PIL, "bicubic": PIL, "sr": PIL}
               If "sr" is same object as "bicubic", no SR panel is shown.
    out_path : where to save gallery.png
    """
    if not entries:
        return

    has_model = any(e["sr"] is not e["bicubic"] for e in entries)

    # Each entry becomes one column group: LR thumbnail | [Bicubic] | [SR]
    panels_per_entry = 2 + int(has_model)   # LR + Bicubic [+ SR]
    num_entries = len(entries)
    cols = min(max_cols, num_entries)
    rows = (num_entries + cols - 1) // cols

    # Compute thumbnail height from the first entry
    sample = _thumb(entries[0]["lr"], thumb_w)
    thumb_h = sample.height

    label_h = 18      # pixels for filename label
    cell_w  = thumb_w * panels_per_entry + (panels_per_entry - 1) * 2
    cell_h  = thumb_h + label_h

    canvas_w = cols * cell_w + (cols - 1) * 6
    canvas_h = rows * cell_h + (rows - 1) * 6

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    for idx, entry in enumerate(entries):
        row_i = idx // cols
        col_i = idx % cols
        cell_x = col_i * (cell_w + 6)
        cell_y = row_i * (cell_h + 6)

        lr_t      = _thumb(entry["lr"],      thumb_w)
        bicubic_t = lr_t.resize(lr_t.size, Image.LANCZOS)  # already HR size
        bicubic_t = _thumb(entry["bicubic"], thumb_w)

        panels = [lr_t, bicubic_t]
        if has_model:
            panels.append(_thumb(entry["sr"], thumb_w))

        for p_i, panel in enumerate(panels):
            px = cell_x + p_i * (thumb_w + 2)
            canvas.paste(panel, (px, cell_y))

        # Filename label (white text approximated with a grey strip)
        # We use PIL's ImageDraw for the label
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            label = entry["name"][:28]
            draw.text((cell_x + 2, cell_y + thumb_h + 2),
                      label, fill=(220, 220, 220))
        except Exception:
            pass

    canvas.save(out_path)
    print(f"  Gallery      : {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Folder-based SR inference on real-world LR images (e.g. RealSRSet)"
    )
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (folder under outputs/experiments/)")
    parser.add_argument("--img_dir",    required=True,
                        help="Directory of LR input images")
    parser.add_argument("--tag",        default=None,
                        help="Output subfolder tag (default: checkpoint name + timestamp)")
    parser.add_argument("--config",     default=None,
                        help="YAML config path (auto-discovered from experiment dir if omitted)")
    parser.add_argument("--checkpoint", default="best",
                        choices=["best", "last"],
                        help="Which checkpoint to load (default: best)")
    parser.add_argument("--scale",      type=int, default=4,
                        help="SR upscale factor (default: 4; overridden by config if found)")
    parser.add_argument("--bicubic_only", action="store_true",
                        help="Save bicubic upsampled images instead of model SR")
    parser.add_argument("--save_gallery", action="store_true",
                        help="Save a LR | Bicubic | SR contact sheet as gallery.png")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    img_dir     = Path(args.img_dir)
    exp_dir     = project_dir / "outputs" / "experiments" / args.experiment

    if not img_dir.exists():
        print(f"[error] Image directory not found: {img_dir}");  return

    # ── Output tag ────────────────────────────────────────────────────────────
    tag = args.tag or (
        f"bicubic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.bicubic_only else
        f"{args.checkpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = exp_dir / "inference" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if (hasattr(torch.backends, "mps") and
                                       torch.backends.mps.is_available())
              else torch.device("cpu"))
    print(f"Device       : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, has_cond, scale = None, False, args.scale
    if not args.bicubic_only:
        try:
            model, cfg = load_model(exp_dir, args.config, args.checkpoint, device)
            has_cond   = getattr(model, "expects_cond_vector", False)
            scale      = cfg.get("training", {}).get("scale", args.scale)
            if has_cond:
                print("  [note] ConditionedSRResNet: zero conditioning vector.")
        except FileNotFoundError as e:
            print(f"[warn] Could not load model:\n  {e}\n  Falling back to bicubic only.")

    # ── Image list ────────────────────────────────────────────────────────────
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        print(f"[error] No images found in {img_dir}");  return
    print(f"Input images : {len(images)}  in  {img_dir}")
    print(f"Output dir   : {out_dir}")
    print(f"Scale        : x{scale}\n")

    # ── Per-image loop ────────────────────────────────────────────────────────
    gallery_entries = []

    for i, img_p in enumerate(images, 1):
        lr_t, bic_t, sr_t = run_sr(img_p, scale, model, device, has_cond)

        sr_pil      = tensor_to_pil(sr_t)
        bicubic_pil = tensor_to_pil(bic_t)
        lr_pil      = tensor_to_pil(lr_t)

        # Save SR image (or bicubic when model is None)
        method = "bicubic" if model is None else "sr"
        sr_save_path      = out_dir / f"{img_p.stem}_{method}.png"
        bicubic_save_path = out_dir / f"{img_p.stem}_bicubic.png"

        sr_pil.save(sr_save_path)
        bicubic_pil.save(bicubic_save_path)

        print(f"  [{i:3d}/{len(images)}] {img_p.name:<24s}  "
              f"LR {lr_pil.width}x{lr_pil.height} -> "
              f"SR {sr_pil.width}x{sr_pil.height}")

        if args.save_gallery:
            gallery_entries.append({
                "name":    img_p.name,
                "lr":      lr_pil,
                "bicubic": bicubic_pil,
                "sr":      sr_pil,
            })

    # ── Gallery ───────────────────────────────────────────────────────────────
    if args.save_gallery and gallery_entries:
        make_gallery(gallery_entries, out_dir / "gallery.png")

    # ── Provenance record ─────────────────────────────────────────────────────
    run_info = {
        "experiment":  args.experiment,
        "checkpoint":  None if args.bicubic_only else f"{args.checkpoint}.pth",
        "bicubic_only": args.bicubic_only,
        "img_dir":     str(img_dir),
        "num_images":  len(images),
        "scale":       scale,
        "output_dir":  str(out_dir),
        "tag":         tag,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "device":      str(device),
    }
    info_path = out_dir / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nProvenance   : {info_path}")
    print(f"SR images    : {out_dir}")

    # ── Next steps hint ───────────────────────────────────────────────────────
    print(f"\nTo score these SR images with NIQE / BRISQUE:")
    print(f"  python scripts/evaluate_noref.py \\")
    print(f"      --experiment {args.experiment} \\")
    print(f"      --img_dir {img_dir} \\")
    print(f"      --bicubic_only" if args.bicubic_only else
          f"      --checkpoint {args.checkpoint}")


if __name__ == "__main__":
    main()
