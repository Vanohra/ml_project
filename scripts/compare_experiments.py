"""
compare_experiments.py
-----------------------
Reads every experiment in outputs/experiments/ and prints a comparison table.
Also writes outputs/comparison_table.csv.

No arguments needed — it discovers experiments automatically.

Usage
  python scripts/compare_experiments.py

  # Filter to specific experiments:
  python scripts/compare_experiments.py --experiments baseline_srcnn srresnet_run

  # Sort by a specific metric (default: psnr_model):
  python scripts/compare_experiments.py --sort psnr_model
  python scripts/compare_experiments.py --sort ssim_model
  python scripts/compare_experiments.py --sort niqe_model   # lower is better

  # Regenerate from scratch instead of using existing results.csv:
  python scripts/compare_experiments.py --rebuild

How it works
  For each outputs/experiments/<name>/ directory it finds:
    1. run_summary.json      — model, domain, curriculum, seed, best training PSNR
    2. metrics/eval_paired_summary.json   — PSNR, SSIM, LPIPS (if eval was run)
    3. metrics/eval_noref_summary.json    — NIQE, BRISQUE  (if eval was run)

  It builds a row per experiment and writes outputs/comparison_table.csv.
  The CSV from run_experiment.py (outputs/results.csv) is also printed if present.
"""

import argparse
import csv
import json
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _get(d: dict, *keys, default=""):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


def _fmt(val, decimals: int = 4) -> str:
    """Format a value for display: round floats, pass strings through."""
    if val == "" or val is None:
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(exp_dir: Path) -> dict | None:
    """
    Build one comparison row from an experiment directory.
    Returns None if the directory has no run_summary.json (not a finished run).
    """
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    paired  = _load_json(exp_dir / "metrics" / "eval_paired_summary.json")
    noref   = _load_json(exp_dir / "metrics" / "eval_noref_summary.json")
    virat   = _load_json(exp_dir / "metrics" / "eval_virat_summary.json")

    loss_info = summary.get("loss", {})

    row = {
        "experiment":       summary.get("experiment", exp_dir.name),
        "model":            summary.get("model", ""),
        "domain":           summary.get("domain") or "generic",
        "curriculum":       "yes" if summary.get("curriculum_enabled", False) else "no",
        "loss_pixel":       loss_info.get("pixel_type", ""),
        "loss_perceptual":  "yes" if loss_info.get("perceptual_enabled", False) else "no",
        "epochs":           summary.get("num_epochs", ""),
        "seed":             summary.get("seed", ""),
        # training PSNR
        "best_psnr_train":  summary.get("best_psnr_db", ""),
        # paired eval
        "psnr_bicubic":     _get(paired, "bicubic", "psnr",  "mean"),
        "psnr_model":       _get(paired, "model",   "psnr",  "mean"),
        "ssim_bicubic":     _get(paired, "bicubic", "ssim",  "mean"),
        "ssim_model":       _get(paired, "model",   "ssim",  "mean"),
        "lpips_bicubic":    _get(paired, "bicubic", "lpips", "mean"),
        "lpips_model":      _get(paired, "model",   "lpips", "mean"),
        # no-reference eval
        "niqe_bicubic":     _get(noref, "bicubic", "niqe",    "mean"),
        "niqe_model":       _get(noref, "model",   "niqe",    "mean"),
        "brisque_bicubic":  _get(noref, "bicubic", "brisque", "mean"),
        "brisque_model":    _get(noref, "model",   "brisque", "mean"),
        # VIRAT eval
        "virat_psnr_bicubic": _get(virat, "bicubic", "psnr", "mean"),
        "virat_psnr_model":   _get(virat, "model",   "psnr", "mean"),
    }
    return row


# ── Table printing ────────────────────────────────────────────────────────────

# Column display config: (key, header, decimals, width)
_TABLE_COLS = [
    ("experiment",      "Experiment",        0,  24),
    ("model",           "Model",             0,  20),
    ("domain",          "Domain",            0,  12),
    ("curriculum",      "Curr.",             0,   5),
    ("loss_pixel",      "Pixel",             0,   5),
    ("loss_perceptual", "Perc.",             0,   5),
    ("best_psnr_train", "Train\nPSNR",       2,   7),
    ("psnr_bicubic",    "PSNR\nBicubic",     2,   8),
    ("psnr_model",      "PSNR\nModel",       2,   7),
    ("ssim_model",      "SSIM\nModel",       4,   7),
    ("lpips_model",     "LPIPS\nModel",      4,   7),
    ("niqe_model",      "NIQE\nModel",       3,   7),
    ("brisque_model",   "BRISQUE\nModel",    2,   9),
    ("virat_psnr_model","VIRAT\nPSNR",       2,   7),
]


def _print_table(rows: list[dict]) -> None:
    """Print a fixed-width comparison table to the console."""
    if not rows:
        print("  (no experiments found)")
        return

    # Build two-line header
    header1 = ""
    header2 = ""
    sep     = ""
    for key, label, decimals, width in _TABLE_COLS:
        parts = label.split("\n")
        h1    = parts[0] if len(parts) > 1 else label
        h2    = parts[1] if len(parts) > 1 else ""
        header1 += h1.ljust(width) + "  "
        header2 += h2.ljust(width) + "  "
        sep     += "-" * width + "  "

    print(header1.rstrip())
    print(header2.rstrip())
    print(sep.rstrip())

    for row in rows:
        line = ""
        for key, label, decimals, width in _TABLE_COLS:
            val = row.get(key, "")
            cell = _fmt(val, decimals) if val != "" else "—"
            line += cell.ljust(width) + "  "
        print(line.rstrip())

    print()
    print("  PSNR/SSIM: higher is better   LPIPS/NIQE/BRISQUE: lower is better")


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_COLUMNS = [col for col, *_ in _TABLE_COLS] + [
    "epochs", "seed",
    "ssim_bicubic", "lpips_bicubic",
    "niqe_bicubic", "brisque_bicubic",
    "virat_psnr_bicubic",
]


def _write_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"Comparison CSV : {out_path}")


# ── Sort helper ───────────────────────────────────────────────────────────────

_LOWER_IS_BETTER = {"lpips_model", "niqe_model", "brisque_model",
                    "lpips_bicubic", "niqe_bicubic", "brisque_bicubic"}


def _sort_key(row: dict, metric: str):
    val = row.get(metric, "")
    try:
        f = float(val)
        return f if metric in _LOWER_IS_BETTER else -f  # flip so best is always first
    except (TypeError, ValueError):
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare all experiments — reads JSON outputs and prints a table"
    )
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiment names to include (default: all)")
    parser.add_argument("--sort",        default="psnr_model",
                        help="Metric column to sort by (default: psnr_model)")
    parser.add_argument("--rebuild",     action="store_true",
                        help="Rebuild from JSON files (ignore existing comparison_table.csv)")
    args = parser.parse_args()

    project_dir  = Path(__file__).resolve().parent.parent
    exp_base     = project_dir / "outputs" / "experiments"
    out_csv      = project_dir / "outputs" / "comparison_table.csv"

    if not exp_base.exists():
        print(f"[error] No experiments directory found at {exp_base}")
        print("  Run at least one experiment first:")
        print("    python scripts/run_experiment.py --config configs/default.yaml")
        return

    # ── Discover experiment directories ───────────────────────────────────────
    if args.experiments:
        candidates = [exp_base / name for name in args.experiments]
    else:
        candidates = sorted(p for p in exp_base.iterdir() if p.is_dir())

    rows = []
    skipped = []
    for exp_dir in candidates:
        row = _build_row(exp_dir)
        if row is None:
            skipped.append(exp_dir.name)
        else:
            rows.append(row)

    if skipped:
        print(f"[skip] No run_summary.json in: {', '.join(skipped)}")

    if not rows:
        print("[error] No finished experiments found.  "
              "run_summary.json is written at the end of train.py.")
        return

    # ── Sort ──────────────────────────────────────────────────────────────────
    sort_col = args.sort
    known    = {col for col, *_ in _TABLE_COLS}
    if sort_col not in known:
        print(f"[warn] Unknown sort column '{sort_col}'. "
              f"Defaulting to 'psnr_model'.")
        sort_col = "psnr_model"

    rows.sort(key=lambda r: _sort_key(r, sort_col))

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment Comparison  ({len(rows)} run(s))")
    print(f"  Sorted by: {sort_col}")
    print(f"{'='*60}\n")
    _print_table(rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
