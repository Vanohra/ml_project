"""
run_experiment.py
-----------------
Runs one complete experiment: train → paired eval → no-ref eval → results row.

All steps are optional and controlled by flags so you can re-run just the eval
on an already-trained model, or train without evaluating yet.

What it does
  1. Reads the config and resolves the experiment name.
  2. Runs train.py  (unless --skip_train).
  3. Runs evaluate_paired.py on the DIV2K validation set
     (unless --skip_eval or HR/LR dirs not found).
  4. Runs evaluate_noref.py on a real-image directory
     (only when --noref_dir is supplied and piq is installed).
  5. Collects results from the JSON files written by those scripts and
     appends one row to outputs/results.csv.

The CSV is the single source of truth for comparing experiments.
Run compare_experiments.py separately to print/regenerate it from scratch.

Usage
  # Full pipeline — train + eval on DIV2K + append results:
  python scripts/run_experiment.py --config configs/default.yaml

  # Override experiment name without editing YAML:
  python scripts/run_experiment.py \\
      --config configs/default.yaml \\
      --experiment baseline_l1_v2

  # Add no-reference eval on RealSRSet:
  python scripts/run_experiment.py \\
      --config configs/default.yaml \\
      --noref_dir data/RealSRSet

  # Skip training (model already trained) and just run eval + append row:
  python scripts/run_experiment.py \\
      --config configs/default.yaml \\
      --skip_train

  # Skip eval entirely (just train):
  python scripts/run_experiment.py \\
      --config configs/default.yaml \\
      --skip_eval

  # Evaluate on a non-default checkpoint:
  python scripts/run_experiment.py \\
      --config configs/default.yaml \\
      --checkpoint last
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess, stream its output live, return exit code."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, executable=sys.executable)
    return result.returncode


def _load_json(path: Path) -> dict:
    """Return parsed JSON or {} if the file does not exist."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _get(d: dict, *keys, default=None):
    """Safe nested dict getter: _get(d, 'a', 'b', 'c', default=0)."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


# ── CSV helpers ───────────────────────────────────────────────────────────────

RESULTS_CSV_COLUMNS = [
    # identity
    "experiment", "model", "domain", "curriculum", "loss_pixel",
    "loss_perceptual", "num_epochs", "seed",
    # training
    "best_psnr_train",
    # paired eval (DIV2K)
    "psnr_bicubic", "psnr_model",
    "ssim_bicubic", "ssim_model",
    "lpips_bicubic", "lpips_model",
    # no-reference eval (RealSRSet or similar)
    "niqe_bicubic", "niqe_model",
    "brisque_bicubic", "brisque_model",
    # provenance
    "config_path",
]


def _append_csv_row(csv_path: Path, row: dict) -> None:
    """Append row to csv_path, writing the header first if the file is new."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS,
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        # Fill any missing keys with empty string so the row always has all columns
        full_row = {col: row.get(col, "") for col in RESULTS_CSV_COLUMNS}
        writer.writerow(full_row)
    print(f"\n[results] Row appended → {csv_path}")


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(exp_name: str, cfg: dict, exp_dir: Path,
               config_path: Path) -> dict:
    """Read all JSON outputs and assemble one results-CSV row."""
    row: dict = {}

    # ── Identity from config ──────────────────────────────────────────────────
    row["experiment"]       = exp_name
    row["model"]            = cfg.get("model", {}).get("name", "srcnn")
    row["domain"]           = cfg.get("data", {}).get("domain") or "generic"
    row["curriculum"]       = ("yes"
                               if cfg.get("curriculum", {}).get("enabled", False)
                               else "no")
    loss_cfg                = cfg.get("loss", {})
    row["loss_pixel"]       = loss_cfg.get("pixel", {}).get("type", "l1")
    row["loss_perceptual"]  = ("yes"
                               if loss_cfg.get("perceptual", {}).get("enabled", False)
                               else "no")
    row["num_epochs"]       = cfg.get("training", {}).get("num_epochs", "")
    row["seed"]             = cfg.get("experiment", {}).get("seed", "")
    row["config_path"]      = str(config_path)

    # ── Training summary ──────────────────────────────────────────────────────
    summary = _load_json(exp_dir / "run_summary.json")
    row["best_psnr_train"] = summary.get("best_psnr_db", "")

    # ── Paired eval (eval_paired_summary.json) ────────────────────────────────
    paired = _load_json(exp_dir / "metrics" / "eval_paired_summary.json")
    row["psnr_bicubic"]  = _get(paired, "bicubic", "psnr",  "mean", default="")
    row["psnr_model"]    = _get(paired, "model",   "psnr",  "mean", default="")
    row["ssim_bicubic"]  = _get(paired, "bicubic", "ssim",  "mean", default="")
    row["ssim_model"]    = _get(paired, "model",   "ssim",  "mean", default="")
    row["lpips_bicubic"] = _get(paired, "bicubic", "lpips", "mean", default="")
    row["lpips_model"]   = _get(paired, "model",   "lpips", "mean", default="")

    # ── No-ref eval (eval_noref_summary.json) ────────────────────────────────
    noref = _load_json(exp_dir / "metrics" / "eval_noref_summary.json")
    row["niqe_bicubic"]    = _get(noref, "bicubic", "niqe",    "mean", default="")
    row["niqe_model"]      = _get(noref, "model",   "niqe",    "mean", default="")
    row["brisque_bicubic"] = _get(noref, "bicubic", "brisque", "mean", default="")
    row["brisque_model"]   = _get(noref, "model",   "brisque", "mean", default="")

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run one experiment end-to-end: train + eval + results row"
    )
    parser.add_argument("--config",      required=True,
                        help="Path to the experiment YAML config")
    parser.add_argument("--experiment",  default=None,
                        help="Override experiment name from config")
    parser.add_argument("--checkpoint",  default="best",
                        choices=["best", "last"],
                        help="Checkpoint to use for evaluation (default: best)")
    parser.add_argument("--noref_dir",   default=None,
                        help="Directory of real-world LR images for no-ref eval "
                             "(e.g. data/RealSRSet). Skipped if not provided.")
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip_eval",   action="store_true",
                        help="Skip evaluation (train only, no results row)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    config_path = (project_dir / args.config
                   if not Path(args.config).is_absolute()
                   else Path(args.config))

    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    cfg      = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_name = args.experiment or cfg["experiment"]["name"]
    exp_dir  = project_dir / cfg["output"]["base_dir"] / exp_name

    # Resolve data dirs for eval (relative paths → absolute)
    data_cfg = cfg.get("data", {})
    hr_valid = project_dir / data_cfg.get("hr_valid_dir", "data/DIV2K/HR_valid")
    lr_valid = project_dir / data_cfg.get("lr_valid_dir", "data/DIV2K/LR_valid")

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [
            sys.executable, str(project_dir / "scripts" / "train.py"),
            "--config", str(config_path),
        ]
        if args.experiment:
            train_cmd += ["--experiment", args.experiment]

        rc = _run(train_cmd, "STEP 1 / 3 — Training")
        if rc != 0:
            print(f"\n[error] Training failed (exit code {rc}). Stopping.")
            sys.exit(rc)
    else:
        print("\n[skip] Training skipped (--skip_train).")

    if args.skip_eval:
        print("\n[skip] Evaluation skipped (--skip_eval). No results row written.")
        return

    # ── Step 2: Paired evaluation ─────────────────────────────────────────────
    # Three cases:
    #
    # A) ConditionedSRResNet + domain set → --regen_lr mode.
    #    HR images are re-degraded on the fly; actual conditioning vectors are
    #    captured and passed to the model.  No LR_valid directory is needed.
    #    This is the most faithful eval for the proposal model.
    #
    # B) Pre-saved LR_valid directory exists → standard paired eval.
    #    Conditioning falls back to zero vector for conditioned models.
    #
    # C) Neither A nor B → skip and point to val_psnr.json.
    use_online    = data_cfg.get("use_online_degradation", True)
    has_hr_valid  = hr_valid.exists()
    has_lr_valid  = lr_valid.exists()
    model_name    = cfg.get("model", {}).get("name", "")
    is_conditioned = model_name == "conditioned_srresnet"
    domain         = data_cfg.get("domain")

    if is_conditioned and domain and has_hr_valid:
        # Case A: re-degrade from HR, true conditioning
        eval_paired_cmd = [
            sys.executable,
            str(project_dir / "scripts" / "evaluate_paired.py"),
            "--experiment", exp_name,
            "--hr_dir",     str(hr_valid),
            "--regen_lr",
            "--checkpoint", args.checkpoint,
        ]
        rc = _run(eval_paired_cmd,
                  "STEP 2 / 3 — Paired evaluation with true conditioning (--regen_lr)")
        if rc != 0:
            print(f"[warn] Paired eval returned exit code {rc} — results may be incomplete.")

    elif has_hr_valid and has_lr_valid:
        # Case B: standard paired eval with pre-saved LR files
        eval_paired_cmd = [
            sys.executable,
            str(project_dir / "scripts" / "evaluate_paired.py"),
            "--experiment", exp_name,
            "--hr_dir",     str(hr_valid),
            "--lr_dir",     str(lr_valid),
            "--checkpoint", args.checkpoint,
        ]
        rc = _run(eval_paired_cmd, "STEP 2 / 3 — Paired evaluation (PSNR / SSIM / LPIPS)")
        if rc != 0:
            print(f"[warn] Paired eval returned exit code {rc} — results may be incomplete.")

    elif use_online and not has_lr_valid:
        # Case C: online degradation, no LR files, model not conditioned (or no domain)
        print(f"\n[skip] Paired eval skipped — this experiment uses online degradation.")
        print(f"  No pre-saved LR validation files at: {lr_valid}")
        print(f"  Best validation PSNR is in:")
        print(f"    {exp_dir / 'metrics' / 'val_psnr.json'}  (written each epoch)")
        print(f"    {exp_dir / 'run_summary.json'}           (best_psnr_db field)")
        print(f"  To also get SSIM/LPIPS with true conditioning (conditioned models):")
        print(f"    python scripts/evaluate_paired.py \\")
        print(f"        --experiment {exp_name} \\")
        print(f"        --hr_dir {hr_valid} \\")
        print(f"        --regen_lr")
    else:
        print(f"\n[skip] Paired eval skipped — directories not found:")
        print(f"         HR: {hr_valid}  (exists: {has_hr_valid})")
        print(f"         LR: {lr_valid}  (exists: {has_lr_valid})")

    # ── Step 3: No-reference evaluation ──────────────────────────────────────
    if args.noref_dir:
        noref_path = (project_dir / args.noref_dir
                      if not Path(args.noref_dir).is_absolute()
                      else Path(args.noref_dir))
        if noref_path.exists():
            eval_noref_cmd = [
                sys.executable,
                str(project_dir / "scripts" / "evaluate_noref.py"),
                "--experiment", exp_name,
                "--img_dir",    str(noref_path),
                "--checkpoint", args.checkpoint,
            ]
            rc = _run(eval_noref_cmd,
                      "STEP 3 / 3 — No-reference evaluation (NIQE / BRISQUE)")
            if rc != 0:
                print(f"[warn] No-ref eval returned exit code {rc} — "
                      "results may be incomplete.")
        else:
            print(f"\n[skip] No-ref eval skipped — directory not found: {noref_path}")
    else:
        print("\n[skip] No-ref eval skipped (pass --noref_dir to enable).")

    # ── Step 4: Collect results and append CSV row ────────────────────────────
    row      = _build_row(exp_name, cfg, exp_dir, config_path)
    csv_path = project_dir / "outputs" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _append_csv_row(csv_path, row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Experiment complete: {exp_name}")
    print(f"  Best PSNR (training) : {row.get('best_psnr_train', 'n/a')} dB")
    print(f"  PSNR model vs bicubic: "
          f"{row.get('psnr_model', 'n/a')} vs {row.get('psnr_bicubic', 'n/a')} dB")
    print(f"  Results CSV          : {csv_path}")
    print(f"  To compare all runs  : python scripts/compare_experiments.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
