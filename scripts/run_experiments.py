"""
run_experiments.py
------------------
Documents and prints the exact commands for the three controlled experiments
that form the project's scientific contribution.

Running without arguments prints full descriptions of all three experiments.
Running with --run 1/2/3 prints the exact copy-pasteable commands for that
experiment.

Three experiments, three questions:
  Exp 1  — Does any CNN beat bicubic on clean (bicubic) LR?
  Exp 2  — Does surveillance-matched degradation improve SRCNN?
  Exp 3  — Does model capacity (SRResNet) add further gain?

Cross-condition evaluation (run after all three experiments):
  Evaluate ALL models on the same surveillance-degraded val set — including
  SRCNN trained on clean LR (cross-condition) — to quantify the PSNR cost of
  training-distribution mismatch.

Usage (from ml_project/ directory):
  python scripts/run_experiments.py             # print all descriptions
  python scripts/run_experiments.py --run 1     # print Exp 1 commands
  python scripts/run_experiments.py --run 2     # print Exp 2 commands
  python scripts/run_experiments.py --run 3     # print Exp 3 + eval commands
"""

import argparse
from pathlib import Path


# ── Experiment descriptions ────────────────────────────────────────────────────

EXPERIMENTS = {
    1: {
        "title": "Does any CNN beat bicubic on clean LR?",
        "question": (
            "Establishes the simplest possible baseline. No realistic degradation.\n"
            "  LR is pure bicubic downsample. Does a 57k-param CNN improve on bicubic?"
        ),
        "lr": "Pure bicubic downscale only (prepare_data.py). No blur, noise, or JPEG.",
        "model": "SimpleSRCNN (~57k params)",
        "epochs": 50,
        "ckpt": "outputs/checkpoints_srcnn_clean/",
        "log":  "outputs/training_log_srcnn_clean.csv",
        "notes": (
            "This is the simplest possible SR experiment. SimpleSRCNN takes a\n"
            "  bicubic-pre-upsampled LR image and learns small residual corrections.\n"
            "  Beating bicubic here shows a 3-layer CNN can learn meaningful sharpening."
        ),
    },
    2: {
        "title": "Does surveillance-matched degradation improve SRCNN?",
        "question": (
            "The core scientific question. Same model, same capacity, different training\n"
            "  distribution. Isolates the effect of degradation modeling."
        ),
        "lr": "Surveillance preset: blur (0.8-2.5) + noise (5-30, 90%) + JPEG (30-70, 90%).",
        "model": "SimpleSRCNN (~57k params, same as Exp 1)",
        "epochs": 50,
        "ckpt": "outputs/checkpoints_srcnn_surv/",
        "log":  "outputs/training_log_srcnn_surv.csv",
        "notes": (
            "Trained on exactly the same LR resolution and model, but with realistic\n"
            "  surveillance degradation instead of clean bicubic. When both are evaluated\n"
            "  on surveillance-degraded val images, the gap reveals the cost of mismatch."
        ),
    },
    3: {
        "title": "Does model capacity add further gain?",
        "question": (
            "With degradation distribution matched, does a 957k-param model with\n"
            "  residual blocks and learned upsampling further improve results?"
        ),
        "lr": "Surveillance preset for both models (SRResNet uses same preset as Exp 2).",
        "model": "SimpleSRCNN (Exp 2 ckpt reused) vs SRResNet (~957k params)",
        "epochs": "50 (SRCNN, reused from Exp 2) / 100 (SRResNet)",
        "ckpt": "outputs/checkpoints_srresnet/",
        "log":  "outputs/training_log_srresnet.csv",
        "notes": (
            "SRResNet processes features at LR resolution (16x fewer pixels per conv),\n"
            "  uses 8 residual blocks, and learns upsampling via PixelShuffle instead of\n"
            "  a fixed bicubic formula. If SRResNet wins, architecture depth matters beyond\n"
            "  just degradation matching."
        ),
    },
}


def print_all_descriptions() -> None:
    print("\n" + "=" * 70)
    print("  Does Realistic Degradation Modeling Improve Super-Resolution?")
    print("  CSC 4850 — Three Controlled Experiments")
    print("=" * 70)

    for num, exp in EXPERIMENTS.items():
        print(f"\n{'-' * 70}")
        print(f"  Experiment {num}: {exp['title']}")
        print(f"{'-' * 70}")
        print(f"  Research question  : {exp['question']}")
        print(f"  LR generation      : {exp['lr']}")
        print(f"  Model(s)           : {exp['model']}")
        print(f"  Epochs             : {exp['epochs']}")
        print(f"  Checkpoint dir     : {exp['ckpt']}")
        print(f"  CSV log            : {exp['log']}")
        print(f"\n  Context:\n  {exp['notes']}")

    print(f"\n{'=' * 70}")
    print("  Cross-condition evaluation (run after all three experiments):")
    print(f"{'=' * 70}")
    print("  Evaluates ALL models on surveillance-degraded val images — including")
    print("  SRCNN trained on clean LR. The result table quantifies:")
    print("    (a) PSNR cost of training-distribution mismatch (Exp 1 vs Exp 2)")
    print("    (b) Additional gain from deeper architecture (Exp 2 vs Exp 3)")
    print()
    print("  Run:  python scripts/evaluate.py --cross_condition \\")
    print("            --srcnn_clean_ckpt outputs/checkpoints_srcnn_clean/best.pth \\")
    print("            --srcnn_surv_ckpt  outputs/checkpoints_srcnn_surv/best.pth \\")
    print("            --srresnet_ckpt    outputs/checkpoints_srresnet/best.pth \\")
    print("            --hr_dir           data/DIV2K/HR_valid \\")
    print("            --n_samples        5")
    print("=" * 70)


def print_exp1_commands() -> None:
    exp = EXPERIMENTS[1]
    print(f"\n{'=' * 70}")
    print(f"  Experiment 1 — {exp['title']}")
    print(f"{'=' * 70}")
    print()
    print("  Step 1 — generate pre-saved bicubic LR images (once):")
    print()
    print("    python scripts/prepare_data.py")
    print()
    print("  Step 2 — train SimpleSRCNN on clean bicubic LR:")
    print()
    print("    python scripts/train.py \\")
    print("        --use_online_degradation False \\")
    print("        --use_surv_preset False \\")
    print(f"        --checkpoint_dir {exp['ckpt']} \\")
    print(f"        --log_file {exp['log']}")
    print()
    print("  Step 3 — run bicubic baseline (reference):")
    print()
    print("    python scripts/baseline.py")
    print()
    print("  Step 4 — evaluate (in-distribution, bicubic LR):")
    print()
    print("    python scripts/evaluate.py \\")
    print(f"        --srcnn_surv_ckpt {exp['ckpt']}best.pth \\")
    print("        --hr_dir data/DIV2K/HR_valid")
    print(f"\n{'=' * 70}")


def print_exp2_commands() -> None:
    exp2 = EXPERIMENTS[2]
    exp1 = EXPERIMENTS[1]
    print(f"\n{'=' * 70}")
    print(f"  Experiment 2 — {exp2['title']}")
    print(f"{'=' * 70}")
    print()
    print("  Train SimpleSRCNN on surveillance-degraded LR:")
    print()
    print("    python scripts/train.py \\")
    print("        --use_online_degradation True \\")
    print("        --use_surv_preset True \\")
    print(f"        --checkpoint_dir {exp2['ckpt']} \\")
    print(f"        --log_file {exp2['log']}")
    print()
    print("  Cross-condition evaluation (Exp 1 vs Exp 2 on surveillance data):")
    print()
    print("    python scripts/evaluate.py \\")
    print(f"        --srcnn_clean_ckpt {exp1['ckpt']}best.pth \\")
    print(f"        --srcnn_surv_ckpt  {exp2['ckpt']}best.pth \\")
    print("        --hr_dir           data/DIV2K/HR_valid \\")
    print("        --cross_condition")
    print(f"\n{'=' * 70}")


def print_exp3_commands() -> None:
    exp1 = EXPERIMENTS[1]
    exp2 = EXPERIMENTS[2]
    exp3 = EXPERIMENTS[3]
    print(f"\n{'=' * 70}")
    print(f"  Experiment 3 — {exp3['title']}")
    print(f"{'=' * 70}")
    print()
    print("  SimpleSRCNN checkpoint reused from Experiment 2.")
    print("  Train SRResNet on surveillance preset (L1 loss, 100 epochs):")
    print()
    print("    python scripts/train_srresnet.py")
    print()
    print("  Full cross-condition evaluation (all three models):")
    print()
    print("    python scripts/evaluate.py \\")
    print(f"        --srcnn_clean_ckpt  {exp1['ckpt']}best.pth \\")
    print(f"        --srcnn_surv_ckpt   {exp2['ckpt']}best.pth \\")
    print(f"        --srresnet_ckpt     {exp3['ckpt']}best.pth \\")
    print("        --hr_dir            data/DIV2K/HR_valid \\")
    print("        --n_samples         5 \\")
    print("        --cross_condition")
    print(f"\n{'=' * 70}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment documentation and command printer for CSC 4850 SR project."
    )
    parser.add_argument(
        "--run",
        choices=["1", "2", "3"],
        default=None,
        help="Print commands for experiment 1, 2, or 3. "
             "Omit to print all descriptions.",
    )
    args = parser.parse_args()

    if args.run is None:
        print_all_descriptions()
        print("\n  To print commands for a specific experiment:")
        print("    python scripts/run_experiments.py --run 1")
        print("    python scripts/run_experiments.py --run 2")
        print("    python scripts/run_experiments.py --run 3")
        return

    if args.run == "1":
        print_exp1_commands()
    elif args.run == "2":
        print_exp2_commands()
    elif args.run == "3":
        print_exp3_commands()


if __name__ == "__main__":
    main()
