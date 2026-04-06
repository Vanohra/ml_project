"""
extract_virat_frames.py
-----------------------
Extracts frames from VIRAT surveillance videos, applies the fixed surveillance
degradation pipeline to generate matched HR/LR image pairs, and organises
them into an 80/20 split for evaluation and human preference study.

Pipeline (run once; outputs are deterministic for a fixed --seed)
  1. Find every video file in --video_dir  (mp4 / avi / mov / mkv).
  2. For each video, sample --frames_per_video evenly-spaced frame indices,
     with a small seed-controlled jitter so the exact positions are not
     perfectly predictable from the video length alone.
  3. Extract each selected frame as a clean PNG  (this is the HR image).
  4. Apply the surveillance degradation preset from domain_degradation.py
     to produce the matched LR image.  Degradation parameters are sampled
     with seed + global_frame_id so they are stable and logged.
  5. Assign frames to two splits using a shuffled list (seed-controlled):
       quantitative  (80%)  — for PSNR / SSIM / LPIPS evaluation
       preference    (20%)  — for human preference / visual comparison study
  6. Write manifest.json with full provenance for every saved frame.

Reproducibility
  Every random decision — jitter, degradation params, split assignment —
  is controlled by --seed (default 42).  Running the script twice with the
  same seed always produces byte-identical outputs.

Requirements
  pip install opencv-python    # video reading

Usage
  python scripts/extract_virat_frames.py \\
      --video_dir data/VIRAT/videos \\
      --out_dir   data/VIRAT/frames \\
      --seed 42

  # More frames per video:
  python scripts/extract_virat_frames.py \\
      --video_dir data/VIRAT/videos \\
      --frames_per_video 30 \\
      --seed 42

  # Dry run — list found videos and planned frame counts without extracting:
  python scripts/extract_virat_frames.py \\
      --video_dir data/VIRAT/videos \\
      --dry_run

Output layout
  data/VIRAT/frames/
    quantitative/
      hr/   0000_hr.png, 0001_hr.png, ...   (clean extracted frames)
      lr/   0000_lr.png, 0001_lr.png, ...   (surveillance-degraded)
    preference/
      hr/   ...
      lr/   ...
    manifest.json               provenance record for every frame pair
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}

# ── Optional OpenCV import ─────────────────────────────────────────────────────

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _require_cv2():
    if not HAS_CV2:
        print("[error] opencv-python is required to read video files.")
        print("        Install with:  pip install opencv-python")
        sys.exit(1)


# ── Frame index sampling ───────────────────────────────────────────────────────

def sample_frame_indices(total_frames: int, n: int, rng: np.random.RandomState,
                         jitter_fraction: float = 0.3) -> list[int]:
    """
    Sample n frame indices evenly spaced across [0, total_frames-1], with
    a small random jitter within each interval (controlled by rng).

    jitter_fraction  controls how far within each interval the sample can
    deviate from the exact centre.  0.0 = perfectly evenly spaced (no jitter).
    0.5 = sample anywhere within the interval.

    Even spacing ensures frames are spread across the whole video rather than
    clustering at the start.  Jitter ensures two videos with the same length
    don't always yield the same absolute timestamps.
    """
    if n >= total_frames:
        # Fewer frames than requested — just return all of them
        return list(range(total_frames))

    interval = total_frames / n
    indices = []
    for i in range(n):
        centre = interval * (i + 0.5)
        half   = interval * jitter_fraction * 0.5
        jitter = rng.uniform(-half, half)
        idx    = int(round(centre + jitter))
        idx    = max(0, min(total_frames - 1, idx))
        indices.append(idx)

    # Deduplicate while preserving order (rare for large videos)
    seen   = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return sorted(unique)


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frame(cap, frame_idx: int) -> Image.Image | None:
    """
    Seek to frame_idx and read one frame.
    Returns a PIL RGB Image or None on failure.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Degradation (surveillance preset, with seed per frame) ────────────────────

def degrade_frame(hr_img: Image.Image, scale: int, global_id: int,
                  seed: int) -> tuple[Image.Image, dict]:
    """
    Apply the surveillance degradation to hr_img.

    The random state is seeded with (seed + global_id) so degradation
    parameters for each frame are stable and independent of processing order.
    """
    from domain_degradation import degrade_domain

    random.seed(seed + global_id)
    np.random.seed((seed + global_id) & 0xFFFFFFFF)

    lr_img, params = degrade_domain(hr_img, scale=scale, domain="surveillance")

    # Reset to a non-deterministic state so other code is not affected
    random.seed()
    np.random.seed()

    return lr_img, params


# ── Split assignment ───────────────────────────────────────────────────────────

def assign_splits(global_ids: list[int], seed: int,
                  quant_fraction: float = 0.8) -> dict[int, str]:
    """
    Shuffle global_ids with a fixed seed and assign the first quant_fraction
    to 'quantitative' and the rest to 'preference'.

    Returns {global_id: split_name}.
    """
    rng      = np.random.RandomState(seed * 31337)   # separate from frame sampling
    order    = global_ids.copy()
    rng.shuffle(order)
    cut      = max(1, int(len(order) * quant_fraction))
    splits   = {}
    for gid in order[:cut]:
        splits[gid] = "quantitative"
    for gid in order[cut:]:
        splits[gid] = "preference"
    return splits


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_pair(hr_img: Image.Image, lr_img: Image.Image,
              split: str, global_id: int, out_dir: Path) -> tuple[str, str]:
    """Save HR and LR images; return their relative paths."""
    tag     = f"{global_id:04d}"
    hr_path = out_dir / split / "hr" / f"{tag}_hr.png"
    lr_path = out_dir / split / "lr" / f"{tag}_lr.png"
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    hr_img.save(hr_path)
    lr_img.save(lr_path)
    return str(hr_path.relative_to(out_dir)), str(lr_path.relative_to(out_dir))


def serialisable_params(params: dict) -> dict:
    """Convert PIL resampler constants to int so the dict is JSON-serialisable."""
    def _fix(v):
        if isinstance(v, dict):
            return {k: _fix(val) for k, val in v.items()}
        if isinstance(v, Image.Resampling):
            return int(v)
        if hasattr(v, 'value'):        # older Pillow enum fallback
            return int(v.value)
        return v

    return {k: _fix(v) for k, v in params.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract VIRAT frames and generate surveillance HR/LR pairs"
    )
    parser.add_argument("--video_dir",        default="data/VIRAT/videos",
                        help="Directory containing VIRAT .mp4/.avi/.mov/.mkv files")
    parser.add_argument("--out_dir",          default="data/VIRAT/frames",
                        help="Root output directory (default: data/VIRAT/frames)")
    parser.add_argument("--frames_per_video", type=int, default=20,
                        help="Number of frames to sample per video (default: 20)")
    parser.add_argument("--scale",            type=int, default=4,
                        help="SR downscale factor (default: 4)")
    parser.add_argument("--seed",             type=int, default=42,
                        help="Master random seed — controls sampling, degradation, split")
    parser.add_argument("--quant_fraction",   type=float, default=0.8,
                        help="Fraction of frames for quantitative eval (default: 0.8)")
    parser.add_argument("--min_short_side",   type=int, default=256,
                        help="Skip frames whose short side < this after HR crop (default: 256)")
    parser.add_argument("--dry_run",          action="store_true",
                        help="List videos and planned frame counts without extracting")
    args = parser.parse_args()

    if not args.dry_run:
        _require_cv2()

    project_dir = Path(__file__).resolve().parent.parent
    video_dir   = project_dir / args.video_dir if not Path(args.video_dir).is_absolute() \
                  else Path(args.video_dir)
    out_dir     = project_dir / args.out_dir   if not Path(args.out_dir).is_absolute() \
                  else Path(args.out_dir)

    if not video_dir.exists():
        print(f"[error] Video directory not found: {video_dir}")
        print(f"  Place VIRAT videos at:  {video_dir}/")
        return

    # ── Collect video files (sorted for determinism) ──────────────────────────
    videos = sorted(p for p in video_dir.rglob("*")
                    if p.suffix in SUPPORTED_VIDEO_EXTS)
    if not videos:
        print(f"[error] No video files found in {video_dir}")
        print(f"  Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}")
        return

    print(f"Found {len(videos)} video(s) in {video_dir}")

    if args.dry_run:
        for v in videos:
            print(f"  {v.name}")
        print(f"\n  frames_per_video : {args.frames_per_video}")
        print(f"  max total frames : ~{len(videos) * args.frames_per_video}")
        print(f"  seed             : {args.seed}")
        print(f"\n  Re-run without --dry_run to extract.")
        return

    # ── Pass 1: compute planned (video, frame_indices) for every video ────────
    # This must happen before extraction so we have all global IDs
    # and can assign splits before writing any files.
    print("\nPass 1 — planning frame indices...")

    plan = []    # list of {video_path, frame_idx, fps, timestamp_s}
    for v_idx, video_path in enumerate(videos):
        cap          = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        if total_frames <= 0:
            print(f"  [warn] Could not read frame count for {video_path.name} — skipping.")
            continue

        rng     = np.random.RandomState(args.seed + v_idx)
        indices = sample_frame_indices(total_frames, args.frames_per_video, rng)

        for f_idx in indices:
            plan.append({
                "video":      video_path,
                "frame_idx":  f_idx,
                "fps":        fps,
                "timestamp_s": round(f_idx / fps, 3),
            })
        print(f"  {video_path.name:<40s}  {total_frames:>6d} frames  "
              f"-> sample {len(indices)} positions")

    total_planned = len(plan)
    if total_planned == 0:
        print("[error] No frames planned — check video files.");  return

    # ── Assign splits (before extraction, so we can skip I/O failures cleanly) -
    global_ids   = list(range(total_planned))
    split_map    = assign_splits(global_ids, args.seed, args.quant_fraction)
    n_quant      = sum(1 for s in split_map.values() if s == "quantitative")
    n_pref       = total_planned - n_quant
    print(f"\nSplit assignment (seed={args.seed}, fraction={args.quant_fraction})")
    print(f"  quantitative : {n_quant} frames")
    print(f"  preference   : {n_pref} frames")

    # ── Pass 2: extract, degrade, save ────────────────────────────────────────
    print(f"\nPass 2 — extracting and degrading {total_planned} frames "
          f"(scale x{args.scale})...")

    manifest_frames = []
    saved           = 0
    skipped         = 0

    prev_video_path = None
    cap             = None

    for global_id, entry in enumerate(plan):
        video_path = entry["video"]
        frame_idx  = entry["frame_idx"]

        # Reuse the capture object if still on the same video
        if video_path != prev_video_path:
            if cap is not None:
                cap.release()
            cap           = cv2.VideoCapture(str(video_path))
            prev_video_path = video_path

        hr_img = extract_frame(cap, frame_idx)
        if hr_img is None:
            print(f"  [warn] Could not read frame {frame_idx} of {video_path.name} — skipping.")
            skipped += 1
            continue

        # Skip frames that are too small (e.g. corrupt partial reads, black frames)
        short_side = min(hr_img.width, hr_img.height)
        if short_side < args.min_short_side:
            print(f"  [warn] Frame {frame_idx} of {video_path.name} "
                  f"is too small ({hr_img.width}x{hr_img.height}) — skipping.")
            skipped += 1
            continue

        # Degrade with per-frame seed
        lr_img, deg_params = degrade_frame(hr_img, args.scale, global_id, args.seed)

        # Save
        split           = split_map[global_id]
        hr_rel, lr_rel  = save_pair(hr_img, lr_img, split, global_id, out_dir)

        manifest_frames.append({
            "global_id":    global_id,
            "split":        split,
            "video":        video_path.name,
            "frame_idx":    frame_idx,
            "timestamp_s":  entry["timestamp_s"],
            "hr_path":      hr_rel,
            "lr_path":      lr_rel,
            "hr_size":      [hr_img.width, hr_img.height],
            "lr_size":      [lr_img.width, lr_img.height],
            "degradation":  serialisable_params(deg_params),
        })
        saved += 1

        if saved % 10 == 0 or saved == total_planned:
            print(f"  [{saved:4d}/{total_planned}]  {split:<14s}  "
                  f"{video_path.name}  frame {frame_idx}")

    if cap is not None:
        cap.release()

    # ── Write manifest ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed":             args.seed,
        "scale":            args.scale,
        "frames_per_video": args.frames_per_video,
        "quant_fraction":   args.quant_fraction,
        "total_saved":      saved,
        "total_skipped":    skipped,
        "n_quantitative":   sum(1 for f in manifest_frames if f["split"] == "quantitative"),
        "n_preference":     sum(1 for f in manifest_frames if f["split"] == "preference"),
        "videos":           [v.name for v in videos],
        "frames":           manifest_frames,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_q = manifest["n_quantitative"]
    n_p = manifest["n_preference"]
    print(f"\n{'='*58}")
    print(f"  Extraction complete")
    print(f"  Saved    : {saved} frame pairs  (skipped {skipped})")
    print(f"  quantitative/ : {n_q} pairs  -> evaluate_virat.py")
    print(f"  preference/   : {n_p} pairs  -> human preference study")
    print(f"  Manifest : {manifest_path}")
    print(f"{'='*58}")
    print(f"\nNext step — evaluate a trained model:")
    print(f"  python scripts/evaluate_virat.py --experiment <name>")


if __name__ == "__main__":
    main()
