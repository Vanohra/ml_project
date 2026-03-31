"""
prepare_data.py
---------------
Preprocesses DIV2K images into matched HR / LR pairs for super-resolution.

Pipeline for each image:
  HR image  →  [optional blur]  →  bicubic downsample x4
            →  [optional JPEG compression]  →  LR image saved

After running this you will have:
  data/DIV2K/HR_train/   your original high-res images  (you put these here)
  data/DIV2K/LR_train/   matching low-res images        (this script creates these)
  data/DIV2K/HR_val/
  data/DIV2K/LR_val/
  outputs/preprocessing.log   a plain-text summary of what was processed
"""

import io
import logging
from pathlib import Path

from PIL import Image, ImageFilter

# ── Settings — edit these to change behaviour ─────────────────────────────────

SCALE     = 4      # Downsample factor: 4 means the LR image is 4x smaller each side

# Optional degradations — set to True to simulate real-world image quality loss
ADD_BLUR  = True   # Adds a gentle Gaussian blur BEFORE downsampling
                   #   Why: cameras aren't perfectly sharp; this makes training harder
                   #   and forces the model to learn real-world sharpness recovery.
BLUR_RADIUS = 1    # Kernel radius for the blur (1 = very light, 2 = more noticeable)

ADD_JPEG  = True   # Re-saves the LR image as JPEG at low quality, then reloads it
                   #   Why: JPEG compression creates blocky artefacts (like old photos
                   #   or screenshots). Training on these makes your model more robust.
JPEG_QUALITY = 75  # 0 (worst) – 95 (best). 75 gives visible but not severe artefacts.

DATA_DIR  = Path(__file__).parent.parent / "data" / "DIV2K"
LOG_FILE  = Path(__file__).parent.parent / "outputs" / "preprocessing.log"

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Pairs of (HR folder, LR folder) to process
SPLITS = [
    ("HR_train", "LR_train"),
    ("HR_val",   "LR_val"),
]

# ──────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:
    """
    Creates a logger that writes to BOTH the terminal and a log file.

    logging.INFO means we record normal messages.
    logging.WARNING would only record warnings and errors.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prepare_data")
    logger.setLevel(logging.INFO)

    # Format: timestamp  LEVEL  message
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Handler 1: print to terminal
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(fmt)

    # Handler 2: write to log file
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    return logger


def apply_blur(image: Image.Image) -> Image.Image:
    """
    Applies a Gaussian blur to simulate a slightly soft/out-of-focus camera lens.

    ImageFilter.GaussianBlur(radius=1) is very subtle — you probably won't
    notice it visually, but the model will learn to handle it during training.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))


def apply_jpeg_compression(image: Image.Image) -> Image.Image:
    """
    Simulates JPEG compression artefacts by:
      1. Encoding the image to JPEG bytes in memory (no file written yet)
      2. Decoding those bytes back into a PIL image

    This round-trip introduces the blocky '8x8 grid' artefacts that JPEG
    compression creates — especially visible in smooth gradients.

    io.BytesIO() is an in-memory buffer (like a file, but stored in RAM).
    We use it so we don't need to write a temporary file to disk.
    """
    buffer = io.BytesIO()                                    # in-memory file
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY) # encode to JPEG
    buffer.seek(0)                                           # rewind to start
    return Image.open(buffer).copy()                         # decode back


def make_lr_image(hr_image: Image.Image) -> Image.Image:
    """
    Full degradation pipeline:
      1. Optional Gaussian blur    (simulates lens blur)
      2. Bicubic downsample x4     (the core LR step)
      3. Optional JPEG compression (simulates compression artefacts)

    Returns the degraded LR image.
    """
    image = hr_image.copy()

    # Step 1: blur (applied at HR size so it scales proportionally)
    if ADD_BLUR:
        image = apply_blur(image)

    # Step 2: downsample
    new_w = image.width  // SCALE
    new_h = image.height // SCALE
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Step 3: JPEG compression (applied at LR size — matches real-world conditions)
    if ADD_JPEG:
        image = apply_jpeg_compression(image)

    return image


def process_split(hr_folder: Path, lr_folder: Path,
                  logger: logging.Logger) -> dict:
    """
    Processes all images in one split (train or val).
    Returns a dict with counts: processed, skipped, errors.
    """
    lr_folder.mkdir(parents=True, exist_ok=True)

    counts = {"processed": 0, "skipped": 0, "errors": 0}

    # Collect all image files in the HR folder
    image_paths = sorted([
        p for p in hr_folder.iterdir()
        if p.suffix.lower() in SUPPORTED
    ])

    if not image_paths:
        logger.warning("No images found in %s — put HR images there and re-run.", hr_folder)
        return counts

    logger.info("Found %d image(s) in %s", len(image_paths), hr_folder.name)

    for i, hr_path in enumerate(image_paths, start=1):
        lr_path = lr_folder / hr_path.name

        # Skip if LR already exists (avoids re-processing on re-runs)
        if lr_path.exists():
            logger.info("[%d/%d] SKIP  %s  (LR already exists)",
                        i, len(image_paths), hr_path.name)
            counts["skipped"] += 1
            continue

        try:
            # Load HR image, force RGB so we always have 3 colour channels
            hr_image = Image.open(hr_path).convert("RGB")

            # Create the degraded LR version
            lr_image = make_lr_image(hr_image)

            # Save the LR image (PNG keeps it lossless; JPEG is fine too)
            lr_path_png = lr_folder / (hr_path.stem + ".png")
            lr_image.save(lr_path_png)

            logger.info(
                "[%d/%d] OK    %s  HR=%s  LR=%s  blur=%s  jpeg=%s",
                i, len(image_paths), hr_path.name,
                f"{hr_image.width}x{hr_image.height}",
                f"{lr_image.width}x{lr_image.height}",
                ADD_BLUR, ADD_JPEG,
            )
            counts["processed"] += 1

        except Exception as exc:
            # Catch any error (corrupt file, disk full, etc.) without stopping
            logger.error("[%d/%d] ERROR %s — %s", i, len(image_paths), hr_path.name, exc)
            counts["errors"] += 1

    return counts


def main():
    logger = setup_logging()

    # Log the settings being used
    logger.info("=" * 60)
    logger.info("Starting preprocessing")
    logger.info("Scale factor : x%d", SCALE)
    logger.info("Blur         : %s  (radius=%s)", ADD_BLUR, BLUR_RADIUS if ADD_BLUR else "n/a")
    logger.info("JPEG quality : %s  (quality=%s)", ADD_JPEG, JPEG_QUALITY if ADD_JPEG else "n/a")
    logger.info("=" * 60)

    total = {"processed": 0, "skipped": 0, "errors": 0}

    for hr_name, lr_name in SPLITS:
        hr_folder = DATA_DIR / hr_name
        lr_folder = DATA_DIR / lr_name

        logger.info("--- %s  →  %s ---", hr_name, lr_name)
        counts = process_split(hr_folder, lr_folder, logger)

        for key in total:
            total[key] += counts[key]

    # Final summary
    logger.info("=" * 60)
    logger.info("DONE — processed: %d | skipped: %d | errors: %d",
                total["processed"], total["skipped"], total["errors"])
    logger.info("Log saved to: %s", LOG_FILE)
    logger.info("Next step: python scripts/baseline.py")


if __name__ == "__main__":
    main()
