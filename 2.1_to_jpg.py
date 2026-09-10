r"""
OPTIONAL STEP 3 of the workflow (training-data preparation).

Converts every image under a source tree to .jpg, preserving the relative
directory structure.  Typical use: turning the RVL-CDIP download into jpgs
so 2.1_slice_text_square.py (optional step 4) can slice them into text
patches for training/fine-tuning the 2.2_* orientation models.

Notes
-----
* Alpha channels (PNG transparency) are flattened to RGB automatically.
* Existing .jpg destination files are NOT overwritten unless you pass
  --overwrite (so the script can safely be re-run after an interruption).
* Parallelism scales with your CPU core count (override with --workers).

Usage
-----
    python 2.1_to_jpg.py --src "D:\RVL-CDIP" --dst "D:\RVL-CDIP_jpg"

Omit an argument and the script will prompt you for it.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

from workflow_common import cpu_workers, prompt_if_missing

VALID_EXTENSIONS = {".png", ".jpeg", ".jpg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def convert_image(src_file_path, src_dir, dst_dir, quality, overwrite):
    """Convert one image to jpg, mirroring the relative directory layout."""
    try:
        rel_path = os.path.relpath(src_file_path, src_dir)
        dst_file_path = os.path.splitext(os.path.join(dst_dir, rel_path))[0] + ".jpg"
        if not overwrite and os.path.exists(dst_file_path):
            return None  # already converted in a previous run
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        with Image.open(src_file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst_file_path, "JPEG", quality=quality)
        return rel_path
    except Exception as exc:  # noqa: BLE001 - report and keep going
        return f"FAILED {src_file_path}: {exc}"


def main():
    ap = argparse.ArgumentParser(
        description="Optional step 3: batch-convert an image tree to jpg "
                    "(e.g. the RVL dataset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", help="Source root folder (e.g. the RVL download).")
    ap.add_argument("--dst", help="Destination root folder for the jpg copies.")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality 1-100.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert even if the destination jpg already exists.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel processes. Default: all CPU cores.")
    args = ap.parse_args()

    src_dir = prompt_if_missing(args.src, "Source folder to convert to jpg")
    dst_dir = prompt_if_missing(args.dst, "Destination folder for jpg output")
    if not src_dir or not dst_dir:
        print("[Error] Both source and destination are required.")
        return
    src_dir, dst_dir = os.path.abspath(src_dir), os.path.abspath(dst_dir)
    if not os.path.isdir(src_dir):
        print(f"[Error] Source folder does not exist: {src_dir}")
        return

    files_to_process = []
    for root, _, files in os.walk(src_dir):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print(f"[Error] No images found in '{src_dir}'.")
        return

    workers = cpu_workers(args.workers)
    print(f"[Info] {len(files_to_process):,} images to convert on {workers} "
          f"process(es), quality={args.quality}.")

    done = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_image, p, src_dir, dst_dir,
                                   args.quality, args.overwrite)
                   for p in files_to_process]
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                skipped += 1
            elif result.startswith("FAILED"):
                failed += 1
                print(f"[Warn] {result}")
            else:
                done += 1
            total = done + skipped + failed
            if total % 1000 == 0:
                print(f"[Progress] {total:,}/{len(files_to_process):,} ...")

    print("-" * 60)
    print(f"[Done] Converted {done:,} | already existed (skipped) {skipped:,} | "
          f"failed {failed}.")
    print("[Next] Optional step 4: run 2.1_slice_text_square.py with "
          f"--root \"{dst_dir}\" to slice the jpgs into text patches.")


if __name__ == "__main__":
    main()
