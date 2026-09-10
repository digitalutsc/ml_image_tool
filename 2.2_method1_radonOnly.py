r"""
STEP 9 — Rotation correction, OPTION 1 of 3: Radon transform only.

WHEN TO USE
-----------
Only when the corpus is known to contain rotations 0 & 90 (CCW) or 0 & 270
(CCW) — i.e. every wrongly-rotated page is exactly ONE quarter turn away
and always in the SAME direction.  The Radon transform tells HORIZONTAL
(text lines vertical, rotation +-90) from VERTICAL (text lines horizontal,
rotation 0/180); it CANNOT tell +90 from -90, and it cannot detect 180 at
all.  For mixed corpora use option 2 (2.2_method2_radonPlusUpDownCNN.py) or
option 3 (2.2_method3_HVPlusUpDownCNN.py) instead.

!!! READ BEFORE RUNNING !!!
---------------------------
Detection runs on the MIRROR SLICES (step 8), and whenever a slice is
classified as HORIZONTAL, this script rotates BOTH the slice and the
matching original image by the SAME quarter turn:

    --rotation 90ccw   turn every horizontal image 90 deg COUNTER-CLOCKWISE
                       (the default; original script default)
    --rotation 90cw    turn every horizontal image 90 deg CLOCKWISE

Choose the direction that makes YOUR wrongly-rotated pages upright.  If you
pick the wrong one, every horizontal page ends up on its opposite side —
run with --dry-run first on a small mirror folder if unsure.

Originals that never got a mirror slice (step 8 skips pages with no good
text slice) are NOT touched; they are listed in the report for manual
rotation.

Usage
-----
    python 2.2_method1_radonOnly.py --mirror "D:\flat_mirror" --original "D:\flat" --rotation 90ccw

Omit an argument and the script will prompt you for it.
"""

import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from skimage.transform import radon

from workflow_common import (
    cpu_workers,
    find_original_for_slice,
    gather_image_files,
    prompt_if_missing,
    rotate_pair,
    unmirrored_originals,
)

warnings.filterwarnings("ignore")


def orientation_detect(imgpath, angle_interval, line_space, scale_down, method):
    """Radon-based HORIZONTAL vs VERTICAL detector (unchanged core logic of
    the original script).  Returns True = VERTICAL (0/180), False =
    HORIZONTAL (+-90)."""
    image = cv2.imread(imgpath)
    height, width, _ = np.shape(image)

    # Crop image to avoid extremely obvious borders which may result in a
    # high magnitude low frequency wave after FFT
    image_cropped = image[height // 7: 6 * height // 7, width // 7: 6 * width // 7]
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 50, 120, apertureSize=3)
    # Demean; make the brightness extend above and below zero
    gray = gray - np.mean(gray)

    # Slice into four parts to remove the central crease which may result in
    # a high magnitude low frequency wave after FFT
    gray_height, gray_width = np.shape(gray)
    small_height = 9 * gray_height // 20
    small_width = 9 * gray_width // 20
    image1 = gray[0: small_height, 0: small_width]
    image2 = gray[0: small_height, -small_width:]
    image3 = gray[-small_height:, 0: small_width]
    image4 = gray[-small_height:, -small_width:]

    new_height = small_height * 2
    new_width = small_width * 2
    no_mid_gray = np.zeros((new_height, new_width))
    no_mid_gray[0:small_height, 0:small_width] = image1              # Top-left
    no_mid_gray[0:small_height, small_width:] = image2               # Top-right
    no_mid_gray[small_height:, 0:small_width] = image3               # Bottom-left
    no_mid_gray[small_height:, small_width:] = image4                # Bottom-right
    resized_image = cv2.resize(no_mid_gray,
                               (new_width // scale_down, new_height // scale_down))

    # Discrete Radon transform every 'angle_interval' degrees
    angles = np.arange(0, 180, angle_interval).tolist()
    sinogram = radon(resized_image, angles)

    # Find the projection row whose spectrum has the most dominant line
    # frequency inside the given frequency interval
    interval = 2 * line_space
    special_rows = []
    y, _ = np.shape(sinogram)
    Tsino = sinogram.transpose()
    for row in enumerate(Tsino):
        normalized_row = row[1] - np.mean(row[1])
        spectrum = np.abs(np.fft.fft(normalized_row)) / y
        spectrum = spectrum[y // interval: y // 4]
        if method == "peak":
            peak = np.max(spectrum)
            avg_magnitude = np.mean(spectrum)
            special_rows.append(peak / avg_magnitude)
        elif method == "max":
            special_rows.append(np.max(spectrum))
        elif method == "var":
            special_rows.append(np.var(spectrum))

    # Angles between 45 and 135 (index-wise) mean the strongest periodic
    # projection is vertical -> the text lines are vertical -> the IMAGE is
    # rotated +-90 (the radon implementation rotates the image by 90 during
    # the transform, hence the swapped reading).
    result_set = []
    angle_accum = 0
    index_accum = 0
    while angle_accum < 135:
        if angle_accum > 45:
            result_set.append(index_accum)
        angle_accum += angle_interval
        index_accum += 1
    return np.argmax(special_rows) in result_set


def process_slice(task):
    """Worker: detect on the slice; rotate slice+original when horizontal."""
    (slice_path, mirror_root, original_root, rotation, angle_interval,
     line_space, scale_down, method, dry_run) = task
    try:
        is_vertical = orientation_detect(slice_path, angle_interval,
                                         line_space, scale_down, method)
    except Exception as exc:  # noqa: BLE001
        return slice_path, "error", f"detection_failed: {exc}"
    if is_vertical:
        return slice_path, "vertical", "untouched"
    original = find_original_for_slice(slice_path, mirror_root, original_root)
    if dry_run:
        return slice_path, "horizontal", f"would_rotate_{rotation}" + \
            ("" if original else " (original not found)")
    slice_ok, original_ok = rotate_pair(slice_path, original, rotation)
    note = f"rotated_{rotation}"
    if not original:
        note += " (original not found)"
    elif not original_ok:
        note += " (original rotation FAILED)"
    if not slice_ok:
        note += " (slice rotation FAILED)"
    return slice_path, "horizontal", note


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 9 / rotation option 1: Radon-only correction via the mirror tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mirror", help="Mirror tree built by 2.2_0_mirror_text_slices.py.")
    ap.add_argument("--original", help="Flattened original tree from step 1.")
    ap.add_argument("--rotation", choices=["90ccw", "90cw"], default="90ccw",
                    help="Quarter turn applied to every HORIZONTAL image. 90ccw if the "
                         "wrong pages need a counter-clockwise turn to become upright, "
                         "90cw if they need a clockwise one.")
    ap.add_argument("--angle-interval", type=int, default=15,
                    help="Radon angle step in degrees (original default).")
    ap.add_argument("--line-space", type=int, default=65,
                    help="Expected line spacing in pixels (original default).")
    ap.add_argument("--scale-down", type=int, default=1,
                    help="Down-scale factor before the Radon transform.")
    ap.add_argument("--method", choices=["peak", "max", "var"], default="max",
                    help="Row-scoring method inside the detector (original default).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel processes. Default: all CPU cores.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect and report only; rotate nothing.")
    return ap.parse_args()


def main():
    args = parse_args()

    mirror_root = prompt_if_missing(args.mirror, "Mirror folder (step-8 output)")
    original_root = prompt_if_missing(args.original, "Flattened original folder (step-1 output)")
    if not mirror_root or not original_root or not os.path.isdir(mirror_root) \
            or not os.path.isdir(original_root):
        print("[Error] Valid --mirror and --original folders are required.")
        sys.exit(1)

    print("!" * 70)
    print(f"[NOTICE] Horizontal images will be rotated {args.rotation.upper()}.")
    print("         90ccw = counter-clockwise quarter turn (default, as in the")
    print("                 original script); use it when the wrong pages need")
    print("                 a CCW turn to become upright.")
    print("         90cw  = clockwise quarter turn; use it when the wrong pages")
    print("                 need a CW turn to become upright.")
    print("         Radon cannot tell +90 from -90 by itself — choose based on")
    print("         what YOUR corpus needs. Use --dry-run when unsure.")
    print("!" * 70)

    slices = [p for p in gather_image_files(mirror_root)
              if p.lower().endswith(".png")]
    if not slices:
        print(f"[Error] No mirror slices (*.png) found under {mirror_root}. "
              f"Run 2.2_0_mirror_text_slices.py first.")
        sys.exit(1)

    no_mirror = unmirrored_originals(mirror_root, original_root)
    print(f"[Info] {len(slices):,} mirror slice(s) | {len(no_mirror):,} original(s) "
          f"without a slice (these stay untouched).")

    workers = cpu_workers(args.workers)
    tasks = [(s, mirror_root, original_root, args.rotation, args.angle_interval,
              args.line_space, args.scale_down, args.method, args.dry_run)
             for s in slices]

    report_path = os.path.join(mirror_root, "_method1_report.csv")
    counts = {"vertical": 0, "horizontal": 0, "error": 0}
    start_time = time.time()

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "decision", "action"])
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_slice, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                slice_path, decision, action = fut.result()
                counts[decision] = counts.get(decision, 0) + 1
                writer.writerow([slice_path, decision, action])
                if i % 50 == 0 or i == len(tasks):
                    elapsed = time.time() - start_time
                    eta = elapsed / i * (len(tasks) - i)
                    print(f"[Progress] {i:,}/{len(tasks):,} | vertical "
                          f"{counts['vertical']:,} | horizontal {counts['horizontal']:,} "
                          f"| errors {counts['error']:,} | ETA {eta:,.0f}s")

    if no_mirror:
        with open(os.path.join(mirror_root, "_method1_no_slice.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["original_without_slice"])
            w.writerows([[p] for p in no_mirror])

    print("-" * 70)
    print(f"[Done] vertical (untouched) : {counts['vertical']:,}")
    print(f"[Done] horizontal           : {counts['horizontal']:,} "
          f"({'would be' if args.dry_run else ''} rotated {args.rotation})")
    print(f"[Done] detection errors     : {counts['error']:,} (see report)")
    print(f"[Report] {report_path}")
    if no_mirror:
        print(f"[Manual] {len(no_mirror):,} original(s) had no mirror slice -> "
              f"rotate manually; list: _method1_no_slice.csv")
    print("[Next] 3_cropping.py on the corrected flattened tree, then the "
          "step-4 split/merge QA scripts.")


if __name__ == "__main__":
    main()
