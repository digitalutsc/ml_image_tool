r"""
STEP 8 of the workflow — build the "mirror" tree of text slices.

What it does
------------
For every image in the FLATTENED tree produced by 1_gather_images.py, run
the slicing algorithm of 2.1_slice_text_square.py (square patches scored by
the projection-band model 2.1_best_projection_band_2.keras) and store the
best 256x256 text patch as a PNG that MIRRORS the flattened layout:

    <flattened>/0/0@@!!!!!!@@000123.tif   ->   <mirror>/0/0@@!!!!!!@@000123.png

When no good text slice can be found (its text-probability stays below
--min-prob-text), the image is NOT mirrored: it is simply skipped and
logged.  The rotation methods (steps 9-11) only touch images that have a
mirror slice, so every un-mirrored page must be rotated manually — the
printed summary tells you how many there are and where the list is.

NOTE — unlike the training-data slicing runs (optional steps 4 and 6, which
route vertical-text patches to the review folder because the training set
only wants horizontal-text slices), the mirror keeps slices whose text
lines run VERTICALLY: those slices are exactly how the rotation methods
recognize +-90-rotated pages.  A page is mirrored whenever its best patch
looks like TEXT at all.

Why: the slices are small and text-focused, so detection is much faster
and more reliable than on the full pages; and because slice and original
share the relative path, "mirror-lookup" rotation is trivial.

Re-running / resuming
---------------------
Already-mirrored images (existing <stem>.png) are skipped automatically,
so you can just re-run this script after an interruption or after adding
more images to the flattened tree.

Usage
-----
    python 2.2_0_mirror_text_slices.py --original "D:\flat" --mirror "D:\flat_mirror"

Omit an argument and the script will prompt you for it.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Import the slicing algorithm from 2.1_slice_text_square.py (its module
# name is not importable, so load it via importlib by file path).
import importlib.util

from workflow_common import (
    cpu_workers,
    gather_image_files,
    load_keras_model,
    prompt_if_missing,
    resolve_model_path,
)

_spec = importlib.util.spec_from_file_location(
    "slice_text_square",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.1_slice_text_square.py"),
)
slice_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slice_lib)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 8: mirror the flattened tree as 256x256 text-slice PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--original", help="Flattened tree produced by 1_gather_images.py.")
    ap.add_argument("--mirror", help="Destination root for the slice PNGs.")
    ap.add_argument("--model", default="2.1_best_projection_band_2.keras",
                    help="Projection-band textline detector shipped in this workspace.")
    ap.add_argument("--side-frac", type=float, default=0.25,
                    help="Patch side as a fraction of min(H, W) — same default as "
                         "2.1_slice_text_square.py.")
    ap.add_argument("--min-prob-text", type=float, default=0.57,
                    help="Minimum text probability for a patch to count as a good slice.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Preprocessing threads. Default: all CPU cores.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-slice even when the mirror PNG already exists.")
    return ap.parse_args()


def finalize_patch(res, best_idx):
    """Same 'adjust patch side' math as 2.1_slice_text_square.py: rescale the
    selected square so that ~6 text lines (T=6) fit inside it, keep it centred
    on the original patch, then resize to 256x256."""
    import numpy as np
    import cv2

    best_band = res["bands"][best_idx].numpy().squeeze()
    if best_band.ndim != 1 or best_band.size != 256:
        M = None
    else:
        fft = np.fft.rfft(best_band)
        mag = np.abs(fft)
        sub_mag = mag[5:51]
        if np.all(sub_mag == 0):
            M = None
        else:
            M = 5 + int(np.argmax(sub_mag))

    y0_best, x0_best = res["coords"][best_idx]
    y0_target, x0_target, side_target = y0_best, x0_best, res["side"]

    if M is not None and M > 0:
        T = 6
        scale = T / float(M)
        new_side = int(round(res["side"] * scale))
        new_side = max(4, min(new_side, res["min_hw"]))

        y_center = y0_best + res["side"] // 2
        x_center = x0_best + res["side"] // 2

        side_final = min(new_side, res["H"], res["W"])
        y0_new = int(np.clip(y_center - side_final // 2, 0, res["H"] - side_final))
        x0_new = int(np.clip(x_center - side_final // 2, 0, res["W"] - side_final))
        y0_target, x0_target, side_target = y0_new, x0_new, side_final

    y1 = min(y0_target + side_target, res["H"])
    x1 = min(x0_target + side_target, res["W"])
    patch = res["bgr_full"][y0_target:y1, x0_target:x1]
    if patch.size == 0:
        patch = res["patch_bgrs"][best_idx]
    return cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()

    original_root = prompt_if_missing(args.original, "Flattened tree (step-1 output)")
    mirror_root = prompt_if_missing(args.mirror, "Mirror destination folder")
    if not original_root or not mirror_root or not os.path.isdir(original_root):
        print("[Error] A valid --original folder is required.")
        sys.exit(1)
    original_root, mirror_root = os.path.abspath(original_root), os.path.abspath(mirror_root)
    if original_root == mirror_root or mirror_root.startswith(original_root + os.sep):
        print("[Error] The mirror folder must live OUTSIDE the flattened tree.")
        sys.exit(1)

    os.makedirs(mirror_root, exist_ok=True)
    model_path = resolve_model_path(args.model)
    print(f"[Info] Loading model: {model_path}")
    model = load_keras_model(model_path)

    image_paths = gather_image_files(original_root)
    if not image_paths:
        print(f"[Error] No images found under {original_root}")
        sys.exit(1)

    todo = []
    for path in image_paths:
        rel = os.path.relpath(path, original_root)
        slice_path = os.path.join(mirror_root, os.path.splitext(rel)[0] + ".png")
        if args.overwrite or not os.path.exists(slice_path):
            todo.append((path, slice_path))

    print(f"[Info] {len(image_paths):,} flattened image(s) | {len(todo):,} to slice "
          f"({len(image_paths) - len(todo):,} already mirrored).")
    print(f"[Info] A slice counts as good when its text probability >= "
          f"{args.min_prob_text} (text lines may run horizontally OR vertically).")

    workers = cpu_workers(args.workers)
    chunk_size = workers * 4

    mirrored = skipped = failed = 0
    manifest_path = os.path.join(mirror_root, "mirror_manifest.csv")
    write_header = not os.path.exists(manifest_path)
    manifest = open(manifest_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(manifest)
    if write_header:
        writer.writerow(["original_path", "slice_path", "prob", "status", "reason"])

    from pathlib import Path

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(todo), chunk_size):
                chunk = todo[start:start + chunk_size]
                results = executor.map(
                    lambda pair: (pair[1], slice_lib.preprocess_single_image(
                        Path(pair[0]), Path(original_root), args.side_frac)),
                    chunk,
                )

                for slice_path, res in results:
                    if res is None:
                        failed += 1
                        writer.writerow([slice_path, "", "", "failed", "unreadable"])
                        continue

                    probs = model.predict(stack_bands(res["bands"]), verbose=0).ravel()

                    best_idx = int(probs.argmax())
                    best_prob = float(probs[best_idx])
                    best_is_horiz = res["is_horiz_flags"][best_idx]

                    reason = ""
                    if best_prob < args.min_prob_text:
                        reason = f"LOW_PROB_p{best_prob:.4f}"

                    if reason:
                        skipped += 1
                        writer.writerow([res["img_path"], slice_path,
                                         f"{best_prob:.4f}", "skipped", reason])
                        continue

                    patch = finalize_patch(res, best_idx)
                    os.makedirs(os.path.dirname(slice_path), exist_ok=True)
                    ok = slice_lib.save_png_bgr(Path(slice_path), patch)
                    if ok:
                        mirrored += 1
                        writer.writerow([res["img_path"], slice_path,
                                         f"{best_prob:.4f}", "mirrored", ""])
                    else:
                        failed += 1
                        writer.writerow([res["img_path"], slice_path,
                                         f"{best_prob:.4f}", "failed", "write_error"])

                done = mirrored + skipped + failed
                print(f"[Progress] {done:,}/{len(todo):,} processed | "
                      f"mirrored {mirrored:,} | skipped {skipped:,}")
    finally:
        manifest.flush()
        manifest.close()

    print("-" * 60)
    print(f"[Done] Mirrored {mirrored:,} image(s) into {mirror_root}")
    print(f"[Done] No good slice found for {skipped:,} image(s) -> NOT mirrored; "
          f"rotate those manually.")
    print(f"[Report] Full log: {manifest_path}")
    print("[Next] Choose ONE rotation method and run it against this mirror:")
    print("        2.2_method1_radonOnly.py            (corpus is 0/90 or 0/270 only)")
    print("        2.2_method2_radonPlusUpDownCNN.py   (radon for +-90, CNN for 180)")
    print("        2.2_method3_HVPlusUpDownCNN.py      (both decisions by CNN)")


def stack_bands(bands):
    import tensorflow as tf
    return tf.stack(bands, axis=0)


if __name__ == "__main__":
    main()
