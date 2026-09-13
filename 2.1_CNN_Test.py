r"""
VALIDATION / CALIBRATION helper for the two binary orientation CNNs
(2.2_Horizontal_Vertical_retrained.keras and 2.2_Up_Down.keras).

What it does
------------
Feeds a folder of images (typically 256x256 text slices from
2.1_slice_text_square.py) through one binary CNN, batch-predicts with the
exact preprocessing recipe the models were trained with, and moves every
image classified as the POSITIVE class into a dump folder so you can
eyeball the mistakes.

Class legend (threshold 0.5) — printed at start-up
--------------------------------------------------
  2.2_Up_Down.keras             score < 0.5 -> UPRIGHT      (0)
                                score >= 0.5 -> UPSIDE-DOWN  (1)  -> dumped
  2.2_Horizontal_Vertical_retrained.keras
                                score < 0.5 -> VERTICAL page, 0/180 deg (0)
                                score >= 0.5 -> HORIZONTAL page, +-90 deg (1) -> dumped
                                (verified empirically: 0/180 pages score ~0.0,
                                 +-90 pages score ~1.0)

Both mappings come from the original training recipes (ocr_hori_vs_vert.py
pins the HV labels with [0,1,0,1] for the 0/90/180/270 CCW rotations; the
up/down recipe documents "0 = Upright, 1 = Upside").

HOW TO CALIBRATE before a big rotation run
------------------------------------------
1. Slice a handful of pages whose orientation you KNOW with
   2.1_slice_text_square.py.
2. Run this script on that folder with --model 2.2_Up_Down.keras.
   If "Upright Images" is far below ~100%, either the model is weak on your
   data or the classes are swapped — try --invert and re-run.
3. Repeat with --model 2.2_Horizontal_Vertical_retrained.keras on slices you
   know are vertical pages (upright text): the accuracy line then reads
   "vertical page" images.

Usage
-----
    python 2.1_CNN_Test.py --model 2.2_Up_Down.keras --src "D:\known_upright_slices" --dump "D:\dump"

Omit an argument and the script will prompt you for it.  --dry-run only
prints statistics and touches nothing.
"""

import argparse
import os
import shutil
import sys

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from workflow_common import (
    cpu_workers,
    load_binary_cnn,
    preprocess_for_binary_cnn,
    prompt_if_missing,
    read_image,
    write_image,
)

MODEL_CLASSES = {
    "2.2_up_down.keras": ("upright", "upside-down"),
    "2.2_horizontal_vertical_retrained.keras": ("vertical page (0/180 deg)",
                                                "horizontal page (+-90 deg)"),
    "2.2_horizontal_vertical.keras": ("vertical page (0/180 deg)",
                                      "horizontal page (+-90 deg)"),
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Validate/calibrate one binary orientation CNN on a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="2.2_Up_Down.keras",
                    help="Model to test: 2.2_Up_Down.keras or "
                         "2.2_Horizontal_Vertical_retrained.keras "
                         "(resolved inside this workspace by default).")
    ap.add_argument("--src", help="Folder of images to score (recursed).")
    ap.add_argument("--dump", help="Folder that collects images classified as the "
                                    "positive class. Prompted for when omitted.")
    ap.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                    help="Model input size (H W).")
    ap.add_argument("--crop-ratio", type=float, default=None,
                    help="Set only if the model was trained with a long-rectangle crop "
                         "(e.g. 2.0); otherwise leave as None.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Positive-class cutoff.")
    ap.add_argument("--invert", action="store_true",
                    help="Swap the class meaning if a re-trained model was saved with "
                         "flipped labels.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None,
                    help="Preprocessing threads. Default: all CPU cores.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Statistics only; do not move anything.")
    return ap.parse_args()


def main():
    args = parse_args()

    src = prompt_if_missing(args.src, "Folder of images to test")
    if not src or not os.path.isdir(src):
        print(f"[Error] Source folder not found: {src}")
        sys.exit(1)
    dump = None if args.dry_run else prompt_if_missing(args.dump, "Dump folder for positive-class images")

    key = os.path.basename(args.model).lower()
    neg_label, pos_label = MODEL_CLASSES.get(key, ("class 0", "class 1"))
    if args.invert:
        neg_label, pos_label = pos_label, neg_label

    print(f"[Info] Loading model: {args.model}")
    model = load_binary_cnn(args.model, img_size=tuple(args.img_size))
    print("[Legend] score <  {:.2f} -> {}   (kept)".format(args.threshold, neg_label))
    print("[Legend] score >= {:.2f} -> {}   (dumped{})".format(
        args.threshold, pos_label, ", inverted" if args.invert else ""))

    from workflow_common import gather_image_files
    image_files = gather_image_files(src)
    if not image_files:
        print(f"[Error] No images found under {src}")
        sys.exit(1)
    print(f"[Info] {len(image_files):,} images to score.")

    if dump:
        os.makedirs(dump, exist_ok=True)

    workers = cpu_workers(args.workers)
    neg_count = pos_count = failed = 0

    def load_one(path):
        return preprocess_for_binary_cnn(read_image(path),
                                         img_size=tuple(args.img_size),
                                         crop_ratio=args.crop_ratio)

    for start in range(0, len(image_files), args.batch_size):
        batch_paths = image_files[start:start + args.batch_size]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tensors = list(pool.map(load_one, batch_paths))

        keep, probs = [], []
        for p, t in zip(batch_paths, tensors):
            if t is None:
                failed += 1
                print(f"[Warn] Could not read {p}")
                continue
            keep.append(p)
            probs.append(t)
        if not keep:
            continue

        batch = np.stack(probs, axis=0)
        scores = model.predict(batch, verbose=0).ravel()

        for score, p in zip(scores, keep):
            if score < args.threshold:
                neg_count += 1
            else:
                pos_count += 1
                if dump:
                    target = os.path.join(dump, os.path.basename(p))
                    if not os.path.exists(target):
                        shutil.move(p, target)
                    else:  # name clash: keep the original, note it
                        print(f"[Warn] Dump target exists, not moved: {target}")

        done = neg_count + pos_count + failed
        if (start // args.batch_size) % 20 == 0:
            print(f"[Progress] {done:,}/{len(image_files):,} scored ...")

    total = neg_count + pos_count
    pct = (neg_count / total * 100.0) if total else 0.0
    print("-" * 60)
    print(f"Model        : {args.model}" + (" (inverted)" if args.invert else ""))
    print(f"Folder       : {src}")
    print(f"Scored       : {total:,}   (failed to read: {failed})")
    print(f"{neg_label:<24}: {neg_count:,}  ({pct:.2f}%)")
    print(f"{pos_label:<24}: {pos_count:,}")
    if dump and not args.dry_run:
        print(f"Positive-class images moved to: {dump}")
    print("-" * 60)
    print("[Tip] For an all-known-good folder this ratio should be ~100% "
          "for the kept class. If it is ~0%, the labels are swapped -> "
          "re-run with --invert.")


if __name__ == "__main__":
    main()
