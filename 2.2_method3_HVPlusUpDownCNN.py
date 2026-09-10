r"""
STEP 11 — Rotation correction, OPTION 3 of 3: two CNNs
(2.2_Horizontal_Vertical.keras for +-90, then 2.2_Up_Down.keras for 180).

How it works
------------
Runs on the MIRROR tree (step 8) with the same mirror-lookup logic as
2.2_method1_radonOnly.py (step 9): whenever a rotation is applied to a
mirror slice, the matching ORIGINAL image in the flattened tree receives
the exact same rotation.

PHASE 1 - 2.2_Horizontal_Vertical.keras (batched, 2.1_CNN_Test.py
preprocessing: RGB [0,1], 256x256, optional --crop-ratio):
    score <  0.5 -> HORIZONTAL (rotation +-90)
                    -> rotate slice + original 90 deg COUNTER-CLOCKWISE
                       (a +90 page becomes 0, a 270 page becomes 180)
    score >= 0.5 -> VERTICAL (0/180) -> untouched for now.

PHASE 2 - 2.2_Up_Down.keras on the (now vertical) slices:
    score >= 0.5 -> UPSIDE-DOWN -> rotate slice + original by 180.
    score <  0.5 -> UPRIGHT     -> untouched.

Class legend of the sigmoid outputs (naming convention "A_vs_B" = {0: A,
1: B}; the up/down meaning "0 = Upright, 1 = Upside" comes from the
original training recipe and 2.1_CNN_Test.py):
    Horizontal_Vertical: 0 = HORIZONTAL (+-90) | 1 = VERTICAL (0/180)
    Up_Down:             0 = UPRIGHT          | 1 = UPSIDE-DOWN
If a checkpoint is ever re-trained with swapped labels, pass --invert-hv /
--invert-ud, or calibrate first with 2.1_CNN_Test.py on slices whose
orientation you know.

Originals that never got a mirror slice are NOT touched and are listed in
a *_no_slice.csv report.

Usage
-----
    python 2.2_method3_HVPlusUpDownCNN.py --mirror "D:\flat_mirror" --original "D:\flat"

Omit an argument and the script will prompt you for it.  --dry-run reports
without rotating (phase-2 scores are then computed on UNrotated slices).
"""

import argparse
import csv
import os
import sys
import time

from workflow_common import (
    cpu_workers,
    find_original_for_slice,
    gather_image_files,
    iter_batch_scores,
    load_binary_cnn,
    prompt_if_missing,
    rotate_pair,
    unmirrored_originals,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 11 / rotation option 3: Horizontal/Vertical CNN then "
                    "Up/Down CNN, both via the mirror tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mirror", help="Mirror tree built by 2.2_0_mirror_text_slices.py.")
    ap.add_argument("--original", help="Flattened original tree from step 1.")
    ap.add_argument("--hv-model", default="2.2_Horizontal_Vertical.keras",
                    help="Horizontal/Vertical binary CNN shipped in this workspace.")
    ap.add_argument("--ud-model", default="2.2_Up_Down.keras",
                    help="Up/Down binary CNN shipped in this workspace.")
    ap.add_argument("--hv-threshold", type=float, default=0.5,
                    help="Scores below this count as horizontal (+-90).")
    ap.add_argument("--ud-threshold", type=float, default=0.5,
                    help="Scores at or above this count as upside-down.")
    ap.add_argument("--invert-hv", action="store_true",
                    help="Swap horizontal/vertical meaning of the HV checkpoint.")
    ap.add_argument("--invert-ud", action="store_true",
                    help="Swap up/down meaning of the UD checkpoint.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="CNN prediction batch size.")
    ap.add_argument("--crop-ratio", type=float, default=None,
                    help="Only if the CNNs were trained with a long-rectangle crop.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Preprocessing threads. Default: all CPU cores.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; rotate nothing.")
    return ap.parse_args()


def run_cnn_phase(name, model, slices, threshold, invert, positive_is_below,
                  mirror_root, original_root, rotate_name, dry_run, workers,
                  batch_size, crop_ratio, rows, rows_col0, rows_col1):
    """Generic CNN phase.  positive_is_below=True means the ACTION class is
    score < threshold (horizontal for the HV model), otherwise the action
    class is score >= threshold (upside-down for the UD model)."""
    print(f"[{name}] Scoring {len(slices):,} slice(s) with batch size {batch_size} ...")
    start = time.time()
    action_count = keep_count = 0
    for i, (path, score) in enumerate(
            iter_batch_scores(model, slices, crop_ratio=crop_ratio,
                              batch_size=batch_size, workers=workers), start=1):
        if positive_is_below:
            is_action = (score < threshold) != invert
        else:
            is_action = (score >= threshold) != invert

        row = rows.setdefault(path, [path, "", "", "", ""])
        row[rows_col0] = f"{score:.4f}"
        if is_action:
            action_count += 1
            if dry_run:
                row[rows_col1] = f"would_rotate_{rotate_name}"
            else:
                original = find_original_for_slice(path, mirror_root, original_root)
                s_ok, o_ok = rotate_pair(path, original, rotate_name)
                note = f"rotated_{rotate_name}"
                if not original:
                    note += " (original not found)"
                elif not o_ok:
                    note += " (original rotation FAILED)"
                if not s_ok:
                    note += " (slice rotation FAILED)"
                row[rows_col1] = note
        else:
            keep_count += 1
            row[rows_col1] = "untouched"
        if i % 200 == 0 or i == len(slices):
            eta = (time.time() - start) / i * (len(slices) - i)
            print(f"[{name}] {i:,}/{len(slices):,} | action {action_count:,} | "
                  f"keep {keep_count:,} | ETA {eta:,.0f}s")
    return action_count, keep_count


def main():
    args = parse_args()

    mirror_root = prompt_if_missing(args.mirror, "Mirror folder (step-8 output)")
    original_root = prompt_if_missing(args.original, "Flattened original folder (step-1 output)")
    if not mirror_root or not original_root or not os.path.isdir(mirror_root) \
            or not os.path.isdir(original_root):
        print("[Error] Valid --mirror and --original folders are required.")
        sys.exit(1)

    hv_pos = "HORIZONTAL (+-90)" if not args.invert_hv else "VERTICAL (0/180)"
    print("[Legend] HV CNN: score <  {:.2f} -> {} (rotate 90ccw)".format(
        args.hv_threshold, hv_pos))
    print("[Legend] UD CNN: score >= {:.2f} -> {} (rotate 180)".format(
        args.ud_threshold, "UPSIDE-DOWN" if not args.invert_ud else "UPRIGHT"))

    slices = [p for p in gather_image_files(mirror_root) if p.lower().endswith(".png")]
    if not slices:
        print(f"[Error] No mirror slices (*.png) under {mirror_root}. "
              f"Run 2.2_0_mirror_text_slices.py first.")
        sys.exit(1)
    no_mirror = unmirrored_originals(mirror_root, original_root)
    print(f"[Info] {len(slices):,} mirror slice(s) | {len(no_mirror):,} original(s) "
          f"without a slice (untouched).")

    workers = cpu_workers(args.workers)
    rows = {}  # slice -> [slice, hv_score, phase1_action, ud_score, phase2_action]

    print(f"[Model] Loading {args.hv_model} ...")
    hv_model = load_binary_cnn(args.hv_model)
    hv_action, hv_keep = run_cnn_phase(
        "Phase 1 HV", hv_model, slices, args.hv_threshold, args.invert_hv,
        positive_is_below=True, mirror_root=mirror_root,
        original_root=original_root, rotate_name="90ccw",
        dry_run=args.dry_run, workers=workers, batch_size=args.batch_size,
        crop_ratio=args.crop_ratio, rows=rows, rows_col0=1, rows_col1=2)
    del hv_model  # free memory before loading the second CNN

    print(f"[Model] Loading {args.ud_model} ...")
    ud_model = load_binary_cnn(args.ud_model)
    ud_action, ud_keep = run_cnn_phase(
        "Phase 2 UD", ud_model, slices, args.ud_threshold, args.invert_ud,
        positive_is_below=False, mirror_root=mirror_root,
        original_root=original_root, rotate_name="180",
        dry_run=args.dry_run, workers=workers, batch_size=args.batch_size,
        crop_ratio=args.crop_ratio, rows=rows, rows_col0=3, rows_col1=4)

    report_path = os.path.join(mirror_root, "_method3_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "hv_score", "phase1_action",
                         "ud_score", "phase2_action"])
        for path in slices:
            writer.writerow(rows.get(path, [path, "?", "?", "?", "?"]))
    if no_mirror:
        with open(os.path.join(mirror_root, "_method3_no_slice.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["original_without_slice"])
            w.writerows([[p] for p in no_mirror])

    print("-" * 70)
    print(f"[Done] Phase 1 (HV): horizontal {hv_action:,} "
          f"{'would be ' if args.dry_run else ''}rotated 90ccw | "
          f"vertical {hv_keep:,}")
    print(f"[Done] Phase 2 (UD): upside-down {ud_action:,} "
          f"{'would be ' if args.dry_run else ''}rotated 180 | upright {ud_keep:,}")
    if args.dry_run:
        print("[Note] Dry-run: phase-2 scores were computed WITHOUT applying "
              "phase-1 rotations.")
    print(f"[Report] {report_path}")
    if no_mirror:
        print(f"[Manual] {len(no_mirror):,} original(s) had no mirror slice -> "
              f"rotate manually; list: _method3_no_slice.csv")
    print("[Next] 3_cropping.py on the corrected flattened tree, then the "
          "step-4 split/merge QA scripts.")


if __name__ == "__main__":
    main()
