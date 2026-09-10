r"""
STEP 10 — Rotation correction, OPTION 2 of 3: Radon (for +-90) + Up/Down CNN
(for 180).

How it works
------------
Runs on the MIRROR tree (step 8) with the same mirror-lookup logic as
2.2_method1_radonOnly.py (step 9): whenever a rotation is applied to a
mirror slice, the matching ORIGINAL image in the flattened tree receives
the exact same rotation.

PHASE 1 - Radon transform (processes, one per CPU core):
    HORIZONTAL slice (text lines vertical, rotation +-90)
        -> rotate slice + original 90 deg COUNTER-CLOCKWISE, which brings
           the page to either 0 or 180.
    VERTICAL slice -> untouched for now.
    (The turn is always 90ccw: a +90 page becomes 0, a 270 page becomes
    180 — the CNN in phase 2 catches those.)

PHASE 2 - 2.2_Up_Down.keras (batched CNN, 2.1_CNN_Test.py preprocessing):
    score >= 0.5 -> UPSIDE-DOWN -> rotate slice + original by 180.
    score <  0.5 -> UPRIGHT     -> untouched.

Class legend of the sigmoid output (threshold --ud-threshold):
    2.2_Up_Down naming convention "A_vs_B" = {0: A, 1: B}; the original
    recipe documents "0 = Upright, 1 = Upside".  If a re-trained checkpoint
    ever swaps these, pass --invert-ud (or calibrate first with
    2.1_CNN_Test.py on slices of known orientation).

Originals that never got a mirror slice are NOT touched and are listed in
a *_no_slice.csv report.

Usage
-----
    python 2.2_method2_radonPlusUpDownCNN.py --mirror "D:\flat_mirror" --original "D:\flat"

Omit an argument and the script will prompt you for it.  --dry-run reports
without rotating (phase-2 scores are then computed on UNrotated slices).
"""

import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Phase 1 reuses the detector of method 1 verbatim.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "method1_radon",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.2_method1_radonOnly.py"),
)
method1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(method1)

warnings.filterwarnings("ignore")


def phase1_worker(task):
    """Radon pass: rotate every HORIZONTAL slice+original by 90ccw."""
    (slice_path, mirror_root, original_root, angle_interval, line_space,
     scale_down, method, dry_run) = task
    try:
        is_vertical = method1.orientation_detect(slice_path, angle_interval,
                                                 line_space, scale_down, method)
    except Exception as exc:  # noqa: BLE001
        return slice_path, "error", f"detection_failed: {exc}"
    if is_vertical:
        return slice_path, "vertical", "untouched"
    original = find_original_for_slice(slice_path, mirror_root, original_root)
    if dry_run:
        return slice_path, "horizontal", "would_rotate_90ccw"
    slice_ok, original_ok = rotate_pair(slice_path, original, "90ccw")
    note = "rotated_90ccw"
    if not original:
        note += " (original not found)"
    elif not original_ok:
        note += " (original rotation FAILED)"
    if not slice_ok:
        note += " (slice rotation FAILED)"
    return slice_path, "horizontal", note


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 10 / rotation option 2: Radon (+-90) then Up/Down CNN (180).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mirror", help="Mirror tree built by 2.2_0_mirror_text_slices.py.")
    ap.add_argument("--original", help="Flattened original tree from step 1.")
    ap.add_argument("--ud-model", default="2.2_Up_Down.keras",
                    help="Up/Down binary CNN shipped in this workspace.")
    ap.add_argument("--ud-threshold", type=float, default=0.5,
                    help="Score at or above this counts as upside-down.")
    ap.add_argument("--invert-ud", action="store_true",
                    help="Swap up/down meaning if the checkpoint was re-trained "
                         "with flipped labels.")
    ap.add_argument("--angle-interval", type=int, default=15,
                    help="Radon angle step in degrees.")
    ap.add_argument("--line-space", type=int, default=65,
                    help="Expected line spacing in pixels for the radon detector.")
    ap.add_argument("--scale-down", type=int, default=1,
                    help="Down-scale factor before the Radon transform.")
    ap.add_argument("--method", choices=["peak", "max", "var"], default="max",
                    help="Row-scoring method inside the radon detector.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="CNN prediction batch size.")
    ap.add_argument("--crop-ratio", type=float, default=None,
                    help="Only if the CNN was trained with a long-rectangle crop.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel processes/threads. Default: all CPU cores.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; rotate nothing.")
    return ap.parse_args()


def main():
    args = parse_args()

    mirror_root = prompt_if_missing(args.mirror, "Mirror folder (step-8 output)")
    original_root = prompt_if_missing(args.original, "Flattened original folder (step-1 output)")
    if not mirror_root or not original_root or not os.path.isdir(mirror_root) \
            or not os.path.isdir(original_root):
        print("[Error] Valid --mirror and --original folders are required.")
        sys.exit(1)

    print("[Legend] Up/Down CNN: score >= {:.2f} -> {} (rotate 180) | "
          "score < {:.2f} -> {} (keep)".format(
              args.ud_threshold,
              "UPSIDE-DOWN" if not args.invert_ud else "UPRIGHT",
              args.ud_threshold,
              "UPRIGHT" if not args.invert_ud else "UPSIDE-DOWN"))

    slices = [p for p in gather_image_files(mirror_root) if p.lower().endswith(".png")]
    if not slices:
        print(f"[Error] No mirror slices (*.png) under {mirror_root}. "
              f"Run 2.2_0_mirror_text_slices.py first.")
        sys.exit(1)
    no_mirror = unmirrored_originals(mirror_root, original_root)
    print(f"[Info] {len(slices):,} mirror slice(s) | {len(no_mirror):,} original(s) "
          f"without a slice (untouched).")

    workers = cpu_workers(args.workers)
    report_path = os.path.join(mirror_root, "_method2_report.csv")
    rows = {}

    # ---------------- PHASE 1: radon (+-90 -> 0/180 via 90ccw) ----------------
    print(f"[Phase 1] Radon detection on {workers} process(es) ...")
    start = time.time()
    tasks = [(s, mirror_root, original_root, args.angle_interval, args.line_space,
              args.scale_down, args.method, args.dry_run) for s in slices]
    counts1 = {"vertical": 0, "horizontal": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(phase1_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            slice_path, decision, action = fut.result()
            counts1[decision] = counts1.get(decision, 0) + 1
            rows[slice_path] = [slice_path, decision, action, "", ""]
            if i % 100 == 0 or i == len(tasks):
                eta = (time.time() - start) / i * (len(tasks) - i)
                print(f"[Phase 1] {i:,}/{len(tasks):,} | vertical "
                      f"{counts1['vertical']:,} | horizontal {counts1['horizontal']:,} "
                      f"| errors {counts1['error']:,} | ETA {eta:,.0f}s")

    # ---------------- PHASE 2: up/down CNN (180 -> 0) ----------------
    print(f"[Phase 2] Up/Down CNN ({args.ud_model}), batch size {args.batch_size} ...")
    print("[Info] Loading model ...")
    ud_model = load_binary_cnn(args.ud_model)
    ud_turned = ud_kept = 0
    for i, (path, score) in enumerate(
            iter_batch_scores(ud_model, slices, crop_ratio=args.crop_ratio,
                              batch_size=args.batch_size, workers=workers), start=1):
        is_down = (score >= args.ud_threshold) != args.invert_ud
        row = rows.setdefault(path, [path, "n/a", "n/a", "", ""])
        row[3] = f"{score:.4f}"
        if is_down:
            ud_turned += 1
            if args.dry_run:
                row[4] = "would_rotate_180"
            else:
                original = find_original_for_slice(path, mirror_root, original_root)
                s_ok, o_ok = rotate_pair(path, original, "180")
                row[4] = "rotated_180" + ("" if o_ok else " (original FAILED)")
                if not s_ok:
                    row[4] += " (slice FAILED)"
        else:
            ud_kept += 1
            row[4] = "upright_untouched"
        if i % 200 == 0 or i == len(slices):
            print(f"[Phase 2] {i:,}/{len(slices):,} | upside-down {ud_turned:,} | "
                  f"upright {ud_kept:,}")

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "radon_decision", "phase1_action",
                         "ud_score", "phase2_action"])
        for path in slices:
            writer.writerow(rows.get(path, [path, "?", "?", "?", "?"]))
    if no_mirror:
        with open(os.path.join(mirror_root, "_method2_no_slice.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["original_without_slice"])
            w.writerows([[p] for p in no_mirror])

    print("-" * 70)
    print(f"[Done] Phase 1 (radon) : horizontal {counts1['horizontal']:,} "
          f"{'would be ' if args.dry_run else ''}rotated 90ccw | "
          f"vertical {counts1['vertical']:,} | errors {counts1['error']:,}")
    print(f"[Done] Phase 2 (CNN)   : upside-down {ud_turned:,} "
          f"{'would be ' if args.dry_run else ''}rotated 180 | upright {ud_kept:,}")
    if args.dry_run:
        print("[Note] Dry-run: phase-2 scores were computed WITHOUT applying "
              "phase-1 rotations.")
    print(f"[Report] {report_path}")
    if no_mirror:
        print(f"[Manual] {len(no_mirror):,} original(s) had no mirror slice -> "
              f"rotate manually; list: _method2_no_slice.csv")
    print("[Next] 3_cropping.py on the corrected flattened tree, then the "
          "step-4 split/merge QA scripts.")


if __name__ == "__main__":
    main()
