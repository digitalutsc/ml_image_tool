r"""
OPTIONAL STEP 5 of the workflow (training-data preparation).

Selects a homogeneous sub-sample of an image set by taking every n-th image
(or every n-th PAIR of images).  Use it to pull a training/fine-tuning
sample out of the flattened folders produced by 1_gather_images.py; the
sample feeds 2.1_slice_text_square.py (optional step 6) and afterwards the
2.2_Horizontal_Vertical / 2.2_Up_Down fine-tuning (2.1_CNN_finetune.py).

Pair convention
---------------
A pair of images belongs together when they share the same file name
except that the RIGHT half carries an "_r" suffix:

    0@@!!!!!!@@000123.jpg   and   0@@!!!!!!@@000123_r.jpg

Pairs are selected/dropped as a unit so that halves never get separated.
(The "_r" halves only exist after the step-4 split scripts, so a fresh
flattened tree from step 1 normally has no pairs at all.)

Usage
-----
    python 2.1_sample_selection.py --src "D:\flat" --dst "D:\sample" --every-nth 20

Omit an argument and the script will prompt you for it.
"""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from workflow_common import cpu_workers, prompt_if_missing


def build_plan(src_root, every_nth):
    """Deterministically walk the tree and decide which files get selected.
    Returns (selection_units, skipped_note) where each unit is a tuple of
    one file (single) or two files (pair)."""
    units = []
    counter = 0
    for root, dirnames, files in os.walk(src_root):
        dirnames.sort()
        base_names = {os.path.splitext(f)[0]: f for f in files}
        for file in sorted(files):
            nm, ext = os.path.splitext(file)

            if "_r" in nm and nm.replace("_r", "") in base_names:
                continue  # the left member of this pair handles the selection

            partner = None
            if nm + "_r" in base_names:
                partner = base_names[nm + "_r"]

            counter += 1
            if counter % every_nth != 0:
                continue

            if partner:
                units.append((os.path.join(root, file), os.path.join(root, partner)))
            else:
                units.append((os.path.join(root, file),))
    return units


def main():
    ap = argparse.ArgumentParser(
        description="Optional step 5: take every n-th image (or pair) as a "
                    "training/QA sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", help="Source root (e.g. the flattened step-1 output).")
    ap.add_argument("--dst", help="Destination folder for the sampled images.")
    ap.add_argument("--every-nth", type=int, default=20,
                    help="Select one image (or pair) out of every N.")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="'copy' keeps the source intact (recommended: the "
                         "flattened tree is still needed for the real workflow).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers. Default: all CPU cores.")
    args = ap.parse_args()

    src_root = prompt_if_missing(args.src, "Source folder to sample from")
    dst_folder = prompt_if_missing(args.dst, "Destination folder for the sample")
    if not src_root or not dst_folder or not os.path.isdir(src_root):
        print("[Error] A valid --src and a --dst are required.")
        return

    if args.every_nth < 1:
        print("[Error] --every-nth must be >= 1.")
        return

    os.makedirs(dst_folder, exist_ok=True)
    units = build_plan(src_root, args.every_nth)
    n_files = sum(len(u) for u in units)
    print(f"[Plan] Selecting every {args.every_nth}-th unit -> {len(units):,} unit(s), "
          f"{n_files:,} file(s), mode={args.mode}.")

    transfer = shutil.copy2 if args.mode == "copy" else shutil.move

    def transfer_unit(unit):
        targets = []
        for src in unit:
            dst = os.path.join(dst_folder, os.path.basename(src))
            transfer(src, dst)
            targets.append(dst)
        return targets

    done = failed = 0
    workers = cpu_workers(args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transfer_unit, unit) for unit in units]
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[Warn] Transfer failed: {exc}")

    print("-" * 60)
    print(f"[Done] {done:,} unit(s) sampled into {dst_folder} ({failed} failure(s)).")
    print("[Next] Optional step 6: slice this sample with "
          "2.1_slice_text_square.py --root <sample folder> --dst <slice folder>.")


if __name__ == "__main__":
    main()
