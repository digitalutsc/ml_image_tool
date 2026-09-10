r"""
STEP 1 of the image-correction workflow (run this first!).
Also enforces STEP 0: the safety warning about backing up your originals.

What it does
------------
Walks the SOURCE folder tree recursively, flattens every image it finds into
the DESTINATION folder as numbered sub-folders ("0", "1", "2", ...) of at
most --max-per-folder files each (default: half of what Windows/NTFS allows
per folder, i.e. (2**32 - 1) // 2), and renames every image to a globally
unique enumerated name:

    <destination>/<folder_index>@@!!!!!!@@<image_index:06d><original ext>
    e.g.  my_dest/0@@!!!!!!@@000123.tif      (file #123 lives in folder 0)

The "@@!!!!!!@@" separator lets the step-4 split/merge scripts recover the
parent folder prefix, so do not remove it.

While gathering, every (old absolute path, new absolute path) pair is
appended to <destination>/mapping.csv.  5_folder_tree_reconstruction.py
(step 14) reads exactly that CSV to put every image back where it came from
after the QA work is done.

Usage examples (run from the repository folder)
----------------------------------------------
    python 1_gather_images.py --root "D:\scans" --dest "D:\flat"

    # keep originals in place (copy instead of move) and use smaller folders
    python 1_gather_images.py --root "D:\scans" --dest "D:\flat" --mode copy --max-per-folder 16384

Omit any required argument and the script will prompt you for it.
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from workflow_common import (
    DEFAULT_MAX_FILES_PER_FOLDER,
    IMAGE_EXTS,
    NTFS_MAX_FILES_PER_FOLDER,
    confirm,
    cpu_workers,
    prompt_if_missing,
)

STEP0_BANNER = r"""
==========================================================================
  STEP 0 - READ THIS BEFORE CONTINUING
==========================================================================
  This workflow MOVES your images around (rotation correction, cropping,
  splitting).  Mistakes and QA edits are part of the process, so:

  >>> MAKE A COMPLETE BACKUP COPY OF YOUR ORIGINAL FOLDER TREE NOW,  <<<
  >>> before running any step of this workflow.                       <<<

  If you have not made a backup yet, press Ctrl+C and do it first.
==========================================================================
"""


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 1: gather a folder tree into flattened, enumerated folders "
                    "and record the location log (mapping.csv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", help="Absolute path of the ORIGINAL folder tree.")
    ap.add_argument("--dest", help="Absolute path of the DESTINATION (flattened) folder.")
    ap.add_argument("--max-per-folder", type=int, default=DEFAULT_MAX_FILES_PER_FOLDER,
                    help=f"Maximum files per numbered sub-folder. Default = NTFS max "
                         f"({NTFS_MAX_FILES_PER_FOLDER}) divided by 2.")
    ap.add_argument("--mode", choices=["move", "copy"], default="move",
                    help="'move' relocates the files (original tree becomes empty); "
                         "'copy' keeps the originals untouched but is slower and "
                         "doubles the disk usage.")
    ap.add_argument("--csv-name", default="mapping.csv",
                    help="Name of the location log written into --dest.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel transfer workers. Default: all CPU cores.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive STEP 0 confirmation (for scripted runs).")
    return ap.parse_args()


def main():
    args = parse_args()

    print(STEP0_BANNER)
    print(f"Source      : {args.root or '(will prompt)'}")
    print(f"Destination : {args.dest or '(will prompt)'}")
    print(f"Mode        : {args.mode}   Max per folder: {args.max_per_folder:,}")
    if not confirm("Have you made a backup and want to continue?", assume_yes=args.yes):
        print("[Abort] Nothing was touched. Make a backup first.")
        sys.exit(0)

    root = prompt_if_missing(args.root, "Root folder of the original image tree")
    dest = prompt_if_missing(args.dest, "Destination folder for the flattened copies")
    if not root or not dest:
        print("[Error] Both root and destination are required.")
        sys.exit(1)

    root = os.path.abspath(root)
    dest = os.path.abspath(dest)
    max_per_folder = max(1, args.max_per_folder)
    workers = cpu_workers(args.workers)

    if not os.path.isdir(root):
        print(f"[Error] Root folder does not exist: {root}")
        sys.exit(1)
    if os.path.commonpath([root, dest]) in (root,) and dest != root:
        print(f"[Error] Destination must NOT be inside the root folder ({root}).")
        sys.exit(1)

    os.makedirs(dest, exist_ok=True)
    csv_path = os.path.join(dest, args.csv_name)
    if os.path.exists(csv_path):
        print(f"[Error] {csv_path} already exists. A previous run recorded this "
              f"destination; rename or delete it before re-gathering, otherwise "
              f"step 14 would restore files to the wrong places.")
        sys.exit(1)

    # ---- deterministic enumeration (sorted walk => reproducible mapping) ----
    print("[Scan] Collecting image files ...")
    plan = []  # (original_path, new_path)
    index = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            folder_index = index // max_per_folder
            new_name = f"{folder_index}@@!!!!!!@@{index:06d}{os.path.splitext(name)[1].lower()}"
            new_path = os.path.join(dest, str(folder_index), new_name)
            plan.append((os.path.join(dirpath, name), new_path))
            index += 1

    if not plan:
        print(f"[Error] No images found under {root} "
              f"(extensions searched: {', '.join(sorted(IMAGE_EXTS))}).")
        sys.exit(1)

    subfolders = sorted({os.path.dirname(p) for _, p in plan})
    for sub in subfolders:
        os.makedirs(sub, exist_ok=True)

    print(f"[Plan] {len(plan):,} images into {len(subfolders)} sub-folder(s) "
          f"(<= {max_per_folder:,} files each) using {workers} worker(s), mode={args.mode}.")
    if not confirm("Start the transfer?", assume_yes=args.yes):
        print("[Abort] Nothing was touched.")
        sys.exit(0)

    transfer = shutil.move if args.mode == "move" else shutil.copy2

    def do_transfer(pair):
        original_path, new_path = pair
        transfer(original_path, new_path)
        return original_path, new_path

    done = failed = 0
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["old_path", "new_path"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(do_transfer, pair) for pair in plan]
            for fut in as_completed(futures):
                try:
                    old, new = fut.result()
                    writer.writerow([old, new])
                    done += 1
                except Exception as exc:  # noqa: BLE001 - keep going, report at end
                    failed += 1
                    print(f"[Warn] Transfer failed: {exc}")
                if done % 500 == 0 and done:
                    csvfile.flush()
                    print(f"[Progress] {done:,}/{len(plan):,} transferred ...")
        csvfile.flush()

    print("-" * 60)
    print(f"[Done] Transferred {done:,} image(s) ({failed} failure(s)) into: {dest}")
    print(f"[Done] Location log written to  : {csv_path}")
    print("[Next] 2.2_0_mirror_text_slices.py  -> build the text-slice mirror of "
          "this flattened tree (required before any rotation method).")
    print("[Last] 5_folder_tree_reconstruction.py will use this CSV to restore "
          "the original tree at the very end.")


if __name__ == "__main__":
    main()
