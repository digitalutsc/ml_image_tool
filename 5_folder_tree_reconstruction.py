r"""
STEP 14 (last step) of the image-correction workflow.

What it does
------------
Reads the location log (mapping.csv) produced by 1_gather_images.py and
moves every flattened image back to its original location, reconstructing
the original folder tree.  Run this only AFTER rotation correction (2.2_*),
cropping (3_cropping.py) and all split/merge QA (4_* scripts) are finished.

Situations you should know about
--------------------------------
* Files that no longer exist (deleted during QA) are skipped and listed in
  a *_missing.csv report next to the log.
* Files that were CREATED after step 1 (e.g. the halves produced by the
  step-4 split scripts) are not in the log; they remain in the flattened
  folder and are listed in a *_leftover.csv report so you can place them
  manually.  Optionally point --leftovers-dest at a folder and they will be
  moved there in one go.

Usage examples
--------------
    python 5_folder_tree_reconstruction.py --csv "D:\flat\mapping.csv"

    # preview only, then move leftovers to a dedicated folder
    python 5_folder_tree_reconstruction.py --csv "D:\flat\mapping.csv" --dry-run
    python 5_folder_tree_reconstruction.py --csv "D:\flat\mapping.csv" --leftovers-dest "D:\new_pages"

Omit --csv and the script will prompt you for it.
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from workflow_common import IMAGE_EXTS, confirm, cpu_workers, prompt_if_missing


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 14: rebuild the original folder tree from the step-1 location log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", help="Path to mapping.csv created by 1_gather_images.py.")
    ap.add_argument("--mode", choices=["move", "copy"], default="move",
                    help="'move' empties the flattened folders; 'copy' keeps them.")
    ap.add_argument("--leftovers-dest", default=None,
                    help="Optional folder to collect files that are NOT in the log "
                         "(e.g. halves created by the step-4 split scripts).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report what would happen; touch nothing.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers. Default: all CPU cores.")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = prompt_if_missing(args.csv, "Path to the mapping.csv location log")
    if not csv_path or not os.path.isfile(csv_path):
        print(f"[Error] Location log not found: {csv_path}")
        sys.exit(1)
    csv_path = os.path.abspath(csv_path)
    flattened_root = os.path.dirname(csv_path)

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append((row["new_path"], row["old_path"]))
    if not rows:
        print("[Error] The location log contains no rows.")
        sys.exit(1)

    restore = shutil.move if args.mode == "move" else shutil.copy2
    workers = cpu_workers(args.workers)

    known_new = {os.path.abspath(new) for new, _ in rows}
    pending = [(new, old) for new, old in rows if os.path.exists(new)]
    missing = [old for new, old in rows if not os.path.exists(new)]

    print(f"[Plan] {len(rows):,} log entries | {len(pending):,} restorable | "
          f"{len(missing):,} missing | mode={args.mode} | dry-run={args.dry_run}")

    # Files sitting in the flattened tree that the log does not know about.
    leftovers = []
    for dirpath, dirnames, filenames in os.walk(flattened_root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.abspath(os.path.join(dirpath, name))
            if name.lower() == os.path.basename(csv_path).lower():
                continue
            if full not in known_new and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                leftovers.append(full)

    print(f"[Plan] {len(leftovers):,} unmapped file(s) found (split halves etc.).")
    if args.dry_run:
        print("[Dry-run] Nothing was moved. Re-run without --dry-run to execute.")
        return
    if not confirm(f"Restore {len(pending):,} image(s) into the original tree?",
                   assume_yes=args.yes):
        print("[Abort] Nothing was touched.")
        sys.exit(0)

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for new, old in pending:
            os.makedirs(os.path.dirname(old), exist_ok=True)
            futures[pool.submit(restore, new, old)] = (new, old)
        for fut in as_completed(futures):
            new, old = futures[fut]
            try:
                fut.result()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[Warn] Could not restore {new} -> {old}: {exc}")
            if ok % 500 == 0 and ok:
                print(f"[Progress] {ok:,}/{len(pending):,} restored ...")

    if args.leftovers_dest and leftovers:
        os.makedirs(args.leftovers_dest, exist_ok=True)
        for src in leftovers:
            try:
                shutil.move(src, os.path.join(args.leftovers_dest, os.path.basename(src)))
            except Exception as exc:  # noqa: BLE001
                print(f"[Warn] Could not move leftover {src}: {exc}")

    # Reports live next to the log.
    base = os.path.splitext(csv_path)[0]
    if missing:
        with open(base + "_missing.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["original_path"])
            w.writerows([[m] for m in missing])
    if leftovers:
        with open(base + "_leftover.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["flattened_path"])
            w.writerows([[l] for l in leftovers])

    print("-" * 60)
    print(f"[Done] Restored {ok:,} image(s) ({failed} failure(s), {len(missing):,} missing).")
    if missing:
        print(f"[Report] Missing originals listed in {base}_missing.csv")
    if leftovers:
        if args.leftovers_dest:
            print(f"[Done] Leftovers moved to {args.leftovers_dest}")
        else:
            print(f"[Report] Unmapped files left in {flattened_root}; see "
                  f"{base}_leftover.csv (use --leftovers-dest to collect them).")


if __name__ == "__main__":
    main()
