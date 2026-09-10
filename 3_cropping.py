r"""
STEP 12 of the workflow — background cropping of the rotation-corrected
pages.

What it does
------------
Crops the dark background around the (brighter) page in the centre of each
image, adding a small safety margin so no page edge is ever clipped.  The
detection pipeline per image is unchanged from the original script:

    gray -> Otsu binarization -> morphological CLOSE (kills text contours)
    -> morphological OPEN (kills specks) -> Canny edges -> largest contours
    -> page borders + safety margin -> crop.

Run it on the ROTATION-CORRECTED flattened tree (after whichever
2.2_method* script you chose).  By default it crops IN PLACE; pass --save
to write cropped copies elsewhere instead.

Failure handling: images that cannot be processed (unreadable, or no
contour detected at all) are MOVED to the error folder so they can be
handled manually; they are also listed in a report CSV.

Usage
-----
    python 3_cropping.py --folder "D:\flat"

    # crop into a separate tree instead of in place
    python 3_cropping.py --folder "D:\flat" --save "D:\flat_cropped"

Omit an argument and the script will prompt you for it.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from workflow_common import (
    cpu_workers,
    gather_image_files,
    prompt_if_missing,
    read_image,
    write_image,
)


def cut_borders(img_path, save_path, error_folder, safe_denominator):
    """One image: detect the page and write the cropped result to save_path.
    On failure the file is moved into error_folder.  Returns (path, status).
    """
    try:
        image = read_image(img_path)
        if image is None:
            raise ValueError("Image is corrupted or cannot be loaded.")

        if image.ndim == 3:
            h, w, _ = np.shape(image)
        else:
            h, w = np.shape(image)

        # There needs to be some padding around the image after cropping.
        # The safe_denominator guarantees a margin of (h+w)/safe_denominator
        # between the strict cut and the actual padded cut.
        crop_safety_margin = (h + w) // safe_denominator

        best_x_left = 0
        best_x_right = w - 1
        padding = 31
        top, bottom, left, right = padding, padding, padding, padding
        # Add black padding around the image so borders touch the frame
        black_padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        black_padded_image = cv2.cvtColor(black_padded_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(black_padded_image, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        height, width = np.shape(binary_image)
        opening_width = (height + width) // 20

        # Close the shape to eliminate any contours from texts
        kernel0 = np.ones((30, 30), np.uint8)
        dilated_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel0)

        # Open the shape to eliminate small bits connected with the page
        kernel1 = np.ones((opening_width, opening_width), np.uint8)
        dilated_image = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel1)

        # Canny edges and contours
        edges1 = cv2.Canny(dilated_image, 50, 120)
        contours1, _ = cv2.findContours(edges1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours1:
            raise ValueError("No contour detected - cannot find the page.")
        biggest_contour = max(contours1, key=cv2.contourArea)
        biggest_contour_area = cv2.contourArea(biggest_contour)
        large_contours = [c for c in contours1
                          if cv2.contourArea(c) > biggest_contour_area // 3]

        # Draw the contours, then read the page borders off the drawn mask
        dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dilated_image, large_contours, -1, (0, 255, 0), 2)
        green_mask = np.all(dilated_image == [0, 255, 0], axis=-1)

        columns_with_contours = np.any(green_mask, axis=0)
        border1 = 1 + np.argmax(columns_with_contours) - padding
        border2 = width - np.argmax(np.flip(columns_with_contours)) - padding

        rows_with_contours = np.any(green_mask, axis=1)
        border3 = 1 + np.argmax(rows_with_contours) - padding
        border4 = height - np.argmax(np.flip(rows_with_contours)) - padding

        if border1 < best_x_left:
            border1 = best_x_left
        if border2 > best_x_right:
            border2 = best_x_right

        cropped = image[max(0, border3 - crop_safety_margin):
                        min(border4 + 1 + crop_safety_margin, h),
                        max(0, border1 - crop_safety_margin):
                        min(border2 + 1 + crop_safety_margin, w)]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not write_image(save_path, cropped, jpeg_quality=100):
            raise ValueError("Failed to write the cropped image.")
        return img_path, "cropped"
    except Exception as exc:  # noqa: BLE001 - original behaviour: move & report
        try:
            os.makedirs(error_folder, exist_ok=True)
            shutil.move(img_path, os.path.join(error_folder, os.path.basename(img_path)))
        except Exception as move_exc:  # noqa: BLE001
            exc = f"{exc} | additionally, moving to error folder failed: {move_exc}"
        return img_path, f"error: {exc}"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 12: crop the background around each page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--folder",
                    help="Rotation-corrected flattened tree (recursive).")
    ap.add_argument("--save", default=None,
                    help="Where to write cropped images. Default: crop IN PLACE. "
                         "When set, the sub-folder structure is preserved.")
    ap.add_argument("--error-folder", default=None,
                    help="Where problem images are moved. Default: <folder>/_cropping_errors.")
    ap.add_argument("--safe-denominator", type=int, default=95,
                    help="Bigger value = tighter crop; smaller = more margin "
                         "((h+w)/value of safety margin pixels).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel processes. Default: all CPU cores.")
    return ap.parse_args()


def main():
    args = parse_args()

    folder = prompt_if_missing(args.folder, "Folder of rotation-corrected images")
    if not folder or not os.path.isdir(folder):
        print(f"[Error] Folder not found: {folder}")
        sys.exit(1)
    folder = os.path.abspath(folder)

    save_root = os.path.abspath(args.save) if args.save else folder
    error_folder = os.path.abspath(args.error_folder) if args.error_folder \
        else os.path.join(folder, "_cropping_errors")
    workers = cpu_workers(args.workers)

    images = gather_image_files(folder)
    # Never re-process the error folder itself if it lives inside the tree.
    images = [p for p in images if not p.startswith(error_folder + os.sep)]
    if not images:
        print(f"[Error] No images found under {folder}")
        sys.exit(1)

    print(f"[Info] {len(images):,} image(s) | output: "
          f"{'IN PLACE' if save_root == folder else save_root} | "
          f"error folder: {error_folder} | workers: {workers}")

    tasks = []
    for path in images:
        if save_root == folder:
            save_path = path
        else:
            rel = os.path.relpath(path, folder)
            save_path = os.path.join(save_root, rel)
        tasks.append((path, save_path, error_folder, args.safe_denominator))

    cropped = errors = 0
    report_path = os.path.join(error_folder, "_cropping_report.csv")
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(cut_borders, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            path, status = fut.result()
            if status == "cropped":
                cropped += 1
            else:
                errors += 1
                os.makedirs(error_folder, exist_ok=True)
                with open(report_path, "a", newline="", encoding="utf-8-sig") as f:
                    csv.writer(f).writerow([path, status])
            if i % 100 == 0 or i == len(tasks):
                eta = (time.time() - start) / i * (len(tasks) - i)
                print(f"[Progress] {i:,}/{len(tasks):,} | cropped {cropped:,} | "
                      f"errors {errors:,} | ETA {eta:,.0f}s")

    print("-" * 60)
    print(f"[Done] Cropped {cropped:,} image(s); {errors:,} moved to the error "
          f"folder ({error_folder}).")
    if errors:
        print(f"[Report] {report_path}")
    print("[Next] Step 13: the split/merge QA scripts with prefix 4_ "
          "(see README), then 5_folder_tree_reconstruction.py.")


if __name__ == "__main__":
    main()
