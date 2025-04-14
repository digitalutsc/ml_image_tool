# Sometimes the split line for the image is so close to the text that it became hard
# for reviewers to determine whether the cut accidentally left some texts out of the
# Scope.
#
# This code adds the left strip of the right image to the left image, and vice versa.
#
# Before using it, make sure all the image pairs in question are in the input_folder.
#
# Before running this script, make sure that a pair of
# images have the same filename except that the right version of it has a '_r' suffix.

import os
import cv2
import numpy as np

input_folder = "C:\\D\\DSU\\Dragomans\\two-page-noncropped\\all"
output_folder = "C:\\D\\DSU\\Dragomans\\two-page-noncropped\\all - margin_added"

def find_image_pairs(folder):
    """Find image pairs (left and right) based on filenames."""
    images = {os.path.splitext(f)[0]: f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    pairs = []

    for name, filename in images.items():
        if name.endswith("_r"):
            base_name = name[:-2]  # Remove "_r"
            if base_name in images:
                pairs.append((images[base_name], filename))
    
    return pairs

def process_and_save_images(input_folder, output_folder):
    """Process image pairs and swap the left strip of width pixels."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pairs = find_image_pairs(input_folder)

    for left_image_name, right_image_name in pairs:
        left_path = os.path.join(input_folder, left_image_name)
        right_path = os.path.join(input_folder, right_image_name)

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f"Skipping {left_image_name} and {right_image_name}: Unable to load.")
            continue

        if left_img.shape[0] != right_img.shape[0]:
            print(f"Skipping {left_image_name} and {right_image_name}: Dimension mismatch.")
            continue

        h, w, _ = left_img.shape
        width = (h + w) // 60

        # Extract left strips
        left_strip = left_img[:, -width:]
        right_strip = right_img[:, :width]

        # Swap and create new images
        cpy_left = left_img.copy()
        cpy_right = right_img.copy()

        new_left = np.hstack((cpy_left, right_strip))
        new_right = np.hstack((left_strip, cpy_right))

        # Save the modified images
        left_output_path = os.path.join(output_folder, left_image_name)
        right_output_path = os.path.join(output_folder, right_image_name)

        cv2.imwrite(left_output_path, new_left)
        cv2.imwrite(right_output_path, new_right)

        print(f"Processed and saved: {left_output_path}, {right_output_path}")

process_and_save_images(input_folder, output_folder)
