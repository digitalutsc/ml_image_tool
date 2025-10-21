# This is the first step of image-processing. Normally, images are stored in a tree of
# directories that are hard to keep track of during the processing (yes, my method 
# still requires some level of supervision) or QA. 
# 
# Hence, gathering images into ordered
# folders of large size, in this case 2^14, is necessary. The following code does 
# this exact task and ensures no collision occurs by renaming images using 
# enumeration. 
# 
# The mapping from their original absolute paths and newly enumerated 
# names are stored in a csv file so that images can go back to where they came from 
# after processing and QA tasks.

import os
import shutil
import csv

root = '' # Absolute path to the root
dest = '' # Absolute path to the destination

def merge_images(root_dir, merged_dir, csv_filename="mapping.csv"):
    MAX_IMAGES_PER_FOLDER = 2**14  # Maximum images per subdirectory (16384)
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    # Create the merged directory if it doesn't exist
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    mapping = []  # To store tuples of (old_absolute_path, new_absolute_path)
    image_index = 0  # Global image counter

    # Walk through the root directory recursively
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in allowed_extensions:
                original_path = os.path.join(dirpath, filename)
                
                # Determine which subdirectory to place the image in
                folder_index = image_index // MAX_IMAGES_PER_FOLDER
                subfolder_name = str(folder_index)
                subfolder_path = os.path.join(merged_dir, subfolder_name)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                
                # Create new file name using a simple enumeration (e.g., "000001.jpg")
                new_filename = f"{folder_index}@@!!!!!!@@{image_index:06d}{ext}"
                new_path = os.path.join(subfolder_path, new_filename)
                
                # Move the image to the new location
                shutil.move(original_path, new_path)
                
                # Record the mapping (absolute paths for clarity)
                mapping.append([os.path.abspath(original_path), os.path.abspath(new_path)])
                image_index += 1

    # Save the mapping to a CSV file for future folderpath reconstruction
    csv_path = os.path.join(merged_dir, csv_filename)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["old_path", "new_path"])
        writer.writerows(mapping)

    print(f"Processed {image_index} images. Mapping saved to {csv_path}")

if __name__ == "__main__":
    merge_images(root, dest)
