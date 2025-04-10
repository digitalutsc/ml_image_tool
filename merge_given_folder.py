# Just a non-automatic version of the python script "automatic_merge.py"

import os
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

saving_folder = "C:\\D\\DSU\\Dragomans\\two-page-noncropped\\down-merged"  # Replace with the path to the root folder
process_folder = "C:\\D\\DSU\\Dragomans\\two-page-noncropped\\merge-folder"  # Replace with the path to the process folder

# Function to merge two images into one
def merge_images(left_image_path, right_image_path, saving_folder, merged_name):
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    if left_image is None or right_image is None:
        print(f"Error loading images: {left_image_path}, {right_image_path}")
        return

    # Ensure both images have the same height for horizontal merging
    height = max(left_image.shape[0], right_image.shape[0])
    left_image = cv2.copyMakeBorder(left_image, 0, height - left_image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    right_image = cv2.copyMakeBorder(right_image, 0, height - right_image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Merge the images horizontally
    merged_image = np.hstack((left_image, right_image))

    # Save the merged image
    output_path = os.path.join(saving_folder, merged_name)
    cv2.imwrite(output_path, merged_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # Remove the original files
    os.remove(left_image_path)
    os.remove(right_image_path)

class ImageHandler(FileSystemEventHandler):
    def __init__(self, root_folder, process_folder):
        self.root_folder = root_folder
        self.process_folder = process_folder
        self.image_cache = {}  # Cache to store unpaired images and their arrival times
        self.processed_pairs = set()  # Track the successfully processed pairs

    def on_created(self, event):
        # Make sure the newly added item is an image
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.TIFF', '.TIF')):
            filename = os.path.basename(event.src_path)

            # Determine if the file is part of a pair
            base_name, ext = os.path.splitext(filename)
            if base_name.endswith('_r'):
                pair_name = base_name[:-2] + ext  # Name of the corresponding left image
                right_image_path = event.src_path
            else:
                pair_name = base_name + '_r' + ext  # Name of the corresponding right image
                right_image_path = None

            # Add the new file to the cache with a timestamp
            self.image_cache[filename] = (event.src_path, time.time())

            # Check if the corresponding pair exists in the cache
            if pair_name in self.image_cache:
                # Pair found, merge the images
                right_image_path = right_image_path or event.src_path
                left_image_path = right_image_path.replace('_r', '')
                right_image_path = left_image_path.replace(ext, '_r' + ext)
                clean_name = base_name.replace('_r', '') if '_r' in base_name else base_name
                merge_images(left_image_path, right_image_path,
                             self.root_folder, clean_name + ext)

                # Mark the pair as processed
                self.processed_pairs.add((left_image_path, right_image_path))
                self.image_cache.pop(filename, None)  # Remove the current file from the cache
                
# Main function
def main(root_folder, process_folder):
    event_handler = ImageHandler(root_folder, process_folder)
    observer = Observer()
    observer.schedule(event_handler, process_folder, recursive=False)
    observer.start()
    print(f"Monitoring folder: {process_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

# Run the program
if __name__ == "__main__":
    main(saving_folder, process_folder)