# When a bad pair of image split is found, we would like to merge them back to a single
# image so that they can be split using other methods later on. This code does just
# that. 
# 
# To use it, just drag the bad images to process_folder and the script will
# find and merge a pair automatically. 
# 
# Before running this script, make sure that a pair of
# images have the same filename except that the right version of it has a '_r' suffix.
# Also, make sure the images to be processed are named in a way so that their direct 
# parent folder names are added as a prefix followed by '@@!!!!!!@@' before their actual 
# name. '@@!!!!!!@@' is used as a separator between their parent folder and name.

import os
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

root_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira"  # Root folder that is one level higher than the images' parent folders
process_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira\\merge"  # Process folder. You will drag the images here during execution

# Function to merge two images into one
def merge_images(left_image_path, right_image_path, root_folder, merged_name):
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

    # Extract folder name from the name of the image
    folder_name, _ = merged_name.split('@@!!!!!!@@', 1)

    # Locate the correct folder in the root directory
    destination_folder = None
    for root, dirs, _ in os.walk(root_folder):
        if folder_name in dirs:
            destination_folder = os.path.join(root, folder_name)
            break

    if destination_folder is None:
        print(f"Destination folder for {folder_name} not found!")
        return

    # Save the merged image
    output_path = os.path.join(destination_folder, merged_name)
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
                right_image_path = left_image_path.replace('.j', '_r.j')
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
    main(root_folder, process_folder)