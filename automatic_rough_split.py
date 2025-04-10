# This code splits images into two with equal areas. Thus, the result is not very intelligent.
# However, there are cases where the exact center is the desired splitting line. If that is
# the case, just run this script. It is much faster than the other splitting algorithm.
#
# To use it, just drag the bad images to process_folder and the script will
# find and merge a pair automatically. 
# 
# Before running this script, make sure the images to be processed are named in a way so 
# that their direct parent folder names are added as a prefix followed by '@@!!!!!!@@' 
# before their actual name. '@@!!!!!!@@' is used as a separator between their parent folder 
# and name.

import os
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

root_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira"  # Root folder that is one level higher than the images' parent folders
process_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira\\rough_split"  # Process folder. You will drag the images here during execution

# Function to split an image into two and save them to the destination folder
def split_in_two(filepath, name, root_folder):
    # Load the image and preprocess it
    image = cv2.imread(filepath)
    if image is None:
        print('img is none')
        return
    # Get image dimensions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, width = gray.shape
    # Split the image into two equal parts
    left_page = image[:, :width // 2]
    right_page = image[:, width // 2:]

    # Extract folder name from filename
    nm, _ = os.path.splitext(name)
    folder_nm, _ = nm.split('@@!!!!!!@@', 1)

    # Find the folder in the root folder
    for root, dirs, _ in os.walk(root_folder):
        if folder_nm in dirs:
            dest_folder = os.path.join(root, folder_nm)
            break
    else:
        print(f"Destination folder for {name} not found!")
        return

    # Construct paths for split images
    left_name = nm + '.jpg'
    right_name = nm + '_r' + '.jpg'
    left_path = os.path.join(dest_folder, left_name)
    right_path = os.path.join(dest_folder, right_name)

    # Save the resulting images
    cv2.imwrite(left_path, left_page, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(right_path, right_page, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # Delete the original image in the process folder
    os.remove(filepath)
    print(f"Deleted processed file: {filepath}")

# Event handler for the watchdog
class ImageHandler(FileSystemEventHandler):
    def __init__(self, root_folder, process_folder):
        self.root_folder = root_folder
        self.process_folder = process_folder

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.JPG', 
                                                                       '.jpeg', '.JPEG', 
                                                                       '.png', '.PNG', 
                                                                       '.TIF', '.tif', 
                                                                       '.tiff', '.TIFF')):
            filename = os.path.basename(event.src_path)
            print(f"Processing new image: {filename}")

            # Check if the file is accessible
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    # Attempt to open the file
                    with open(event.src_path, 'rb'):
                        pass
                    break  # File is accessible
                except (IOError, PermissionError):
                    # File is still being written or locked; wait and retry
                    time.sleep(0.5)
            else:
                print(f"File {filename} is not accessible after several attempts.")
                return

            split_in_two(event.src_path, filename, self.root_folder, 1)


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