# This code splits images into two using an intelligent method detailed in the function
# description.
#
# To use it, drag the bad images to process_folder and the script will
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

root_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira"  # Replace with the path to your root folder
process_folder = "C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira\\split"  # Replace with the path to your process folder

# Function to split an image into two and save them to the destination folder
def split_in_two(filepath, name, root_folder, n, x, middle_kth):
    # This function scans every nth vertical line of the image and determine the best
    # split line. The parameter x determines how lenient the code it at determineing
    # candidates for good splitting lines. The bigger the value is, the more vertical
    # lines will be considered. You want to change this value because different sets
    # of images have different needs. middle_kth determines the reasonable middle kth 
    # region to find the croppling line.

    # in detail, the code first finds the vertical line that passes through the least
    # amount of edges, and store that number of edges as min_boundary_pixels. The x
    # parameter then determines the vertical lines that passes through x times
    # min_boundary_pixels or lower should also be counted as the candidates. Finally,
    # the function finds the average place of these candidates and make the cut.

    # Load the image and preprocess it
    image = cv2.imread(filepath)
    if image is None:
        print('img is none')
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)

    # Get image dimensions
    _, width = edges.shape

    # Define the middle kth region for crease detection
    start_x = width // middle_kth*2 * (middle_kth - 1)
    end_x = width // middle_kth*2 * (middle_kth + 1)

    min_boundary_pixels = float('inf')  # Initialize to infinity
    best_x = []  
    best_x.append((0, min_boundary_pixels)) # Initialize the best x

    # Find the best line to split the image separated by every n pixels
    for x in range((end_x - start_x) // n):
        boundary_pixels = np.sum(edges[:, start_x + n * x]) // 255
        if boundary_pixels <= min_boundary_pixels:
            min_boundary_pixels = boundary_pixels
            cutoff_range = max(min_boundary_pixels + 4, int(min_boundary_pixels * 2.618))
            for pair in best_x[:]:
                if pair[1] > cutoff_range:
                    best_x.remove(pair)
            best_x.append((start_x + n * x, boundary_pixels))

    # Find the best x value by getting the average position of the x value 
    # With the lowest edge passes
    print(best_x)
    x_vals = [i[0] for i in best_x]
    x_val = sum(x_vals) // len(x_vals)
    print(x_vals)

    # Split the image into two parts
    left_page = image[:, :int(x_val)]
    right_page = image[:, int(x_val):]

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

            split_in_two(event.src_path, filename, self.root_folder, 1, 2.6, 5)


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