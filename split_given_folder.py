# This is just a non-automatic version of the script "split_given_folder.py"
# For more information, go and read the top notes in that one.

import os
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from PIL import Image

process_folder = 'C:\\D\\DSU\\Kanagaratnam\\fixing\\184'

# Function to split an image into two and save them to the destination folder
def split_in_two(filepath, filename, n, x, middle_kth):
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
    img_PIL = Image.open(filepath)
    if image is None:
        print('img is none')
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 110, 170, apertureSize=3)

    # Get image dimensions
    height, width = edges.shape

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
            cutoff_range = max(min_boundary_pixels + 4, int(min_boundary_pixels * x))
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
    left_page = img_PIL.crop((0, 0, int(x_val), height))
    right_page = img_PIL.crop((int(x_val), 0, width, height))

    # Construct paths for split images
    nm, ext = os.path.splitext(filename)
    left_name = nm + '_l' + ext
    right_name = nm + '_r' + ext
    left_path = os.path.join(process_folder, left_name)
    right_path = os.path.join(process_folder, right_name)

    # Save the resulting images
    # cv2.imwrite(left_path, left_page, [cv2.IMWRITE_TIFF_COMPRESSION, 8])
    # cv2.imwrite(right_path, right_page, [cv2.IMWRITE_TIFF_COMPRESSION, 8])
    left_page.save(left_path, compression="tiff_jpeg")
    right_page.save(right_path, compression="tiff_jpeg")
    img_PIL.close()
    os.remove(os.path.join(process_folder, filename))

i = 0
for filename in os.listdir(process_folder):
    imgpath = os.path.join(process_folder, filename)
    start_time = time.time()
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')):
        i += 1
        print(f'{filename}')
        split_in_two(imgpath, filename, 1, 1.15, 5)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{execution_time} seconds")