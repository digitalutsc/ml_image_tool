# This script crops images by removing the background around the page in the center.
# The assumption is that the page itself is brighter than the background, which
# should be reasonable. 

import cv2
import numpy as np
import os
import time
import shutil

# C:\\D\\DSU\\Dragomans\\upright-corrected
folder = 'C:\\D\\DSU\\Dragomans\\CNN-labelled-Dragomans - Copy\\CNN-labelled-Dragomans\\upright'
save = folder #'C:\\D\\DSU\\Gunda_Gunde\\GG_new_media\\Asir Matira\\AM-010 save'
error_folder = 'C:\\D\\DSU\\Dragomans'

# Load the image and pre-process it
def cut_borders(img_path, filename, safe_denominator):
    # There needs to be some padding around the image after cropping. The 
    # safe_denominator variable does just that. It makes sure that there
    # will be a distance of (img_height+img_width)/safe_denominator 
    # between the strict cut and the actual padded cut

    # Load the image
    try:
        # Load the image
        image = cv2.imread(img_path)

        # Check if the image was loaded correctly
        if image is None:
            raise ValueError("Image is corrupted or cannot be loaded.")

        # Check the dimensions of the image
        if image.ndim == 3:
            h, w, _ = np.shape(image)
        else:
            h, w = np.shape(image)
        crop_safety_margin = (h + w) // safe_denominator
        # Find whether or not the image has part of the other page
        # If so, cut it off
        best_x_left = 0
        best_x_right = w - 1
        padding = 31
        top, bottom, left, right = padding, padding, padding, padding
        # Add black padding (pixel value 0) around the image
        black_padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        black_padded_image = cv2.cvtColor(black_padded_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(black_padded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        height, width = np.shape(binary_image)
        opening_width = (height + width) // 20
        
        # Close the shape to eliminate any contours from texts
        kernel0 = np.ones((30, 30), np.uint8)
        dilated_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel0)
        edges2 = cv2.Canny(dilated_image, 50, 120)
        cv2.imwrite(f'{filename[4]}.jpg', edges2)
        # Identify such page and cut it
        # if np.sum(edges2[:, padding]) >= 2*255 ^ np.sum(edges2[:, - padding - 1]) >= 2*255:
        #     print('incomplete page found')
        #     if np.sum(edges2[:, 31]) > 2*255:
        #         min_boundary_pixels = float('inf')
        #         for x in range(0, w // 6):
        #             # Count the number of boundary pixels this vertical line passes through
        #             boundary_pixels = np.sum(edges2[:, padding + x])
        #             # Update the best line if the current one passes through fewer boundary pixels
        #             if boundary_pixels < min_boundary_pixels:
        #                 min_boundary_pixels = boundary_pixels
        #                 best_x_left = x
        #     else:
        #         min_boundary_pixels = float('inf')
        #         for x in range(5 * w // 6, w - 1):
        #             # Count the number of boundary pixels this vertical line passes through
        #             boundary_pixels = np.sum(edges2[:, padding + x])
        #             # Update the best line if the current one passes through fewer boundary pixels
        #             if boundary_pixels < min_boundary_pixels:
        #                 min_boundary_pixels = boundary_pixels
        #                 best_x_right = x
        #     print(f'------------------------------{max(best_x_left, w - 1 - best_x_right)}')
        # Open the shape to eliminate any small bits connected with the main page
        kernel1 = np.ones((opening_width, opening_width), np.uint8)
        dilated_image = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel1)

        # Canny edge detection
        edges1 = cv2.Canny(dilated_image, 50, 120)
        # Find contours
        contours1, _ = cv2.findContours(edges1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        biggest_contour = max(contours1, key=cv2.contourArea)
        biggest_contour_area = cv2.contourArea(biggest_contour)
        large_contours = [c for c in contours1 if cv2.contourArea(c) > biggest_contour_area // 3]
        
        # if 0.75 * cv2.contourArea(biggest_contour) < cv2.contourArea(biggest_contour2) < 0.93 * cv2.contourArea(biggest_contour):
        #     green_mask = np.all(dilated_image == [0, 255, 0], axis = -1)

        #     # Left and right boundary
        #     columns_with_contours = np.any(green_mask, axis = 0)
        #     border1 = 1 + np.argmax(columns_with_contours) - padding
        #     border2 = width - np.argmax(np.flip(columns_with_contours)) - padding

        #     # Top and bottom boundary
        #     rows_with_contours = np.any(green_mask, axis = 1)
        #     border3 = 1 + np.argmax(rows_with_contours) - padding
        #     border4 = height - np.argmax(np.flip(rows_with_contours)) - padding

        # Draw contours on the original image
        # Convert binary image back to BGR for contour drawing
        dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dilated_image, large_contours, -1, (0, 255, 0), 2)

        green_mask = np.all(dilated_image == [0, 255, 0], axis = -1)

        # Left and right boundary
        columns_with_contours = np.any(green_mask, axis = 0)
        border1 = 1 + np.argmax(columns_with_contours) - padding
        border2 = width - np.argmax(np.flip(columns_with_contours)) - padding

        # Top and bottom boundary
        rows_with_contours = np.any(green_mask, axis = 1)
        border3 = 1 + np.argmax(rows_with_contours) - padding
        border4 = height - np.argmax(np.flip(rows_with_contours)) - padding

        # border1 = 0
        # border2 = width - 1 - 2 * padding
        # border3 = 0
        # border4 = height - 1 - 2 * padding

        # for x in range(padding // 5 + 1, width):
        #     # Count the number of boundary pixels this vertical line passes through
        #     boundary_pixels = dilated_image[:, x * 5]

        #     # Update the best line if the current one passes through fewer boundary pixels
        #     if any(np.array_equal(pixel, [0, 255, 0]) for pixel in boundary_pixels):
        #         if x == padding // 5 + 1:
        #             border1 = 0
        #             break
        #         else:
        #             border1 = (x - 1) * 5 - padding
        #             print(f"cropped, left, {border1} pixels cut")
        #             break

        # for x in range(padding // 5 + 1, width):
        #     # Count the number of boundary pixels this vertical line passes through
        #     boundary_pixels = dilated_image[:, width - 1 - x * 5]

        #     # Update the best line if the current one passes through fewer boundary pixels
        #     if any(np.array_equal(pixel, [0, 255, 0]) for pixel in boundary_pixels):
        #         if x == padding // 5 + 1:
        #             border2 = width - 1 - 2 * padding
        #             break
        #         else:
        #             border2 = width - 1 - padding - (x - 1) * 5
        #             print(f"cropped, right, {width - 1 - border2} pixels cut")
        #             break

        # for x in range(padding // 5 + 1, height):
        #     # Count the number of boundary pixels this vertical line passes through
        #     boundary_pixels = dilated_image[x * 5, :]

        #     # Update the best line if the current one passes through fewer boundary pixels
        #     if any(np.array_equal(pixel, [0, 255, 0]) for pixel in boundary_pixels):
        #         if x == padding // 5 + 1:
        #             border3 = 0
        #             break
        #         else:
        #             border3 = (x - 1) * 5 - padding
        #             print(f"cropped, up, {border3} pixels cut")
        #             break

        # for x in range(padding // 5 + 1, height):
        #     # Count the number of boundary pixels this vertical line passes through
        #     boundary_pixels = dilated_image[height - 1 - x * 5, :]

        #     # Update the best line if the current one passes through fewer boundary pixels
        #     if any(np.array_equal(pixel, [0, 255, 0]) for pixel in boundary_pixels):
        #         if x == padding // 5 + 1:
        #             border4 = height - 1 - 2 * padding
        #             break
        #         else:
        #             border4 = height - 1 - padding - (x - 1) * 5
        #             print(f"cropped, down, {height - 1 - border4} pixels cut")
        #             break

        if border1 < best_x_left:
            border1 = best_x_left
        if border2 > best_x_right:
            border2 = best_x_right
        image = image[max(0, border3 - crop_safety_margin) : min(border4 + 1 + crop_safety_margin, h),
                       max(0, border1 - crop_safety_margin) : min(border2 + 1 + crop_safety_margin, w)]
        
        # draw red cropping borders on the dialated image and write it. This step is used to debug indexing
        # cv2.line(dilated_image, (border1 + padding, 0), (border1 + padding, height - 1), (0, 0, 255), thickness = 2)
        # cv2.line(dilated_image, (border2 + 1 + padding, 0), (border2 + 1 + padding, height - 1), (0, 0, 255), thickness = 2)
        # cv2.line(dilated_image, (0, border3 + padding), (width - 1, border3 + padding), (0, 0, 255), thickness = 2)
        # cv2.line(dilated_image, (0, border4 + 1 + padding), (width - 1, border4 + 1 + padding), (0, 0, 255), thickness = 2)

        # new_name2 = filename.split('.')[0] + '8' + '.' + filename.split('.')[1]
        # save_path2 = os.path.join(save, new_name2)
        # cv2.imwrite(save_path2, dilated_image)

        new_name1 = filename
        save_path1 = os.path.join(save, new_name1)
        cv2.imwrite(save_path1, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    except Exception as e:
        print(f"Error processing image {img_path}: {e}") # Normally an error is reported when no contour is detected
        # Move the corrupted image to the error folder
        error_path = os.path.join(error_folder, os.path.basename(img_path))
        shutil.move(img_path, error_path)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Moved corrupted image to {error_path}")

i = 0
for filename in os.listdir(folder):
    imgpath = os.path.join(folder, filename)
    start_time = time.time()
    if filename.endswith(('.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.gif', '.tif', 'TIF')):
        i += 1
        print(f'{filename}')
        cut_borders(imgpath, filename, 95)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{execution_time} seconds")