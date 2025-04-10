# This code determines whether the image is horizontal (+-90 degrees of rotation)
# or vertical (0 or 180 degrees of rotation) by using radon transform.
#
# This section takes in images from src, a folder with vertical images only, and sees how well the parameters
# of the function orientation_detect performs. After finding the proper parameters, please change the code
# accordingly to rotate the images based on the output of orientation_detect


from skimage.transform import radon
import numpy as np
import cv2
import warnings
import shutil
import time
import os
warnings.filterwarnings("ignore")
try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    print('import parabolic error')



src = 'C:\\D\\DSU\\Dragomans\\types of pages we have'
mistake_folder = 'C:\\D\\DSU\\Dragomans\\types of pages we have\\m'


def orientation_detect(imgpath, angle_interval, line_space, scale_down, method):
    image = cv2.imread(imgpath)
    height, width, _ = np.shape(image)

    # Crop image to avoid extremely obvious borders which may result in a high magnitude low frequency wave after FFT
    image_cropped = image[height // 7 : 6 * height // 7, width // 7 : 6 * width // 7]
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 50, 120, apertureSize=3)
    # Demean; make the brightness extend above and below zero
    gray = gray - np.mean(gray)  # Remove the horizontal and vertical middle 10th of the gray image so that

    # It is sliced into four parts to remove central crease which may result in a high magnitude low frequency wave after FFT
    gray_height, gray_width = np.shape(gray)
    small_height = 9 * gray_height // 20
    small_width = 9 * gray_width // 20
    image1 = gray[0 : small_height, 0 : small_width]
    image2 = gray[0 : small_height, -small_width :]
    image3 = gray[-small_height :, 0 : small_width]
    image4 = gray[-small_height :, -small_width :]

    # Create a new canvas with double the height and width to join the 4 slices
    new_height = small_height * 2
    new_width = small_width * 2
    no_mid_gray = np.zeros((new_height, new_width))  # Initialize a black canvas

    # Place images in the new canvas
    no_mid_gray[0:small_height, 0:small_width] = image1  # Top-left
    no_mid_gray[0:small_height, small_width:] = image2  # Top-right
    no_mid_gray[small_height:, 0:small_width] = image3  # Bottom-left
    no_mid_gray[small_height:, small_width:] = image4  # Bottom-right
    resized_image = cv2.resize(no_mid_gray, (new_width // scale_down, new_height // scale_down))

    # do discrete Radon transform every 'angle_interval' degree
    angles = np.arange(0, 180, angle_interval)
    angles = angles.tolist()
    sinogram = radon(resized_image, angles)

    # find the desired row by using FFT and find the row with the most dominant
    # frequency among a certain frequency interval
    interval = 2 * line_space
    special_rows = []
    y, _ = np.shape(sinogram)
    Tsino = sinogram.transpose()
    # Loop through each row of the image
    for row in enumerate(Tsino):
        # Subtract the mean to normalize the row
        normalized_row = row[1] - np.mean(row[1])
        
        # Compute the Fourier Transform of the row
        spectrum = np.abs(np.fft.fft(normalized_row)) / y
        # Focus on the desired frequencies
        spectrum = spectrum[y // interval : y // 4]
        
        # print(f'col {i} dominant is {peak / avg_magnitude} stronger than average')
        # find the best row by checking whether or not the peak frequency's magnitude
        # is significantly higher than the average magnitude
        if method == 'peak':
            peak = np.max(spectrum)
            avg_magnitude = np.mean(spectrum)
            special_rows.append(peak / avg_magnitude)
        
        # Find the best row by selecting the row with the frequency of maximum magnitude
        elif method == 'max':
            peak = np.max(spectrum)
            special_rows.append(peak)

        # find the best row by using variance
        elif method == 'var':
            special_rows.append(np.var(spectrum))

    # if the best row appears close the 90 degrees region, mark this image as upright
    result_set = []
    angle_accum = 0
    index_accum = 0
    while angle_accum < 135:
        if angle_accum > 45:
            result_set.append(index_accum)
        angle_accum += angle_interval
        index_accum += 1
    if np.argmax(special_rows) in result_set:
        # the image is determined to be vertical (yes, angles between 135 and 45 is supposed to be vertical
        # , but due to how the radon transform was implemented, the image is rotated 90 degrees during the
        # transformation)
        return True
    else:
        print (result_set, np.argmax(special_rows))
        # the image is determined to be horizontal
        return False

start_time = time.time()
# This section takes in images from src, a folder with vertical images only, and sees how well the parameters
# of the function orientation_detect performs.
# s stands for success, f stands for failure, and i is the counter for the images
s = 0
f = 0
i = 0
for filename in os.listdir(src):
    imgpath = os.path.join(src, filename)
    if filename.endswith(('.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.gif', '.tif')):
        i += 1
        print(f'the {i}th image')
        if orientation_detect(imgpath, 15, 65, 1, 'max'):
            s += 1
        else:
            f += 1
            shutil.copy(imgpath, mistake_folder)
    print(f'success rate is {s/(s+f)}')
end_time = time.time()
time_length = end_time - start_time
print(f"Time elapsed: {time_length:.6f} seconds")