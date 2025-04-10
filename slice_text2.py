# This is used to pre-process images
# with texts so that they can be fed into the CNN model with more
# emphasis on the letters. After all, what best characterizes the 
# orientation of images are the letters.

# This code now achieves an average processing speed of around 0.25 sec per images on i9-9880h



import time
import numpy as np
import cv2
import os
from multiprocessing import Pool

src = 'C:\\D\\DSU\\Dragomans\\testing'

# Load the EAST model once
net = cv2.dnn.readNet(r"C:\\Users\\user\\Downloads\\frozen_east_text_detection.pb")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Change to DNN_TARGET_CUDA if GPU is available

def process_with_east(input_image_path, crop_width, crop_height):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return None, None
    
    (H, W) = image.shape[:2]

    image = image[H // 6 : 5*H // 6, W // 6 : 5*W // 6]
    H = 2*H // 3
    W = 2*W // 3
    
    # After testing, I find this scaling factor to be the fastest one without compromising noticable accuracy
    # You can tune it based on the capabilities of your machine(s).
    scale_factor = 900 / min(H, W)
    newW = (int(W * scale_factor) + 31) // 32 * 32
    newH = (int(H * scale_factor) + 31) // 32 * 32
    rW, rH = W / float(newW), H / float(newH)
    
    # Image preprocessing (optimized)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sharpened_image = cv2.filter2D(gray, -1, np.array([[0, -1.2, 0], [-1.2, 6, -1.2], [0, -1.2, 0]]))
    resized_image = cv2.resize(gray, (newW, newH), interpolation=cv2.INTER_NEAREST)
    
    # Prepare the image for EAST model
    blob1 = cv2.dnn.blobFromImage(cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR), 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob1)
    scores1, geometry1 = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    # score2 corresponds to the image rotated 90 degrees
    rotated_reized_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
    blob2 = cv2.dnn.blobFromImage(cv2.cvtColor(rotated_reized_image, cv2.COLOR_GRAY2BGR), 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob2)
    scores2, geometry2 = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    def east_predict(scores, geometry, image):
        # Decode the predictions (optimized)
        mask = scores[0, 0] > 0.97
        y_idx, x_idx = np.where(mask)
        rects, confidences, orientations = [], [], []
        
        for y, x in zip(y_idx, x_idx):
            score = scores[0, 0, y, x]
            angle = geometry[0, 4, y, x]
            w, h = geometry[0, 1, y, x], geometry[0, 0, y, x]
            
            offsetX, offsetY = x * 4.0, y * 4.0
            cos, sin = np.cos(angle), np.sin(angle)
            startX, startY = int(offsetX - (cos * w) - (sin * h)), int(offsetY + (sin * w) - (cos * h))
            endX, endY = int(offsetX + (cos * w) + (sin * h)), int(offsetY - (sin * w) + (cos * h))
            
            rects.append((startX, startY, endX, endY))
            confidences.append(score)
            orientations.append((round(np.rad2deg(angle) / 90) * 90) % 360)
        
        if not rects:
            print("No valid bounding box found with confidence > 0.97.")
            return None, None
        
        # Select the highest confidence box
        best_index = np.argmax(confidences)
        print(max(confidences))
        count = len(confidences)
        startX, startY, endX, endY = rects[best_index]
        startX, startY, endX, endY = int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)
        
        centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
        x, y = max(0, centerX - crop_width // 2), max(0, centerY - crop_height // 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_image = image[y:y+crop_height, x:x+crop_width]
        _, binarized = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binarized, image, count
    bin1, im1, _ = east_predict(scores1, geometry1, image)
    return bin1, im1

def process_image(imgpath):
    try:
        nm, ext = os.path.splitext(os.path.basename(imgpath))
        start_time = time.time()
        cropped, _ = process_with_east(imgpath, 768, 768)
        if cropped is not None:
            cv2.imwrite(os.path.join(os.path.dirname(imgpath), f'{nm}_Slice{ext}'), cropped)
        print(f"Processed {imgpath} in {time.time() - start_time:.4f} seconds")
    except Exception as e:
        print(f"Error processing {imgpath}: {e}")

# Get all image files
image_paths = [
    os.path.join(root, filename)
    for root, _, files in os.walk(src)
    for filename in files if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
]

# Parallel processing
if __name__ == "__main__":
    with Pool(processes=8) as pool:  # Adjust based on CPU cores
        pool.map(process_image, image_paths)