import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# Parameters
model_path = '256_binary_orientation_model.h5'
root_folder = 'C:/D/DSU/Kanagaratnam/TestImages'
image_size = (256, 256)
output_folder = root_folder  # Change this if you want to save corrected images elsewhere
max_workers = os.cpu_count()

# Load model globally in each process
model = None
def load_shared_model():
    global model
    if model is None:
        model = load_model(model_path)
    return model

def rotate_image(img, k):
    return np.rot90(img, k).copy()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, 256, 256, 1)

def correct_orientation(image_path, output_folder):
    model = load_shared_model()
    filename = os.path.basename(image_path)

    image = cv2.imread(image_path)
    if image is None:
        return f"Failed to read {image_path}"

    scores = []
    for k in range(4):
        rotated = rotate_image(image, k)
        input_tensor = preprocess(rotated)
        score = model.predict(input_tensor, verbose=0)[0][0]
        scores.append(score)

    best_k = int(np.argmax(scores))
    corrected = rotate_image(image, best_k)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, corrected)
    return f"Processed: {filename}, best rotation: {best_k * 90}°"

# Gather all image paths
image_paths = []
for dirpath, _, filenames in os.walk(root_folder):
    for f in filenames:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            image_paths.append(os.path.join(dirpath, f))

# Run with multiprocessing
if __name__ == '__main__':
    print(f"Processing {len(image_paths)} images using {max_workers} workers...\n")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(correct_orientation, path, output_folder) for path in image_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Orientation Correction"):
            try:
                result = f.result()
                # Optionally: print(result) if you want per-file logging
            except Exception as e:
                print(f"Error processing image: {e}")
