# The function that validates the CNN model. If the model is trained on 
# images with Canny applied, then make sure Canny is enabled below (row 29)
# Normally, don't apply Canny edge detection!

import os
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2  # For image preprocessing

# The folder path is supposed to be a folder with all 0 rotation images. The code below will check for all four rotations of
# those images
MODEL_PATH = '4_rotations_384_gray_10th_7patience_canny.h5'  # Path to the trained model
IMAGE_FOLDER = 'C:\\D\\DSU\\github\\test_sample'  # Path to the folder containing images to predict
IMAGE_SIZE = (384, 384)  # Must match the input size used for training

# Class labels corresponding to the four output neurons
CLASS_LABELS = ['0 CCW', '180 CCW', '270 CCW', '90 CCW']

# Function to preprocess an image
def preprocess_image(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = gray_image.astype(np.uint8)
        # Apply Canny edge detection
        # edges = cv2.Canny(edges, 50, 120, apertureSize=3)
        # Normalize the image
        edges_normalized = edges / 255.0
        # Resize and add channel dimension
        resized_image = cv2.resize(edges_normalized, IMAGE_SIZE)
        return np.expand_dims(resized_image, axis=(0, -1))  # Add batch and channel dimensions
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load the trained model
model = load_model(MODEL_PATH)

# Predict images in the folder
def predict_images_in_folder(folder_path):
    predictions = []
    counter = 0
    total_time = 0
    # variable naming convention: 
    # l refers to the image being recognized as ccw 90 degrees, 
    # r refers to the image being recognized as ccw 270 degrees, 
    # u refers to the image being recognized as 0 degrees, 
    # d refers to the image being recognized as 180 degrees, 
    # on top of that, the number following one of the four letters simply tells the group which the result is in.
    l0 = 0
    r0 = 0
    u0 = 0
    d0 = 0

    l1 = 0
    r1 = 0
    u1 = 0
    d1 = 0

    l2 = 0
    r2 = 0
    u2 = 0
    d2 = 0
    
    l3 = 0
    r3 = 0
    u3 = 0
    d3 = 0
    
    for file_name in os.listdir(folder_path):
        num_of_files = len(os.listdir(folder_path))
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(folder_path, file_name)
            preprocessed_image = preprocess_image(image_path)
            if preprocessed_image is not None:
                counter += 1
                start_time = time.time()# Remove batch and channel dimensions for rotation
                image_to_rotate = preprocessed_image[0, :, :, 0]  # From (1, height, width, 1) to (height, width)
                preprocessed_image_90 = cv2.rotate(image_to_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)
                preprocessed_image_180 = cv2.rotate(preprocessed_image_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
                preprocessed_image_270 = cv2.rotate(preprocessed_image_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Add dimensions back after rotation
                preprocessed_image_90 = np.expand_dims(np.expand_dims(preprocessed_image_90, axis=0), axis=-1)
                preprocessed_image_180 = np.expand_dims(np.expand_dims(preprocessed_image_180, axis=0), axis=-1)
                preprocessed_image_270 = np.expand_dims(np.expand_dims(preprocessed_image_270, axis=0), axis=-1)

                # Predict using the model
                prediction0 = model.predict(preprocessed_image)
                predicted_label_index0 = np.argmax(prediction0)  # Index of the highest probability
                predicted_label0 = CLASS_LABELS[predicted_label_index0]
                if predicted_label0 == '0 CCW':
                    u0 += 1
                elif predicted_label0 == '90 CCW':
                    l0 += 1
                elif predicted_label0 == '270 CCW':
                    r0 += 1
                else:
                    d0 += 1
                print(f'                >>> P(0 given   0) is {u0/(u0+l0+r0+d0)*100}%')
                print(f'                    P(90 given  0) is {l0/(u0+l0+r0+d0)*100}%')
                print(f'                    P(180 given 0) is {d0/(u0+l0+r0+d0)*100}%')
                print(f'                    P(270 given 0) is {r0/(u0+l0+r0+d0)*100}%')
                # Predict using the model
                prediction0 = model.predict(preprocessed_image_90)
                predicted_label_index0 = np.argmax(prediction0)  # Index of the highest probability
                predicted_label0 = CLASS_LABELS[predicted_label_index0]
                if predicted_label0 == '0 CCW':
                    u1 += 1
                elif predicted_label0 == '90 CCW':
                    l1 += 1
                elif predicted_label0 == '270 CCW':
                    r1 += 1
                else:
                    d1 += 1
                print(f'                  > P(0 given   90) is {u1/(u1+l1+r1+d1)*100}%')
                print(f'                >>> P(90 given  90) is {l1/(u1+l1+r1+d1)*100}%')
                print(f'                    P(180 given 90) is {d1/(u1+l1+r1+d1)*100}%')
                print(f'                    P(270 given 90) is {r1/(u1+l1+r1+d1)*100}%')
                # Predict using the model
                prediction0 = model.predict(preprocessed_image_180)
                predicted_label_index0 = np.argmax(prediction0)  # Index of the highest probability
                predicted_label0 = CLASS_LABELS[predicted_label_index0]
                if predicted_label0 == '0 CCW':
                    u2 += 1
                elif predicted_label0 == '90 CCW':
                    l2 += 1
                elif predicted_label0 == '270 CCW':
                    r2 += 1
                else:
                    d2 += 1
                print(f'                  > P(0 given   180) is {u2/(u2+l2+r2+d2)*100}%')
                print(f'                    P(90 given  180) is {l2/(u2+l2+r2+d2)*100}%')
                print(f'                >>> P(180 given 180) is {d2/(u2+l2+r2+d2)*100}%')
                print(f'                    P(270 given 180) is {r2/(u2+l2+r2+d2)*100}%')
                # Predict using the model
                prediction0 = model.predict(preprocessed_image_270)
                predicted_label_index0 = np.argmax(prediction0)  # Index of the highest probability
                predicted_label0 = CLASS_LABELS[predicted_label_index0]
                if predicted_label0 == '0 CCW':
                    u3 += 1
                elif predicted_label0 == '90 CCW':
                    l3 += 1
                elif predicted_label0 == '270 CCW':
                    r3 += 1
                else:
                    d3 += 1
                print(f'                  > P(0 given   270) is {u3/(u3+l3+r3+d3)*100}%')
                print(f'                    P(90 given  270) is {l3/(u3+l3+r3+d3)*100}%')
                print(f'                    P(180 given 270) is {d3/(u3+l3+r3+d3)*100}%')
                print(f'                >>> P(270 given 270) is {r3/(u3+l3+r3+d3)*100}%')
                end_time = time.time()
                current_time_period = (end_time - start_time)
                total_time += current_time_period
                avg_time = total_time/counter
                est_eventual_time = avg_time * num_of_files
                est_remaining_time = (num_of_files - counter) * avg_time
                print(f'                    Avg time: {avg_time} seconds')
                print(f'                    Tot time: {est_eventual_time//1} seconds')
                print(f'                    Rem time: {est_remaining_time//1} seconds')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                predictions.append((file_name, predicted_label0))
    return predictions

# Run the predictions
predictions = predict_images_in_folder(IMAGE_FOLDER)

# Save results to a text file
with open('predictions.txt', 'w') as f:
    for file_name, label in predictions:
        f.write(f"{file_name}: {label}\n")

print("Predictions saved to 'predictions.txt'.")