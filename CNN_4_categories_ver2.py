# The code that trains the model which recognizes the orientation of the 
# image in terms of multiples of 90 degrees.
# Normally, don't apply Canny edge detection! (row 35)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
import os
import cv2
from PIL import ImageFile
import numpy as np
import scipy
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# This is where the images should be
base_dir = 'C:\\D\\DSU\\Kanagaratnam\\Train 10th - Copy'
train_dir = os.path.join(base_dir)

# Image parameters
IMAGE_SIZE = (384, 384)
BATCH_SIZE = 32

# The function to apply grayscale conversion and Canny edge detection
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.uint8)  # Convert to uint8 just to be safe
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 120, apertureSize=3)
    
    # Normalize the image (important since we're dividing by 255 later in rescale)
    edges_normalized = edges / 255.0
    
    # Expand dimensions to make it compatible with CNN input
    return np.expand_dims(edges_normalized, axis=-1)  # Add channel dimension

# custom generator that works with categorical labels
def custom_preprocessing_generator(generator):
    while True:
        try:
            batch_x, batch_y = next(generator)
            batch_x_processed = np.array([preprocess_image(x) for x in batch_x])
            yield batch_x_processed, batch_y
        except OSError as e:
            print(f"Error loading image, skipping batch: {e}")

# Data augmentation and basic preprocessing with ImageDataGenerator
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=5,
    validation_split=0.1
)

# Create training and validation datasets
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Create custom generators that include the preprocessing
train_preprocessed = custom_preprocessing_generator(train_generator)
validation_preprocessed = custom_preprocessing_generator(validation_generator)

# Build the CNN model
# Update model architecture for 4 classes
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(384, 384, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),

    # Fourth convolutional block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.5),

    # Global Average Pooling and dense layers
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Output layer with 4 classes
])


# Print model summary to inspect layer sizes
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# EarlyStopping: Stops training when validation accuracy stops improving
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=4,  # Stop after 5 epochs of no improvement
    restore_best_weights=True  # Restore the weights of the best epoch
)

# ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.7,  # Reduce learning rate by a factor of 0.5
    patience=3,  # Wait for 3 epochs before reducing
    min_lr=2e-6  # Set a minimum learning rate
)

# Train the model using the custom preprocessed generator
history = model.fit(
    train_preprocessed,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=80,  # Adjust number of epochs as needed
    validation_data=validation_preprocessed,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler]  # Add callbacks here
)

# Save the model for future use
model.save('4_rotations_384_canny_10th_NameChange_No2.h5')