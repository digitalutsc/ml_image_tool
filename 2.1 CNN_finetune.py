# The function that fine-tune the CNN model with extra images. If the model is trained on 
# images with Canny applied, then make sure Canny is enabled below (row 36)
# Normally, don't apply Canny edge detection!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Paths
base_dir = 'C:\\D\\DSU\\Kanagaratnam\\Train 10th - Copy'
train_dir = os.path.join(base_dir)
model_path = '4_rotations_384_canny_10th_NameChange_No2.h5'

# === Image Parameters ===
IMAGE_SIZE = (384, 384)
BATCH_SIZE = 32

# === Preprocessing: Grayscale + Canny ===
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, 50, 120, apertureSize=3)
    edges_normalized = edges / 255.0
    return np.expand_dims(edges_normalized, axis=-1)

# === Custom generator ===
def custom_preprocessing_generator(generator):
    while True:
        try:
            batch_x, batch_y = next(generator)
            batch_x_processed = np.array([preprocess_image(x) for x in batch_x])
            yield batch_x_processed, batch_y
        except OSError as e:
            print(f"Error loading image, skipping batch: {e}")

# === Data Augmentation ===
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=5,
    validation_split=0.1
)

# === Generators ===
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

train_preprocessed = custom_preprocessing_generator(train_generator)
validation_preprocessed = custom_preprocessing_generator(validation_generator)

# === Load and Modify Model ===
model = load_model(model_path)

# Freeze all layers
for layer in model.layers:
    layer.trainable = False

# Unfreeze only the last Dense layers (adaptable!)
model.layers[-1].trainable = True         # Dense(4)
model.layers[-3].trainable = True         # Dense(512) — optional

# Optional: confirm what’s trainable
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

# Recompile with lower LR
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=3,
    min_lr=2e-6
)

# === Fine-Tune the Model ===
history = model.fit(
    train_preprocessed,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=20,
    validation_data=validation_preprocessed,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler]
)

# === Save the Fine-Tuned Model ===
model.save('4_rotations_384_canny_10th_NameChange_No2_finetuned.h5')
