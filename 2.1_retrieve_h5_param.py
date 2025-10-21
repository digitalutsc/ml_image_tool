# Inspects and prints the basic info of the CNN model

from tensorflow.keras.models import load_model

# Load the model you want to inspect
model = load_model("4_rotations_384_gray_10th_7patience_canny.h5")

# Summarize model structure
model.summary()

# Check optimizer config (if available)
optimizer_config = model.optimizer.get_config()
print("Optimizer Config:")
print(optimizer_config)

# Check loss function
print("Loss function:", model.loss)

# Check metrics
print("Metrics:", model.metrics_names)
