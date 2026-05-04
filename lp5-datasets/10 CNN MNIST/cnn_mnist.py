# ### Convolutional neural network (CNN)
# Use MNIST Fashion Dataset and create a classifier to classify fashion clothing into
# categories.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Training samples:", train_images.shape)
print("Testing samples:", test_images.shape)

# Data Preprocessing
# Normalize pixel values (0–255 → 0–1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add channel dimension (required for CNN)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Define Class Names
class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

# Build CNN Model
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        ),  # Convolution Layer 1
        MaxPooling2D((2, 2)),  # Convolution Layer 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),  # Flatten Layer
        Flatten(),  # Fully Connected Layer
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),  # Output Layer (10 classes)
    ]
)

# Compile Model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# Train Model
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

print("\nTest Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()

# Display Predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {class_names[predicted_labels[i]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
