# ### Mini Project: Human Face Recognition

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Dataset Path
dataset_dir = "face_dataset"

img_height = 224
img_width = 224
batch_size = 32

# # 🔧 FOR CPU
# img_height = 160
# img_width = 160
# batch_size = 8

# Data Generators
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

test_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

num_classes = train_generator.num_classes

# Load Pre-trained VGG16
base_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add Custom Classifier
x = base_model.output
x = Flatten()(x)

# # FOR CPU
# x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel Summary:")
model.summary()

# Train Model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate Model
loss, accuracy = model.evaluate(test_generator)

print("\nTest Accuracy:", accuracy)

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()
