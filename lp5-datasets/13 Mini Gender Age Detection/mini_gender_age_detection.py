# Mini Project: Gender and Age Detection: predict if a person is a male or female and also their age
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Input,
    GlobalAveragePooling2D,
)
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

import zipfile

zip_path = "./UTKFace.zip"
extract_path = "./UTKFace"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction done!")

# # Only for GPU #REMOVE THIS#
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# CONFIG
DATA_DIR = "./UTKFace"  # change if needed
IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 16

# # FOR CPU
# BATCH_SIZE = 8

# Get valid files only
all_files = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".jpg"):
        parts = file.split("_")
        if len(parts) >= 2:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                all_files.append(file)
            except:
                continue

print("Total valid images:", len(all_files))

# Train-Test Split
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)


# Data Generator
class DataGenerator(Sequence):
    def __init__(self, files, image_dir, batch_size=16, img_size=64):
        self.files = files
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = []
        ages = []
        genders = []

        for file in batch_files:
            try:
                parts = file.split("_")
                age = int(parts[0])
                gender = int(parts[1])

                img_path = os.path.join(self.image_dir, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img / 255.0

                images.append(img)
                ages.append(age / 100.0)
                genders.append(gender)

            except:
                continue

        return np.array(images), {"gender": np.array(genders), "age": np.array(ages)}


# Create generators
train_gen = DataGenerator(train_files, DATA_DIR, BATCH_SIZE, IMG_SIZE)
val_gen = DataGenerator(val_files, DATA_DIR, BATCH_SIZE, IMG_SIZE)

# Model
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3, 3), activation="relu")(input_layer)
x = MaxPooling2D()(x)

x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3, 3), activation="relu")(x)
x = MaxPooling2D()(x)

x = Flatten()(x)

# # 🔧 FOR CPU
# x = GlobalAveragePooling2D()(x)

# Outputs
gender_output = Dense(1, activation="sigmoid", name="gender")(x)
age_output = Dense(1, activation="linear", name="age")(x)

model = Model(inputs=input_layer, outputs=[gender_output, age_output])

model.compile(
    optimizer="adam",
    loss={"gender": "binary_crossentropy", "age": "mse"},
    metrics={"gender": "accuracy", "age": "mae"},
)

model.summary()

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save Model
model.save("age_gender_model.h5")


# Prediction Function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    gender_pred, age_pred = model.predict(img)

    gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
    age = int(age_pred[0][0] * 100)

    print(f"Predicted Gender: {gender}")
    print(f"Predicted Age: {age}")


# Test Prediction
# Put any image path here
predict_image("./UTKFace/39_0_0_20170104202631251.jpg.chip.jpg")
