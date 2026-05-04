# ### Classification using Deep neural network
# Binary classification using Deep Neural Networks Example: Classify movie reviews into "positive" reviews and "negative" reviews, just based on the text content of the reviews.Use IMDB dataset

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load IMDB Dataset
# Load top 10,000 most frequent words
vocab_size = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# Data Preprocessing
# Pad sequences to fixed length
max_len = 250

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print("Shape after padding:", x_train.shape)

# Build Deep Neural Network Model
model = Sequential(
    [
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=128),  # Word Embedding Layer
        Bidirectional(LSTM(64, return_sequences=True)),  # Bi-directional LSTM Layer
        Bidirectional(LSTM(32)),  # Bi-directional LSTM Layer
        Dense(1, activation="sigmoid"),  # Output Layer
    ]
)

print("\nModel Summary:")
model.summary()

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)

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

# Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)
