# ### Linear regression by using Deep Neural network:
# Implement Boston housing price prediction problem by Linear regression using Deep Neural network. Use Boston House price prediction dataset.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
df = pd.read_csv("boston_housing.csv")

print(df.head())


# Data Preprocessing
# Missing Data
df = df.fillna(df.mean())

# Separate features (X) and target (y)
X = df.drop("MEDV", axis=1)  # MEDV = Median house value (Target)
y = df["MEDV"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled feature sample:")
X_scaled[:1]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# Build Deep Neural Network Model
model = Sequential(
    [
        Input(shape=(13,)),  # Input Layer
        Dense(64, activation="relu"),  # Hidden Layer 1
        Dropout(0.2),
        Dense(32, activation="relu"),  # Hidden Layer 2
        Dense(1),  # Output layer
    ]
)

print("\nModel Summary:")
model.summary()


# Compile Model
model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)


# Train Model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)


# Training Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()


# Evaluate Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Mean Squared Error :", loss)
print("Test Mean Absolute Error:", mae)
