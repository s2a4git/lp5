# ### Recurrent neural network (RNN)
# Use the Google stock prices dataset and design a time series analysis and prediction system using RNN.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
data = pd.read_csv("GOOGL.csv")

print("Dataset shape:", data.shape)
print(data.head())

# Data Preprocessing
# Use 'Open' price for prediction
dataset = data["Open"].values.reshape(-1, 1)

# Normalize data (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Split into training (80%) and testing (20%)
training_data_len = int(len(dataset_scaled) * 0.8)

training_data = dataset_scaled[:training_data_len]
testing_data = dataset_scaled[training_data_len:]


# Create Time Series Dataset
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : (i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 60

X_train, Y_train = create_dataset(training_data, time_step)
X_test, Y_test = create_dataset(testing_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Build RNN (LSTM) Model
model = Sequential(
    [
        Input(shape=(time_step, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1),
    ]
)

# Compile Model
model.compile(optimizer="adam", loss="mean_squared_error")

print("\nModel Summary:")
model.summary()

# Train Model
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    Y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# Model Evaluation
predictions = model.predict(X_test)

# Inverse transform to get actual prices
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(Y_test_actual, predictions))

print("\nRoot Mean Squared Error (RMSE):", rmse)

# Visualization
plt.figure(figsize=(12, 6))

plt.plot(Y_test_actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Google Stock Price Prediction using RNN")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()
