
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)

X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")

model = Sequential([
    Embedding(input_dim=20000, output_dim=128),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

print("Starting BiLSTM Model Training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

model.save("models/bilstm_fake_news_model.keras")

print("\nBiLSTM MODEL TRAINING COMPLETED!")
print("Model saved successfully!")

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy * 80, "%")
-