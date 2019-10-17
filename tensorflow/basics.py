import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras  # tf.keras
import time
from sklearn.preprocessing import StandardScaler

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = (fashion_mnist.load_data())
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 2.1
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
keras.utils.plot_model(model)

model.compile(
    optimizer=keras.optimizers.SGD(lr=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(X_train, y_train, epochs=10)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

model.evaluate(X_test, y_test)

X_new = X_test[:10,:,:]
y_pred = model.predict(X_new)
y_pred = y_pred.round(3)
np.argmax(y_pred, axis=1)
y_class = model.predict_classes(X_new)
np.max(y_pred, axis=1)
y_sort = np.argsort(-y_pred, axis=1)
y_sort = y_sort[:,:3]
y_prob = -np.sort(-y_pred, axis=1)[:, :3]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train.astype(np.float).reshape(-1, 28 ** 2)
).reshape(-1, 28, 28)
X_test_scaled = scaler.transform(
    X_test.astype(np.float).reshape(-1, 28 ** 2)
).reshape(-1, 28, 28)
X_valid_scaled = scaler.transform(
    X_valid.astype(np.float).reshape(-1, 28 ** 2)
).reshape(-1, 28, 28)

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizers='adam',
#     metrics=['accuracy']
# )
history_scaled = model.fit(
    X_train_scaled, y_train, epochs=20,
    validation_data=(X_valid_scaled, y_valid)
)
pd.DataFrame(history_scaled.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
