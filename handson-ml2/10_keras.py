import tensorflow as tf 
from tensorflow import keras
import os
import numpy as np
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt 

# basic models are in basic.py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_valid = scaler.transform(x_valid)

# functional api
inputs = keras.layers.Input(shape=x_train.shape[1:])
x = keras.layers.Dense(30, activation='relu')(inputs)
x = keras.layers.Dense(30, activation='relu')(x)
concat = keras.layers.concatenate([inputs, x])
outputs = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[inputs], outputs=[outputs])

model.summary()
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.SGD(lr=1e-3),
)
history = model.fit(x_train, y_train, epochs=20, 
    validation_data=[x_valid, y_valid])
mse_test = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_hat = np.squeeze(y_pred)

# subclassing
class widedeepmodel(keras.models.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation)
        self.hidden2 = keras.layers.Dense(units, activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_a, input_b = inputs
        hidden1 = self.hidden1(input_b)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_a, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(concat)
        return main_output, aux_output

model = widedeepmodel(units=30, activation='relu')

# callbacks
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("tf10.h5", save_best_only=True)

# tensorboard
