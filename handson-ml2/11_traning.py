import numpy as np 
import pandas as pd 
import os 
import tensorflow as tf 
from tensorflow import keras 
import sklearn
import matplotlib.pyplot as plt 

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

