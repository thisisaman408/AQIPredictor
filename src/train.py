import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data/FinalData/AmanData.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values(by='timestamp', inplace=True)

features = ['NOx', 'O3', 'SO2', 'NH3', 'PM2.5', 'PM10', 'NO', 'NO2', 'CO']

def prepare_data(data, features, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values.flatten())
        y.append(data[features].iloc[i+window_size].values)
    return np.array(X), np.array(y)

window_size = 24
X, y = prepare_data(data, features, window_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.Sequential([
    keras.layers.Input(shape=(window_size * len(features),)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(features))
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('TinyTroopers.tflite', 'wb') as f:
    f.write(tflite_model)
