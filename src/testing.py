import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def test_tflite_model(tflite_model_path, X_test, y_test):
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []

    for i in range(len(X_test)):
        input_data = np.array([X_test[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    predictions = np.array(predictions)
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error on Test Data: {mse}")

    return predictions

predictions = test_tflite_model('FinalModel/TinyTroopers.tflite', X_test, y_test)

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.plot(y_test[:, i], label='Actual', alpha=0.7)
    plt.plot(predictions[:, i], label='Predicted', alpha=0.7)
    plt.title(feature)
    plt.legend()

plt.tight_layout()
plt.show()
