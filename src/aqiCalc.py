import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('Data/FinalData/AmanData.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values(by='timestamp', inplace=True)

features = ['NOx', 'O3', 'SO2', 'NH3', 'PM2.5', 'PM10', 'NO', 'NO2', 'CO']


breakpoints = {
    'PM2.5': [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    'PM10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    'SO2': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
    'CO': [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    'O3': [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300), (201, 300, 301, 400), (301, 400, 401, 500)]
}

def calculate_aqi_for_pollutant(concentration, pollutant):
    for bp in breakpoints[pollutant]:
        c_lo, c_hi, i_lo, i_hi = bp
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    return None

def calculate_overall_aqi(pollutant_values):
    aqi_values = []
    for pollutant, concentration in pollutant_values.items():
        if pollutant in breakpoints:
            aqi = calculate_aqi_for_pollutant(concentration, pollutant)
            if aqi is not None:
                aqi_values.append(aqi)
    return max(aqi_values) if aqi_values else None

def prepare_data(data, features, window_size):
    X, y, timestamps = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values.flatten())
        y.append(data[features].iloc[i+window_size].values)
        timestamps.append(data['timestamp'].iloc[i+window_size])
    return np.array(X), np.array(y), timestamps


window_size = 24
X, y, all_timestamps = prepare_data(data, features, window_size)


X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(
    X, y, all_timestamps, test_size=0.2, random_state=42
)

actual_aqi = [
    calculate_overall_aqi(dict(zip(features, y_test[i]))) for i in range(len(y_test))
]

aqi_df = pd.DataFrame({
    'timestamp': timestamps_test,
    'actual_aqi': actual_aqi
})

aqi_df.to_csv('Test_AQI_Data.csv', index=False)
print("Test AQI data saved to Test_AQI_Data.csv")
