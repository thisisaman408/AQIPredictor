import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import os
import matplotlib.pyplot as plt


data = pd.read_csv('Data/FinalData/Test_AQI_Data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by='timestamp')


os.makedirs('visualisation', exist_ok=True)
os.makedirs('Data/FinalData', exist_ok=True)


features = ['NOx', 'O3', 'SO2', 'NH3', 'PM2.5', 'PM10', 'NO', 'NO2', 'CO']


def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_values(interpreter, input_details, output_details, X):
    predictions = []
    for i in range(len(X)):
        input_data = np.array([X[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
    return np.array(predictions)

def calculate_aqi_for_pollutant(concentration, pollutant, breakpoints):
    for bp in breakpoints[pollutant]:
        c_lo, c_hi, i_lo, i_hi = bp
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    return None

def calculate_overall_aqi(pollutant_values, breakpoints):
    aqi_values = []
    for pollutant, concentration in pollutant_values.items():
        if pollutant in breakpoints:
            aqi = calculate_aqi_for_pollutant(concentration, pollutant, breakpoints)
            if aqi is not None:
                aqi_values.append(aqi)
    return max(aqi_values) if aqi_values else None


breakpoints = {
    'PM2.5': [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    'PM10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
    'SO2': [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
    'CO': [(0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    'O3': [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300), (201, 300, 301, 400), (301, 400, 401, 500)]
}


interpreter, input_details, output_details = load_model('FinalModel/TinyTroopers.tflite')


window_size = 24

def prepare_sliding_windows(data, features, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[features].iloc[i:i+window_size].values.flatten())
    return np.array(X)

test_features = pd.read_csv('Data/FinalData/AmanData.csv')
test_features = prepare_sliding_windows(test_features, features, window_size)


predicted_values = predict_values(interpreter, input_details, output_details, test_features)


data['month'] = data['timestamp'].dt.to_period('M')
accuracy_data = []

for month, group in data.groupby('month'):
    monthly_actual_aqi = group['actual_aqi'].mean()

    monthly_predicted_aqi = np.mean([
        calculate_overall_aqi(dict(zip(features, predicted_values[i])), breakpoints) for i in group.index
    ])

   
    accuracy_data.append({
        'month': str(month),
        'monthly_actual_aqi': monthly_actual_aqi,
        'monthly_predicted_aqi': monthly_predicted_aqi,
        'accuracy': 100 - abs((monthly_predicted_aqi - monthly_actual_aqi) / monthly_actual_aqi * 100)
    })

   
    plt.figure(figsize=(10, 6))
    plt.bar(['Actual AQI', 'Predicted AQI'], [monthly_actual_aqi, monthly_predicted_aqi], color=['blue', 'orange'])
    plt.ylabel('AQI')
    plt.title(f'Monthly AQI Comparison for {month}')
    plt.tight_layout()

   
    plt.savefig(f'visualisation/{month}.png')
    plt.close()


accuracy_df = pd.DataFrame(accuracy_data)
accuracy_df.to_csv('Data/FinalData/Monthly_AQI_Accuracy.csv', index=False)

print("Monthly AQI comparison images saved in 'visualisation' folder.")
print("Monthly accuracy data saved to 'Data/FinalData/Monthly_AQI_Accuracy.csv'.")
