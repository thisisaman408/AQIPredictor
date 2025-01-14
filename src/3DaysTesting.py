import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter


data = pd.read_csv('Data/FinalData/AmanData.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values(by='timestamp', inplace=True)


features = ['NOx', 'O3', 'SO2', 'NH3', 'PM2.5', 'PM10', 'NO', 'NO2', 'CO']
window_size = 24


def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_next_window(interpreter, input_details, output_details, last_window):
    input_data = np.array([last_window], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data

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


last_window = data[features].iloc[-window_size:].values.flatten()


predictions = []
current_window = last_window.copy()

for day in range(1, 4):
    predicted_pollutants = predict_next_window(interpreter, input_details, output_details, current_window)
    predicted_aqi = calculate_overall_aqi(dict(zip(features, predicted_pollutants)), breakpoints)
    predictions.append({
        'day': f"2023-01-{24 + day}",
        'predicted_aqi': predicted_aqi
    })

    
    current_window = np.roll(current_window, -len(features))
    current_window[-len(features):] = predicted_pollutants


predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('Data/FinalData/PredictedFOrUnseenDatra.csv', index=False)

print("Predictions for the next 3 days saved to 'Data/FinalData/Predicted_AQI_Next_3_Days.csv'.")
