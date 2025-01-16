# AQI Predictor with TinyML

The AQI (Air Quality Index) Predictor is a machine learning project designed to forecast air quality levels based on historical pollution data. It utilizes **TinyML frameworks like TensorFlow and TensorFlow Lite** to create a lightweight model suitable for deployment on mobile and embedded devices.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Training and Testing](#training-and-testing)
- [Team Members](#team-members)
- [Future Scope](#future-scope)

---

## Overview
Air quality has a direct impact on human health, and timely forecasting can help mitigate risks. The AQI Predictor tracks key pollutants like:
- **Nitrogen Oxides (NOx)**
- **Ozone (O3)**
- **Sulfur Dioxide (SO2)**
- **Ammonia (NH3)**
- **Particulate Matter (PM2.5 and PM10)**
- **Nitric Oxide (NO)**
- **Nitrogen Dioxide (NO2)**
- **Carbon Monoxide (CO)**

The project provides daily and monthly predictions for AQI, helping users make informed decisions to reduce exposure to pollution.

---

## Features
- Predicts hourly, daily, and monthly AQI values.
- Lightweight model optimized using TensorFlow Lite for efficient deployment.
- Visualizations for actual vs. predicted AQI trends.
- Uses historical data to predict future air quality levels.

---

## Project Workflow
1. **Data Preparation**:  
   - Load historical AQI data and preprocess it.
   - Use a sliding window approach to generate sequential data for time-series forecasting.

2. **Model Training**:  
   - Build a neural network with TensorFlow.
   - Train the model using pollutant data with hourly time windows.

3. **Model Conversion**:  
   - Convert the trained model to TensorFlow Lite format for deployment on low-power devices.

4. **Model Testing**:  
   - Test the model with unseen data.
   - Evaluate performance using Mean Squared Error (MSE) and visualize results.

5. **Daily and Monthly Predictions**:  
   - Use the trained model to predict hourly AQI values.
   - Aggregate hourly data into daily and monthly summaries.

---

## Model Architecture
The model is a sequential neural network with the following layers:
- Input Layer: Accepts flattened sequential data.
- Two Hidden Layers: Dense layers with 128 and 64 neurons, activated using ReLU.
- Output Layer: Predicts pollutant concentrations for the next time step.

---

## Training and Testing
- **Training**: The model is trained on 80% of the prepared data for 50 epochs using the Adam optimizer and Mean Squared Error loss function.
- **Testing**: The remaining 20% of the data is used to evaluate model performance.
- **Daily/Monthly Comparisons**: Visualizations compare actual vs. predicted AQI values at daily and monthly levels.

---

## Team Members
- Aaditya Chachra - 2022UCM2303
- Radhacharan - 2022UCM2365
- Aniket Rathore - 2022UCM2366
- Aman Kumar - 2022UCM2386

---

## Future Scope
- Enhance prediction accuracy by incorporating additional features like temperature, humidity, and wind speed.
- Integrate real-time data collection and processing for continuous monitoring.
- Deploy the model on IoT devices for on-site air quality predictions.
- Extend the model to predict other environmental parameters like weather and noise levels.

---
