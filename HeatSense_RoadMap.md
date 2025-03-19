# HeatSense: A Real-Time Global Temperature Monitoring with Kafka & ML

## 🌟 Project Overview

Climate change and temperature fluctuations are critical global issues affecting agriculture, logistics, energy consumption, and daily life. This project, **HeatSense**, is a comprehensive initiative aimed at addressing the challenges posed by climate change and temperature variability. By leveraging **Machine Learning (ML)**, **Deep Learning (DL)**, and **Apache Kafka**, it provides a real-time solution for monitoring, forecasting, and analyzing global temperature data. 🌍

### 🔑 Key Highlights

- **Real-Time Monitoring**: Continuous temperature data streaming using Kafka. 📡
- **Forecasting**: Predict future temperatures with advanced ML/DL models. 📈
- **Anomaly Detection**: Identify extreme temperature events like heatwaves and cold waves. ⚠️
- **Interactive Dashboards**: Visualize trends, forecasts, and alerts in real-time. 📊

### 🌐 Potential Applications

- **Agriculture**: Optimize crop planning and irrigation schedules. 🌾
- **Logistics**: Enhance supply chain efficiency by accounting for weather conditions. 🚚
- **Energy**: Predict energy demands based on temperature trends. ⚡
- **Research**: Facilitate climate studies and anomaly detection. 🔬

This project is a perfect blend of **data engineering** and **machine learning**, showcasing practical applications of technology to solve real-world problems.

### 🔥 Why This Project?

- **For Businesses**: Helps in planning logistics, demand forecasting, and risk assessment.
- **For Research**: Enables real-time climate analysis and anomaly detection.
- **For ML Engineers**: Showcases skills in **EDA, ML/DL modeling, experiment tracking, real-time streaming, API development, Test Driven Development (TDD) and dashboard visualization**.

---

## 🚀 Technical Stack

- **Data Science & ML**: Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch
- **Experiment Tracking**: MLflow, Weights & Biases
- **Streaming & Messaging**: Apache Kafka
- **Visualization**: Streamlit / Dash, Matplotlib, Seaborn
- **API Development**: FastAPI, Uvicorn
- **Logging**: Loguru
- **Testing**: Pytest
- **CI/CD**: GitHub Actions, Docker (Optional)

---

## 🔹 Sub Projects:

1. **HeatScope** 🕵️‍♂️ → Exploratory Data Analysis (EDA): Unveiling the Temperature Trends
2. **HeatCast** 🔮 → Temperature Forecasting (ML/DL Model): Predicting Tomorrow's Heat
3. **HeatAlert** 🚨 → Anomaly Detection (Heatwave & Coldwave Alerts): Detecting Extreme Temperatures
4. **HeatView** 📺 → Real-Time Dashboard (Streamlit / Dash): Live Temperature Monitoring
5. **HeatStream** 🌊 → Kafka Producer – Streaming Real-Time Temperature Data: Continuous Temperature Feed
6. **HeatStreamCast** 🔗 → Kafka Consumer – Temperature Forecasting (ML/DL Model): Forecasting the Future
7. **HeatGuard** 🛡️ → Kafka Consumer – Anomaly Detection (Heatwave & Coldwave Alerts): Real-Time Anomaly Detection
8. **HeatPulse** 💓 → Final Dashboard with Kafka (Real-Time Monitoring): Interactive Temperature Dashboard

---

## 🔹 Sub-Project 1: HeatScope – Exploratory Data Analysis

### 📌 Goal

Understand the structure, distribution, and anomalies in global temperature data, unveiling the temperature Trends.

### 📥 Input

- **Dataset**: `city_temperature.csv` (Daily temperature records from major cities) --> download it and store in dir `data/`
  - kaggle link 🔗: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities/data

### 📤 Output

- Statistical insights, visualizations (seasonality, trends, temperature variations)

### 🔑 Key Steps

1. Load and clean data (handle missing values, outliers, data types).
2. Aggregate statistics (mean, median, min-max temperature per city & year).
3. Visualize trends using **Matplotlib & Seaborn**.
4. Identify anomalies in temperature variations.

### 🛠️ Technologies

Pandas, NumPy, Matplotlib, Seaborn

---

## 🔹 Sub-Project 2: HeatCast – Temperature Forecasting

### 📌 Goal

Develop a predictive model to forecast future temperatures based on historical data.

### 📥 Input

- Processed temperature data from EDA.

### 📤 Output

- ML/DL model predicting future daily temperatures.

### 🔑 Key Steps

1. Feature engineering (time series transformation, lag features).
2. Train models (ARIMA, LSTM, GRU, Transformer models).
3. Evaluate & compare performance using RMSE, MAE.
4. Hyperparameter tuning for optimization.

### 🛠️ Technologies

Scikit-learn, TensorFlow/PyTorch, Statsmodels

---

## 🔹 Sub-Project 3: HeatAlert – Anomaly Detection

### 📌 Goal

Detect temperature anomalies (extreme temperature like heatwaves, cold waves) using ML-based anomaly detection.

### 📥 Input

- Temperature data from historical records.

### 📤 Output

- Flagging days with extreme temperatures as potential anomalies.

### 🔑 Key Steps

1. Define threshold-based vs ML-based anomaly detection.
2. Implement Isolation Forest, Autoencoders, and Statistical Thresholding.
3. Generate alerts for anomalies.

### 🛠️ Technologies

Scikit-learn, TensorFlow/PyTorch, PyOD (Python Outlier Detection)

---

## 🔹 Sub-Project 4: HeatView – Real-Time Dashboard

### 📌 Goal

Create an interactive dashboard to visualize temperature trends & anomaly alerts.

### 📥 Input

- Processed temperature data & model predictions.

### 📤 Output

- A web-based dashboard displaying real-time temperature trends & alerts.

### 🔑 Key Steps

1. Build UI with Streamlit/Dash.
2. Integrate data from Sub-Projects 2 & 3.
3. Add interactivity (city selection, forecast range, alert triggers).

### 🛠️ Technologies

Streamlit, Dash, Plotly

---

## 🔹 Sub-Project 5: HeatStream – Kafka Producer – Streaming Real-Time Temperature Data

### 📌 Goal

Simulate real-time temperature streaming using Apache Kafka.

### 📥 Input

- Historical temperature dataset streamed as a real-time feed.

### 📤 Output

- Kafka topics broadcasting temperature data.

### 🔑 Key Steps

1. Set up Kafka producer.
2. Stream temperature data to a Kafka topic.
3. Ensure message serialization & proper partitioning.

### 🛠️ Technologies

Apache Kafka, Python Kafka Client (confluent-kafka)

---

## 🔹 Sub-Project 6: HeatStreamCast – Kafka Consumer for Forecasting

### 📌 Goal

Consume real-time temperature data & generate future forecasts.

### 📥 Input

- Kafka topic streaming temperature data.

### 📤 Output

- Predicted temperature values stored & visualized.

### 🔑 Key Steps

1. Consume Kafka messages.
2. Preprocess data & make real-time predictions.
3. Store predictions in database.

### 🛠️ Technologies

Kafka, TensorFlow/PyTorch, PostgreSQL

---

## 🔹 Sub-Project 7: HeatGuard – Kafka Consumer for Anomaly Detection

### 📌 Goal

Consume temperature data & detect real-time anomalies.

### 📥 Input

- Kafka topic streaming temperature data.

### 📤 Output

- Alerts generated & stored for extreme temperature events.

### 🔑 Key Steps

1. Consume Kafka messages.
2. Apply anomaly detection models.
3. Trigger alerts for detected anomalies.

### 🛠️ Technologies

Kafka, PyOD, PostgreSQL

---

## 🔹 Sub-Project 8: HeatPulse – Final Dashboard with Kafka

### 📌 Goal

Integrate Kafka-driven real-time temperature data, forecasting, and anomaly alerts into a final interactive dashboard.

### 📥 Input

- Kafka-consumed temperature data, predictions, and alerts.

### 📤 Output

- Real-time visualization & alerting system.

### 🔑 Key Steps

1. Combine all previous components into a single Streamlit/Dash UI.
2. Enable real-time updates from Kafka.
3. Display insights & alerts interactively.

### 🛠️ Technologies

Streamlit, Kafka, PostgreSQL, TensorFlow/PyTorch

---

## 🌍 Final Thoughts

This project serves as a **real-world demonstration** of handling **real-time streaming data** with **ML/DL & Kafka**. It’s a powerful **showcase of data engineering, ML, and system design** that could be applicable in multiple industries.
