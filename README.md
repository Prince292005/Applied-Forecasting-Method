# India Power Demand Forecasting 🇮🇳⚡

Forecasting India’s daily electricity demand using Statistical Models, Deep Learning Architectures, and Probabilistic Forecasting techniques.

---

## 📌 Project Overview

This project presents a comprehensive forecasting study on India’s national daily power demand using time series data from January 2019 to April 2024.

The project compares:
- Classical Statistical Models
- Deep Learning Models
- Probabilistic Forecasting Methods

The study was conducted as part of the **Applied Forecasting Methods** course at **Dhirubhai Ambani University**.

---

# 🎯 Objectives

- Forecast India’s daily electricity demand accurately
- Compare statistical and deep learning models
- Implement walk-forward forecasting
- Generate probabilistic prediction intervals
- Evaluate models using MAPE

---

# 📊 Dataset

## Source
National Power Portal (NPP), Central Electricity Authority (CEA), India

https://npp.gov.in

## Dataset Information

| Feature | Value |
|---|---|
| Time Range | Jan 2019 – Apr 2024 |
| Original Frequency | Hourly |
| Resampled Frequency | Daily |
| Total Daily Observations | 1,947 |
| Original Records | 46,728 hourly records |
| Target Variable | National Daily Power Demand |
| Unit | MWh/day |

---

# 🧠 Models Implemented

## 📈 Statistical Models

- AR (AutoRegressive)
- MA (Moving Average)
- ARMA
- ARIMA
- SARIMA

### Features
- Walk-forward forecasting
- Periodic refitting
- Log transformation
- One-step-ahead prediction

---

## 🤖 Deep Learning Models

- RNN
- LSTM
- GRU
- MLP
- TCN (Temporal Convolutional Network)

### Pipeline
- Log transformation
- MinMax scaling
- 30-day sliding window
- Early stopping
- TensorFlow/Keras implementation

---

## 📉 Probabilistic Forecasting

### Quantile Regression

Implemented:
- QR-10
- QR-50
- QR-90

Features:
- Prediction intervals
- Lag features
- Rolling statistics
- Calendar features

---

# 🔄 Walk-Forward Forecasting

The statistical models use a realistic walk-forward forecasting strategy:

1. Train model on historical data
2. Predict next day
3. Add actual observation
4. Refit periodically
5. Continue sequential forecasting

Example:

```python
def walk_forward_arima(train_series, val_series, test_series,
                       order, seasonal_order=None, refit_every=7):
