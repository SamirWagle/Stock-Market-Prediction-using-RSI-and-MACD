
 
# Stock Market Prediction with LSTM and Technical Analysis [RSI MACD]

This project predicts stock market prices using **LSTM (Long Short-Term Memory)** neural networks along with **technical analysis indicators** such as Moving Averages, MACD, and Bollinger Bands. It uses data from the Nepali Stock Exchange (NEPSE) to train and evaluate the model.

## Project Overview

The goal of this project is to build a predictive model that forecasts the future stock prices of a given company using historical stock prices and technical analysis. We implemented an LSTM model, a type of recurrent neural network (RNN), which is particularly well-suited for time series data like stock prices. This project integrates several financial indicators and uses them to improve prediction accuracy.

### Key Features:
- **Data Collection**: Stock price data is retrieved from CSV files hosted on GitHub.
- **Technical Analysis**: Calculation of indicators such as Moving Averages (MA), MACD, Bollinger Bands, Exponential Moving Averages (EMA), and Momentum.
- **LSTM Model**: An LSTM neural network is used for predicting future stock prices.
- **Data Visualization**: Graphical representation of stock prices and indicators.
- **Performance Evaluation**: The model's predictions are compared with actual stock prices.

## Technical Indicators Used
- **Moving Average (MA)**: A 7-day and 21-day moving average to smooth out price data.
- **MACD (Moving Average Convergence Divergence)**: A trend-following indicator calculated using 12-day and 26-day exponential moving averages.
- **Bollinger Bands**: Upper and lower bands to measure volatility.
- **Exponential Moving Average (EMA)**: Exponential smoothing for recent data.
- **Momentum**: Measures the speed of price changes.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `pandas_datareader`
  - `scikit-learn`
  - `tensorflow` (Keras)

You can install the required libraries using:
```bash
pip install -r requirements.txt
