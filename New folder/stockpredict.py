#!/usr/bin/env python
# coding: utf-8

# # Install all

#!pip install seaborn
#!pip install pandas_datareader
#!  pip install keras


# # Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from datetime import date
import math
import pandas_datareader as web


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


url = 'https://raw.githubusercontent.com/Aabishkar2/nepse-data/refs/heads/main/data/company-wise/ADBL.csv'
data = pd.read_csv(url)
selected_data = data[['published_date', 'close']]
selected_data.head()



print('There are {} number of days in the dataset.'.format(selected_data.shape[0]))


# Plotting the graph
# 

plt.figure(figsize=(28, 12))#, dpi=100)
plt.plot(selected_data.index, data['close'], label='ADBL')
plt.xlabel('published_date')
plt.ylabel('Rs')
plt.title('Stock Graph')
plt.legend()
plt.show()


# # Logic For Technical  Analysis
# 
# Moving averages (7 and 21 dayss)
# MACD
# Bolinger Bands
# EMA
# Momentum

def technical_analysis(dataset):
    # Moving Average of 7 and 21 days
    dataset.loc[:, 'ma7'] = dataset['close'].rolling(window=7).mean()
    dataset.loc[:, 'ma21'] = dataset['close'].rolling(window=21).mean()

    # Create MACD
    dataset.loc[:, '26ema'] = dataset['close'].ewm(span=26).mean()
    dataset.loc[:, '12ema'] = dataset['close'].ewm(span=12).mean()
    dataset.loc[:, 'MACD'] = dataset['12ema'] - dataset['26ema']

    # Create Bollinger Bands
    dataset.loc[:, '20sd'] = dataset['close'].rolling(window=21).std()
    dataset.loc[:, 'upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset.loc[:, 'lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    # Create Exponential moving average
    dataset.loc[:, 'ema'] = dataset['close'].ewm(com=0.5).mean()

    # Create Momentum
    dataset.loc[:, 'momentum'] = dataset['close'] - 1
    dataset.loc[:, 'log_momentum'] = np.log(dataset['momentum'])

    return dataset


# # Relative Strength Index

def calculate_RSI(data, window=14):
    # Calculate price changes
    delta = data['close'].diff(1)
    
    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the rolling average of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the RS (Relative Strength)
    RS = avg_gain / avg_loss

    # Calculate RSI
    RSI = 100 - (100 / (1 + RS))
    
    # Assign the RSI to a new column
    data['RSI'] = RSI
    
    return data

# Add RSI to your dataset
df = calculate_RSI(df)
df.head()


df=technical_analysis(selected_data)


df


df = df.dropna()
df.rows = df.iloc[0]
df

# Drop the first row after setting it as the header
df = df.set_index('published_date')

print(df)
df.head()


# # Plotting these analysis

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)

    plt.figure(figsize=(30,20))
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['close'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators Agriculture Development Bank Limited {} days.'.format(last_days))
    plt.ylabel('NPR')
    plt.legend()

    # Plot second subplot

    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
#     plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
#     plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend()
    plt.show()


plot_technical_indicators(df, 100)


# # RSI
# 

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    plt.figure(figsize=(30, 25))
    
    # Plot first subplot: Moving Averages and Bollinger Bands
    plt.subplot(3, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['close'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title(f'Technical indicators for last {last_days} days')
    plt.ylabel('NPR')
    plt.legend()

    # Plot second subplot: MACD
    plt.subplot(3, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.plot(dataset['log_momentum'], label='Momentum', color='b', linestyle='-')
    plt.legend()

    # Plot third subplot: RSI
    plt.subplot(3, 1, 3)
    plt.title('RSI')
    plt.plot(dataset['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')  # Overbought
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')  # Oversold
    plt.ylabel('RSI')
    plt.legend()

    plt.show()

plot_technical_indicators(df, 100)


df


plt.figure(figsize = (28,12))
sns.set_context('poster',font_scale=1)
sns.heatmap(df.corr(), annot = True).set_title('Params')


print('Total dataset has {} samples, and {} features.'.format(df.shape[0], \
                                                              df.shape[1]))


df.columns


df


data_training = df[df.index < '2022-01-31'].copy()
data_training


data_testing = df[df.index >= '2022-01-31'].copy()
data_testing


# 

scalar = MinMaxScaler()

data_training_scaled = scalar.fit_transform(data_training)
print(data_training_scaled.shape)
data_training_scaled


X_train = []
y_train = []


for i in range(60, data_training.shape[0]):
    X_train.append(data_training_scaled[i-60: i])
    y_train.append(data_training_scaled[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape, y_train.shape


regressor = Sequential()    #Sequential   because we are predicting the stock market price.....

regressor.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 12)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units = 1))


regressor.summary()


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


regressor.fit(X_train, y_train, epochs=50, batch_size = 64)


# # Testing Started

past_60 = data_training.tail(60)

dt = pd.concat([past_60, data_testing], ignore_index=True)
dt


inputs = scalar.fit_transform(dt)
print(inputs.shape)
inputs


X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape


y_pred = regressor.predict(X_test)


scale = 1/scalar.scale_[0]# Reshape y_pred and y_test to match the shape expected by inverse_transform
y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Inverse transform to get the actual stock prices
y_pred_actual = scalar.inverse_transform(np.concatenate((y_pred, inputs[60:, 1:]), axis=1))[:, 0]
y_test_actual = scalar.inverse_transform(np.concatenate((y_test, inputs[60:, 1:]), axis=1))[:, 0]





# # Visualization of All the Codes and Work Done Above

# Visualising the results
plt.figure(figsize=(30,15))
plt.plot(y_test_actual, color = 'red', label = 'Real Price of the stock')
plt.plot(y_pred_actual, color = 'blue', label = 'Predicted Predicted Price of the Stock')
plt.title('Nepse Stock Perediction of given company')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()




