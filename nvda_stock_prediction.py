#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:40:18 2021
@author: arena

This is the Main module that reads necessary data, preprocesses, builds models
and makes predictions through imported modules. The results are visualized and evaluated.
"""

# Importing modules and libraries
import models
import data_handling
import visualizations
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# %% ----------------------- Data Exploration section -------------------------- #

# read and clean data
stock_data = data_handling.read_data(filepath='NVDA.csv')
visualizations.plot_stock(stock_data, 'NVIDIA all time Stock Closing Prices')

# calculate Moving Average
for n in [5, 20, 100]:
    models.moving_average(stock_data, n)

# zoom into one year
year_data = data_handling.extract_dates(stock_data, start='2020-1-1', end='2021-1-1')

line1 = year_data[year_data['Date'] == '17-03-2020'].index.values[0]
line2 = year_data[year_data['Date'] == '02-09-2020'].index.values[0]
visualizations.plot_stock(year_data, 'NVIDIA Stock Closing Price in 2020',
                          vlines=[line1, line2])

# zoom into selected time period
linear_data = data_handling.extract_dates(stock_data, start='2020-3-17', end='2020-9-2')
visualizations.plot_stock(linear_data, 'NVIDIA stock from 2020-3-17 to 2020-9-2',
                          trendline=True)

# plot seasonal decomposition
visualizations.plot_decomposition(
    linear_data.Close, title="Stock Closing Price Decomposition")

# visualize candlestick and MA
visualizations.plot_candlestick(linear_data, 'NVIDIA stock from 2020-3-17 to 2020-9-2',
                                MA1='MA5', MA2='MA20', MA3='MA100')

# %% ------------------------  Training section ------------------------------ #

# split to training and test data
TEST_N = 24
X_train, y_train, X_test, y_test, scaler = data_handling.preprocess_data(
    linear_data, seq_len=10, test_size=TEST_N)

# ARIMA
arima = models.StockARIMA(X_train, X_test, y_train, y_test)
arima.train()
arima.forecast()
arima.evaluate_model(model_name='ARIMA')

# Linear Regression
linreg = models.StockLinReg(X_train, X_test, y_train, y_test)
linreg.train()
linreg.predict()
linreg.evaluate_model(model_name='Linear Regression')

# LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# setting up architecture
lstm = models.StockLSTM(X_train_lstm, X_test_lstm, y_train, y_test, input_nodes=15)
lstm.add_dropout()
lstm.add_lstm_layer(nodes=15, return_seq=True)
lstm.add_dropout()
lstm.add_lstm_layer(nodes=15, return_seq=True)
lstm.add_dropout()
lstm.add_dense_layer(nodes=1)
lstm.add_compile()

# train and evaluate
lstm.train(epochs=15, batch_size=10)
lstm.predict()
lstm.evaluate_model(model_name='LSTM')


# %% ------------------------ Final Visualization ---------------------------#
# extract training and test values
close_true = linear_data[['Date', 'Close']].set_index('Date')
close_pred = close_true[-TEST_N:].copy()
y = np.reshape(y_test, (y_test.shape[0], 1))
close_pred['Test'] = data_handling.revert_scale(scaler, y).astype(float)

# add LSTM
lstm_preds = np.reshape(np.array(lstm.predictions),
                        (np.array(lstm.predictions).shape[0], 1))
close_pred['LSTM'] = data_handling.revert_scale(scaler, lstm_preds).astype(float)

# add Linear Regression
linreg_preds = np.reshape(linreg.predictions, (linreg.predictions.shape[0], 1))
close_pred['LinReg'] = data_handling.revert_scale(scaler, linreg_preds)

# add ARIMA
arima_preds = np.reshape(np.array(arima.predictions),
                         (np.array(arima.predictions).shape[0], 1))
close_pred['ARIMA'] = data_handling.revert_scale(scaler, arima_preds)


# show predictions
visualizations.plot_final_predictions(close_pred)
