#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:40:18 2021
@author: arena

This is the Main module that reads necessary data, preprocesses, builds models
and makes predictions through imported modules. The results are visualized and evaluated.
"""

# Importing modules
import models
import data_handling
import visualizations

# Importing necessary libraries
import numpy as np
import pandas as pd


if __name__ == '__main__':
    
    
    # read data
    stock_df = data_handling.read_data(filepath='NVDA.csv') # read NVDA.csv
    visualizations.plot_linear(stock_df, 'NVIDIA all time Stock Closing Prices')
    
    # calculate Moving Average
    for n in [5, 20, 100]:
        models.moving_average(stock_df, n)

    # zoom into one year
    year_df = data_handling.extract_dates(stock_df, start='2020-1-1', end='2021-1-1')
    
    line1 = year_df[year_df['Date']=='17-03-2020'].index.values[0]
    line2 = year_df[year_df['Date']=='02-09-2020'].index.values[0]
    visualizations.plot_linear(year_df, 'NVIDIA Stock Closing Price in 2020',
                               vlines=[line1, line2])


    # zoom into selected time period
    linear_df = data_handling.extract_dates(stock_df, start='2020-3-17', end='2020-9-2')
    visualizations.plot_linear(linear_df, 'NVIDIA stock from 2020-3-17 to 2020-9-2', 
                               trendline=True)
    
    # visualize candlestick and MA
    visualizations.plot_candlestick(linear_df, 'NVIDIA stock from 2020-3-17 to 2020-9-2',
                                    MA1='MA5', MA2='MA20', MA3='MA100')
    
    # split training and test data
    TEST_N = 24
    X_train, y_train, X_test, y_test, scaler = data_handling.preprocess_data(linear_df, seq_len=10, test_size=TEST_N)
    
    # extract training and test values
    close = linear_df[['Date', 'Close']].set_index('Date')
    close_train = close[:-TEST_N].copy()
    close_test = close[-TEST_N:].copy()


    # --------------------- Linear Regression --------------------- #
    # train linred model
    linreg = models.StockLinReg(X_train, y_train)
    linreg.train()
    
    # predict and evaluate
    linreg.predict(X_test, y_test)
    
    # adding Linear Regression predictions
    linreg_preds = np.reshape(linreg.preds, (linreg.preds.shape[0], 1))
    close_test['LinReg'] = data_handling.revert_scale(scaler, linreg_preds)


    # -------------------------- LSTM ------------------------------ #
    # reshape for LSTM input
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # setting up architecture
    lstm = models.StockLSTM(X_train_lstm, y_train, input_nodes=15) # Init and first layer
    lstm.add_dropout() 
    lstm.add_lstm_layer(nodes=15, return_seq=True) # Second layer
    lstm.add_dropout() 
    lstm.add_lstm_layer(nodes=15, return_seq=True) # Third layer
    lstm.add_dropout() 
    lstm.add_dense_layer(nodes=1) # Output layer
    
    # compiling
    lstm.add_compile()
    
    # train and evaluate
    lstm.train(epochs=40, batch_size=10)
    lstm.predict(X_test_lstm, y_test)
    
    # adding LSTM predictions
    close_test['LSTM'] = data_handling.revert_scale(scaler, lstm.preds).astype(float)


    # ----------------------- Final result ------------------------- #
    
    # plot predictions
    visualizations.plot_predictions(close_train, close_test, legend=['Train', 'Test', 'LinReg', 'LSTM'])
    
    print('\nFinished running code.')
    


# END