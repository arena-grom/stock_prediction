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
import visualizations

# Importing necessary libraries
import numpy as np
import pandas as pd


# Functions for data handling

def read_stock_data(filepath='NVDA.csv'):
    """ Read CSV with stock data """
    
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Date'], dayfirst=True) 
    print(f'Dataset read with shape: {df.shape}')
    print(f'\nHead:\n{df.head()}')
    print(f'\nSummary:\n{df.describe()}')
    print(f'\nNaNs in the data: {df.isnull().sum()}')

    return df


def extract_dates(df, start='2020-3-17', end='2021-2-22'):
    """ Retrieve specific time interval from data """
    
    df_extract = df[df.Datetime.between(start, end)]
    return df_extract.reset_index(drop=True)



  
    
#%% Import and MA calculations

# read data
stock_df = read_stock_data()
visualizations.plot_linear(stock_df, 'NVIDIA all time Stock Closing Prices')

# calculate new features
for n in [5, 20, 100]:
    models.moving_average(stock_df, n)


#%% Zoom into one year
year_df = extract_dates(stock_df, start='2020-1-1', end='2021-1-1')

line1 = year_df[year_df['Date']=='17-03-2020'].index.values[0]
line2 = year_df[year_df['Date']=='02-09-2020'].index.values[0]
visualizations.plot_linear(year_df, 'NVIDIA Stock Closing Price in 2020',
                           vlines=[line1, line2])


#%% Add trendline to selected time period
linear_df = extract_dates(stock_df, start='2020-3-17', end='2020-9-2')
visualizations.plot_linear(linear_df, 'NVIDIA stock from 2020-3-17 to 2020-9-2', 
                           trendline=True)


#%% Visualize candlestick and moving average
visualizations.plot_candlestick(linear_df, 'NVIDIA stock from 2020-3-17 to 2020-9-2',
                                MA1='MA5', MA2='MA20', MA3='MA100')


#%% Create training and test data

X_train, y_train, X_test, y_test = models.preprocess_data(linear_df, seq_len=10, test_size=24)




#%%
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    
#%% Setting up LSTM 1

lstm_model = StockLSTM(X_train, y_train, input_nodes=15) # Init and first layer
lstm_model.add_dropout() 

lstm_model.add_lstm_layer(nodes=15, return_seq=True) # Second layer
lstm_model.add_dropout() 

#lstm_model.add_lstm_layer(units=92) # last LSTM layer does not return seq
#lstm_model.add_dropout()

lstm_model.add_dense_layer() # Output layer

# Compiling
lstm_model.add_compile()

#%% Training LSTM
lstm_model.train(epochs=30, batch_size=100)
lstm_model.plot_history()

#%% Prediction and Evaluation
# predicted_y = clf.test_predict(X_test)

# evaluate_model(y=y_test_raw, pred=predicted_y)

#%%
print('\nFinished running code.')
    




# END