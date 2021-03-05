#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:53:40 2021
@author: arena

This module contains functions for reading and preprocessing time-series stock data.
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Functions for data handling

def read_data(filepath):
    """ Read CSV with historical stock data """
    
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


def create_sequences(dataset, seq_len):
    """ Create sequences of prior datapoints as X, and current datapoint as y """
    
    length = len(dataset)
    X, y = [], []
    for i in range(seq_len, length):
        X.append(dataset[i-seq_len:i, 0])
        y.append(dataset[i, 0])
        
    return X, y


def revert_scale(scaler, values):
    
    return scaler.inverse_transform(values)
    

def preprocess_data(df, feature='Close', seq_len=60, test_size=600):
    """ Function that transforms raw data to time-series training blocks """
    
    df = df[[feature]]
    print(f'\nInput data shape: {df.shape}')
    train = df.iloc[:-test_size,:]
    test = df.iloc[-(test_size+seq_len):,:]
    print(f'\nSplit Train {train.shape}, Test {test.shape}')
    
    # scale dataset
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # create sequences for training
    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)

    # reshape
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    print(f'\nNew Shape Train: X_train {X_train.shape}, y_train {y_train.shape}')
    print(f'New Shape Test: X_test {X_test.shape}, y_test {y_test.shape}')
    
    
    return X_train, y_train, X_test, y_test, scaler



# END