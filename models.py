#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:51:14 2021
@author: arena

This module contains the model classes and functions that apply machine learning 
algorithms to input time-series data. 
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten #, SimpleRNN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#%% Functions

def create_sequences(dataset, seq_len):
    """ Function that creates sequences of prior datapoints as X and current datapoint as y """
    
    length = len(dataset)
    X, y = [], []
    for i in range(seq_len, length):
        X.append(dataset[i-seq_len:i, 0])
        y.append(dataset[i, 0])
        
    return X, y
    

def preprocess_data(df, feature='Close', seq_len=60, test_size=600):
    """ Function that transforms raw data to time-series training blocks """
    
    df = df[[feature]]
    print(f'\nInput data shape: {df.shape}')
    train = df.iloc[:-test_size,:]
    test = df.iloc[-test_size:,:]
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
    
    
    return X_train, y_train, X_test, y_test


def moving_average(df, n, feature='Close'):
    """ Calculating moving average and adding as a column """
    
    colname = 'MA' + str(n)
    df[colname] = df[feature].rolling(n).mean()
    
    
#%% Model Classes with methods

class MLModel():

    def __init__(self, X_train, y_train):
        print('\nInitializing MLModel.\n')
        self.X = X_train
        self.y = y_train
        self.trained=None
        self.preds=None
        self.model=None
        
        

class StockLSTM():
    """ Class that sets up a LSTM network """
    
    def __init__(self, X_train, y_train, input_nodes=92):
        print('\nInitializing LSTM network.\n')
        self.X = X_train
        self.y = y_train
        self.trained=None
        self.compliled=False
        self.preds=None
        
        # initialize Keras MLP model and first input layer
        self.model = Sequential()
        self.model.add(LSTM(units=input_nodes, input_shape=(self.X.shape[1], 1), return_sequences=True))
        
    """ Adding layers and compiling the architecture """
    def add_lstm_layer(self, nodes=92, return_seq=False):
        self.model.add(LSTM(units=nodes, return_sequences=return_seq))
       
    def add_dense_layer(self, nodes=1):
        self.model.add(Flatten())
        self.model.add(Dense(units=nodes))
        
    def add_dropout(self, dropout_rate=0.2):
        self.model.add(Dropout(dropout_rate))
        
    def add_compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
        self.compiled=True
        print(self.model.summary())
        
    
    def train(self, epochs=100, batch_size=100, verbose=1):
        """ Training the LSTM model on X_train """
        if self.compiled:
            print('\nTraining LSTM model on training data.')
            self.trained = self.model.fit(self.X, self.y, 
                                          epochs=epochs, batch_size=batch_size,
                                          verbose=verbose)
        else: print('Error: Model not compiled!') 
        
    
    def plot_history(self):
        """ Plotting training """
        if self.trained != None:
            fig = plt.figure(figsize=(14,7))
            plt.plot(self.trained.history['loss'])
            # plt.plot(self.trained.history['val_loss'])
            plt.suptitle('LSTM Loss during epochs', fontsize=18)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.show()
        
        else: print('Error: Model not trained!') 
        
        
    """ Predicting test data and evaluating the model """
    
    # def test_accuracy(self, X_val, y_val):
    #     _, accuracy = self.model.evaluate(X_val, y_val)
    #     print(f'\nAccuracy: {accuracy*100}')
        
    def test_predict(self, X):
        self.preds = self.model.predict_classes(X)
        
        return self.preds


class LinearRegression():

    def __init__(self, X_train, y_train):
        print('\nInitializing LSTM network.\n')
        self.X = X_train
        self.y = y_train
        self.model = LinearRegression()
        
    def train(self):
        self.model.fit(self.X, self.y)


    
# END