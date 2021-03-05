#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:51:14 2021
@author: arena

This module contains the model classes and functions that apply machine learning 
algorithms to input time-series data. 
"""

# Importing necessary libraries
import visualizations

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression


# Algorithms for Stock Analysis

def moving_average(df, n, feature='Close'):
    """ Calculating moving average and adding as a column """
    
    colname = 'MA' + str(n)
    df[colname] = df[feature].rolling(n).mean()
    

class StockModel():
    """ Parent class for setting up a model on Stock time-series data"""
    
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train        
        self.trained=None
        self.preds=None
        self.rmse=None
        
    def predict(self, X, y):
        """ Predict new samples """
        
        if self.trained != None:
            self.preds = self.model.predict(X)
            self.rmse = np.sqrt(np.mean(((self.preds - y) ** 2)))
            print(f'\nRMSE: {np.round(self.rmse,3)}')
            
        else: print('You need to train a model first!')


class StockLinReg(StockModel):
    """ Child class that trains a Linear Regression model """
    
    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        
        print('\nInitializing Linear Regression.\n')
        self.model = LinearRegression()
      
    def train(self):
        print('\nTraining Linear Regression model on training data.')
        self.model.fit(self.X, self.y)
        self.trained=True


class StockLSTM(StockModel):
    """ Child class that sets up and trains a LSTM network"""
    
    def __init__(self, X_train, y_train, input_nodes=92):
        super().__init__(X_train, y_train)
        
        print('\nInitializing LSTM network.\n')
        # initialize Keras MLP model and first input layer
        self.compliled=False
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
        
    def train(self, epochs=100, batch_size=100, verbose=1, plot=True):
        if self.compiled:
            print('\nTraining LSTM model on training data.')
            self.trained = self.model.fit(self.X, self.y, 
                                          epochs=epochs, batch_size=batch_size,
                                          verbose=verbose)
            
            if plot:
                visualizations.plot_training(values=self.trained.history['loss'],
                                             title='LSTM Loss during epochs',
                                             y='Loss', x='Epochs')
            
        else: print('Error: LSTM not compiled!') 
               
    
# END