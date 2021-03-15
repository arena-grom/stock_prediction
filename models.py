#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:51:14 2021
@author: arena

This module contains the model classes and functions that apply machine learning 
algorithms to input time-series data. 
"""

# Importing necessary modules and libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.models import Sequential
from pmdarima import auto_arima
import numpy as np
import visualizations
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Algorithms for Stock Analysis

def moving_average(df, n, feature='Close'):
    """ Calculating moving average and adding as a column """

    colname = 'MA' + str(n)
    df[colname] = df[feature].rolling(n).mean()


class StockModel():
    """ Parent class for setting up a model on Stock time series data. """

    def __init__(self, X_train, X_test, y_train, y_test):

        # initialized model attributes
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # model attributes to be set
        self.model = None
        self.trained = False
        self.predictions = None
        self.resididuals = None
        self.mse = None
        self.mae = None
        self.r2 = None

    def evaluate_model(self, model_name):

        if self.trained and self.predictions is not None:

            # show residuals
            self.residuals = self.y_test - self.predictions
            title = model_name + ' Residuals'
            visualizations.plot_kde(self.residuals, title=title)

            # calculate MSE
            self.mse = mean_squared_error(self.y_test, self.predictions)
            print(f'\nMSE Score: {self.mse}')

            # calculate MAE
            self.mae = mean_absolute_error(self.y_test, self.predictions)
            print(f'\nMAE Score: {self.mae}')

            # calculate R2
            self.r2 = r2_score(self.y_test, self.predictions)
            print(f'\nR2: {self.r2}')

        else:
            warnings.warn('Error evaluating model. No predictions available!')


class StockARIMA(StockModel):
    """ Statistical ARIMA model for Stock forecasting """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        print('\nInitializing ARIMA.')

    def train(self):
        print('\nTraining Auto ARIMA model to find P, Q, D...')
        self.model = auto_arima(self.y_train, trace=True,
                                error_action="ignore", suppress_warnings=True)
        self.trained = True

    def forecast(self):
        print('\nForecasting with ARIMA...')
        if self.trained:
            self.predictions = []
            for X in self.X_test:
                self.model.fit(X)
                self.predictions.append(self.model.predict(n_periods=1)[0])
        else:
            warnings.warn('Error forecasting. Model must be trained!')


class StockLinReg(StockModel):
    """ Linear Regression model for Stock prediction """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        print('\nInitializing Linear Regression.\n')
        self.model = LinearRegression()

    def train(self):
        print('\nTraining Linear Regression model...')
        self.model.fit(self.X_train, self.y_train)
        self.trained = True

    def predict(self):
        print('\nPredicting with Linear Regression model...')
        if self.trained:
            self.predictions = self.model.predict(self.X_test)
        else:
            warnings.warn('Error predicting. Model must be trained!')


class StockLSTM(StockModel):
    """ LSTM Neural Network for Stock prediction """

    def __init__(self, X_train, X_test, y_train, y_test, input_nodes=92):
        super().__init__(X_train, X_test, y_train, y_test)
        print('\nInitializing LSTM network.\n')
        self.history = None
        self.compiled = False
        self.model = Sequential()
        self.model.add(LSTM(units=input_nodes, input_shape=(
            self.X_train.shape[1], 1), return_sequences=True))

    def add_lstm_layer(self, nodes=92, return_seq=False):
        self.model.add(LSTM(units=nodes, return_sequences=return_seq))

    def add_dense_layer(self, nodes=1):
        self.model.add(Flatten())
        self.model.add(Dense(units=nodes))

    def add_dropout(self, dropout_rate=0.2):
        self.model.add(Dropout(dropout_rate))

    def add_compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
        self.compiled = True
        print(self.model.summary())

    def train(self, epochs=100, batch_size=100, verbose=1, plot=True):
        if self.compiled:
            print('\nTraining LSTM network...')
            self.history = self.model.fit(self.X_train, self.y_train,
                                          epochs=epochs, batch_size=batch_size,
                                          verbose=verbose)
            self.trained = True
            if plot:
                visualizations.plot_loss(values=self.history.history['loss'],
                                         title='LSTM Loss over epochs',
                                         y='Loss', x='Epochs')
        else:
            warnings.warn('Error training. Model must be compiled!')

    def predict(self):
        print('\nPredicting with LSTM model...')
        if self.trained:
            predictions = self.model.predict(self.X_test)
            self.predictions = [pred[0] for pred in predictions]
            self.predictions = np.reshape(self.predictions, -1)
        else:
            warnings.warn('Error predicting. Model must be trained!')
