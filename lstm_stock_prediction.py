#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:51:14 2021
@author: arena
"""
import numpy as np
import pandas as pd
#import time
#from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM #, SimpleRNN

#todays_date = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")


#%% Functions

def plot_stock_data(data, title, feature='Close' , date_formatter='%Y', color='darkred'):
    '''Function for plotting stock data'''
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data[feature], color=color)
    #
    
    if date_formatter == '%Y':
        ax.xaxis.set_major_formatter(DateFormatter(date_formatter))
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    elif date_formatter == '%Y-%m':
        ax.xaxis.set_major_formatter(DateFormatter(date_formatter))
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))

    plt.suptitle(title)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.grid()
    plt.show()
    
    
def preprocess_data(data, feature='Close', seq_len=60):
    """ Function that transforms raw data to LSTM input data """
    
    
    df = data[['Date', feature]]
    df.set_index('Date', inplace=True)
    
    train_data = df.iloc[:-600,:]
    test_data = df.iloc[-600:,:]
    
    # scale training dataset
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    print(f'Scaled data:\n{train_scaled}\nShape: {train_scaled.shape}')
    
    seq_len=60
    length=len(train_data)
    X_train, y_train = [], []
    for i in range(seq_len, length):
        
        X_train.append(train_scaled[i-seq_len:i, 0])
        y_train.append(train_scaled[i, 0])

    # reshape
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(f'\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}')
    
    return X_train, y_train, test_data
    

def plot_loss_over_epochs():
    pass

def plot_future_prediction():
    pass

def evaluate_model():
    pass


#%% Building LSTM architecture

class StockLSTM():
    """ Class that sets up a LSTM network for Stock price prediction """
    
    def __init__(self, X_train, y_train, input_units=92):
        print('\nInitializing LSTM network.\n')
        self.X = X_train
        self.y = y_train
        self.train_history=None
        self.preds=None
        
        # initialize Keras MLP model and first input layer
        self.model = Sequential()
        self.model.add(LSTM(units=input_units, input_shape=(self.X.shape[1], 1), return_sequences=True))
        
    """ Adding layers and compiling the architecture """
    def add_lstm_layer(self, units=92, return_seq=False):
        self.model.add(LSTM(units=units, return_sequences=return_seq))
       
    def add_dense_layer(self, units=1):
        self.model.add(Dense(units=units))
        
    def add_dropout(self, dropout_rate=0.2):
        self.model.add(Dropout(dropout_rate))
        
    def add_compile(self, loss='mean_squared_error', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)
        print(self.model.summary())
        
    """ Training the LSTM model on X_train """
    def train(self, epochs=100, batch_size=100, validation_rate=0.3):
        print('\nTraining MLP classifier on training data.')
        self.train_history = self.model.fit(self.X, self.y, 
                                            validation_split=validation_rate, 
                                            epochs=epochs, batch_size=batch_size)
        
    """ Predicting test data and evaluating the model """
    
    # def test_accuracy(self, X_val, y_val):
    #     _, accuracy = self.model.evaluate(X_val, y_val)
    #     print(f'\nAccuracy: {accuracy*100}')
        
    def test_predict(self, X):
        self.preds = self.model.predict_classes(X)
        
        return self.preds



#%% Main read data


nvidia_df = pd.read_csv('NVDA.csv')
print('Data read.')
print(nvidia_df.shape)
print(nvidia_df.describe())

print(f'NaNs in the data: {nvidia_df.isnull().sum()}')

print('# ------------------------------- #')

plot_stock_data(data=nvidia_df, title='NVIDIA stock price all time') # date_formatter : ['%Y-%m-%d']
plot_stock_data(data=nvidia_df.tail(1000), title='NVIDIA stock price 2017-2021', date_formatter='%Y-%m')

# predict the opening price
X_train, y_train, test = preprocess_data(nvidia_df)

print('# ------------------------------- #')

#%% Setting up LSTM

lstm_model = StockLSTM(X_train, y_train, input_units=35) 
lstm_model.add_dropout() 
lstm_model.add_lstm_layer(units=35) # last LSTM layer does not return seq
lstm_model.add_dropout() 
lstm_model.add_dense_layer()
lstm_model.add_compile()

lstm_model.train(epochs=20, batch_size=100)

# training MLP classifier 
# clf.train(X_val=X_test, y_val=y_test)
# predicted_y = clf.test_predict(X_test)

# evaluate_model(y=y_test_raw, pred=predicted_y)


print('\nFinished running code.')


