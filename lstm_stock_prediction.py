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
#from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM #, SimpleRNN

#todays_date = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")


#%% Functions

def plot_stock_data(df, title, feature='Close' , date_formatter='%Y', color='darkred'):
    '''Function for plotting stock data'''
    
    df[feature].plot(figsize=(14,7), color=color)
    plt.suptitle(title, fontsize=18)
    # plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.grid()
    plt.show()
    
    
def preprocess_data(data, feature='Close', seq_len=60, test_n=600):
    """ Function that transforms raw data to LSTM input data """
    
    df = data[[feature]]
    print(f'Input data shape: {df.shape}\nExtracting test set.')
    # df.reset_index(drop=True, inplace=True)
    #df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
    train_data = df.iloc[:-test_n,:]
    test_data = df.iloc[-test_n:,:]
    
    # scale training dataset
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    print(f'\nScaled training data:\n{train_scaled}\nShape: {train_scaled.shape}')
    
    # create sequences of training and predictors
    length=len(train_data)
    X_train, y_train = [], []
    for i in range(seq_len, length):
        X_train.append(train_scaled[i-seq_len:i, 0])
        y_train.append(train_scaled[i, 0])

    # reshape for LSTM
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(f'X_train:\n{X_train}\nShape: {X_train.shape}')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(f'\nNew shape of training data: X_train {X_train.shape}, y_train {y_train.shape}')
    
    # TODO reshape test_data
    
    return X_train, y_train, test_data
    

def plot_loss_over_epochs():
    pass

def plot_future_prediction():
    pass

def evaluate_model():
    pass


#%% Class LSTM architecture

class StockLSTM():
    """ Class that sets up a LSTM network for Stock price prediction """
    
    def __init__(self, X_train, y_train, input_units=92):
        print('\nInitializing LSTM network.\n')
        self.X = X_train
        self.y = y_train
        self.trained=None
        self.compliled=False
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
        self.compiled=True
        print(self.model.summary())
        
    """ Training the LSTM model on X_train """
    def train(self, epochs=100, batch_size=100, validation_rate=0.3, verbose=1):
        if self.compiled:
            print('\nTraining LSTM model on training data.')
            self.trained = self.model.fit(self.X, self.y, 
                                          validation_split=validation_rate, 
                                          epochs=epochs, batch_size=batch_size,
                                          verbose=verbose)
        else: print('Error: Model not compiled!') 
        
    """ Plotting training """
    def plot_history(self):
        if self.trained != None:
            fig = plt.figure(figsize=(14,7))
            plt.plot(self.trained.history['loss'])
            plt.plot(self.trained.history['val_loss'])
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



#%% Main read data


nvidia_df = pd.read_csv('NVDA.csv')
nvidia_df.set_index('Date', drop=False, inplace=True)
print('Data read.')
print(nvidia_df.shape)
print(f'\nDataset:\n{nvidia_df.head()}')
print(f'\nSummary:\n{nvidia_df.describe()}')

print(f'\nNaNs in the data: {nvidia_df.isnull().sum()}')

print('# ------------------------------- #')
#%% Plot

plot_stock_data(nvidia_df, title='NVIDIA stock price all time') # date_formatter : ['%Y-%m-%d']
plot_stock_data(nvidia_df.tail(1000), title='NVIDIA stock price 2017-2021', date_formatter='%Y-%m')

print('# ------------------------------- #')
#%% Process the training data
X_train, y_train, test = preprocess_data(nvidia_df)

print('# ------------------------------- #')
#%% Setting up LSTM 1

lstm_model = StockLSTM(X_train, y_train) # Init and first layer
lstm_model.add_dropout() 

lstm_model.add_lstm_layer(return_seq=True) # Second layer
lstm_model.add_dropout() 

#lstm_model.add_lstm_layer(units=92) # last LSTM layer does not return seq
#lstm_model.add_dropout()

lstm_model.add_dense_layer() # Output layer

# Compiling and training
lstm_model.add_compile()

#%%
lstm_model.train(epochs=10, batch_size=500)
lstm_model.plot_history()


# predicted_y = clf.test_predict(X_test)

# evaluate_model(y=y_test_raw, pred=predicted_y)


print('\nFinished running code.')


