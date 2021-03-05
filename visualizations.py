#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:42:09 2021
@author: arena

This module contains visualization functions of time-series stock data.
"""
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from pandas.plotting import lag_plot, autocorrelation_plot

plt.style.use('seaborn-darkgrid')
pio.renderers.default='browser'


# Visualization Functions

def plot_linear(input_df, title, feature='Close', vlines=None, trendline=False):
    '''Function for creating stock data lineplot with Matplotlib '''
    
    df = input_df[['Date', feature]]
    df.set_index('Date', drop=True, inplace=True)
    
    # plot linear data
    df[feature].plot(figsize=(14,7), color='darkred', linewidth=2.5)

    # add linear fitted line
    if trendline:
        X = np.arange(0, len(df[feature]), 1)
        z = np.polyfit(X, df[feature], 1)
        p = np.poly1d(z)
        plt.plot(X, p(X), color='steelblue', linewidth=2)
    
    # add vertical lines
    if isinstance(vlines, list): 
        for xvalue in vlines:
            plt.axvline(x=xvalue, color='gray', linewidth=2, ls='--')
            
    plt.suptitle(title, fontsize=18)
    plt.xlabel('Date')
    plt.ylabel(feature + ' ($)')
    plt.show()
    
    
def plot_candlestick(df, title, MA1, MA2, MA3):
    """ Function for creating candlestick graph of stocks with Plotly """  
    
    fig = go.Figure(data=[go.Candlestick(x=df['Date'], 
                                         open=df['Open'], 
                                         high=df['High'], 
                                         low=df['Low'], 
                                         close=df['Close'], name='Bull/Bear'),
                          go.Scatter(x=df['Date'], y=df[MA1], name=MA1,
                                     line=dict(color='darkred', width=1)),
                          go.Scatter(x=df['Date'], y=df[MA2], name=MA2,
                                     line=dict(color='darkorange', width=1)),
                          go.Scatter(x=df['Date'], y=df[MA3], name=MA3,
                                     line=dict(color='tan', width=1)),
                          ])
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=title,
                      yaxis_title='Price ($)', xaxis_title='Date')
    fig.show()
    
    
def plot_training(values, title, y, x):
    """ Plotting training features"""
    plt.figure(figsize=(14,7))
    plt.plot(values)
    plt.suptitle(title, fontsize=18)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.show()
    

def plot_predictions(df_train, df_test, legend):
    """ Visualize stock training data, the test data and different predictions """

    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(df_train, color='darkred')
    ax.plot(df_test)
    [ax.set_xticks(ax.get_xticks()[::2]) for i in range(4)] # sparse out xticks
        
    plt.title('Model Predictions', fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend(legend, loc='lower right')
    plt.show()



# END