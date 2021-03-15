#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:42:09 2021
@author: arena

This module contains visualization functions of time-series stock data.
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('ggplot')
pio.renderers.default = 'browser'


# Visualization Functions

def plot_stock(input_df, title, feature='Close', vlines=None, trendline=False):
    '''Function for creating stock data lineplot with Matplotlib '''
    df = input_df[['Date', feature]]
    df.set_index('Date', drop=True, inplace=True)

    # plot linear data
    df[feature].plot(figsize=(12, 6), linewidth=2.5)

    # add linear fitted line
    if trendline:
        X = np.arange(0, len(df[feature]), 1)
        z = np.polyfit(X, df[feature], 1)
        p = np.poly1d(z)
        plt.plot(X, p(X), color='k', linewidth=2, ls='dotted')

    # add vertical lines
    if isinstance(vlines, list):
        for xvalue in vlines:
            plt.axvline(x=xvalue, color='gray', linewidth=2, ls='--')

    plt.suptitle(title, fontsize=18)
    plt.xlabel('Date')
    plt.ylabel(feature + '  ($)')
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


def plot_decomposition(values, title='Time Series Decomposition'):
    """ Plotting time series model decomposition """
    result = seasonal_decompose(values, model='additive', freq=30)
    result.plot()
    plt.show()


def plot_kde(values, title='Density plot'):
    """ Kernel Density Estimate of a set of values """
    pd.Series(values).plot(kind='density', title=title, color='gray', linewidth=2.5)
    plt.show()


def plot_loss(values, title, y, x):
    """ Plotting loss over epochs """
    plt.figure(figsize=(6, 3.5))
    plt.plot(values, linewidth=2.5, color='gray')
    plt.suptitle(title, fontsize=16)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.show()


def plot_final_predictions(df):
    """ Visualize stock training data, the test data and different predictions """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df, linewidth=3, alpha=0.8)
    [ax.set_xticks(ax.get_xticks()[::2]) for i in range(2)]  # sparse out xticks

    plt.title('Model Predictions', fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend(df.columns, loc='lower right')
    plt.show()
