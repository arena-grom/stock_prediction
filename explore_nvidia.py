#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:40:18 2021
@author: arena

"""
import pandas as pd
import matplotlib.pyplot as plt


def read_stock_data(filepath='NVDA.csv'):
    
    df = pd.read_csv(filepath)
    df.set_index('Date', drop=False, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']) 
    
    print(df.shape)
    print(f'Dataset:\n{df.head()}')
    print(f'\nSummary:\n{df.describe()}')
    print(f'\nNaNs in the data: {df.isnull().sum()}')

    return df

def extract_dates(df, start='2020-3-17', end='2021-2-22'):
    
    return df[df.Date.between(start, end)]


def plot_stock_data(df, title, features=['Close'], date_formatter='%Y'):
    '''Function for plotting stock data'''
    
    for feature in features:
        df[feature].plot(figsize=(14,7))
   
    plt.suptitle(title, fontsize=18)
    # plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.grid()
    plt.show()
    
    
#%% Main  

stock_df = read_stock_data()
plot_stock_data(stock_df, 'NVIDIA all time')

reduced_df = extract_dates(stock_df)
plot_stock_data(reduced_df, 'NVIDIA from 2020-3-17 to now')

reduced_df = extract_dates(reduced_df, end='2020-9-2')
plot_stock_data(reduced_df, 'NVIDIA from 2020-3-17 to 2020-9-2', features=['Close', 'Open'])



