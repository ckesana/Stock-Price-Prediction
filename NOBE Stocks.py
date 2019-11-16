# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

import pandas as pd


import pandas_datareader.data as web
import matplotlib.pyplot as plt

def predictLinear(ticker, start_date, days_in_future):

    
    end = datetime.now()
    
    # Retrieves stock data using Pandas DataReader
    df = web.DataReader(ticker, "yahoo", start_date, end)
    
    df.to_csv(ticker + "_history.csv")
    
    #Retrieve close values of the stock
    close_vals = df['Close'].values
    
    #Make a list of numbers that correspond to a date
    #i.e 0 - > 1/1/2017, 1 -> 1/2/2017, ...
    dates = np.arange(len(df))
    
    
    plt.plot(dates, close_vals)
    
    #Generate matrix to feed into Linear Regression algorithm
    A = np.zeros((len(dates), 2))
    
    #First column is a vector of ones
    A[:,0] = np.ones(len(dates))
    #Second column is out dates (x - values)
    A[:,1] = dates
    
    #Generate Linear Regression Model
    model = LinearRegression().fit(A,close_vals)
    coeffs = model.coef_

    #Graphing stuff
    a = np.linspace(1,len(df),1000)
    b = model.intercept_ + coeffs[1]*a
    
    plt.title('Linear Regression Model for ' + ticker + ' starting at ' + start_date.strftime('%m-%d-%Y'))
    plt.plot(dates,close_vals, color='b')
    plt.plot(a,b, color='r')
    plt.show()
    
    #Compute prediction using computed coefficients
    #y = b + ax
    #x is the number of days in the future + the number of dates we have used - 1
    #b is the intercept
    #a is coeffs[1]
    #y is the prediction
    prediction = model.intercept_ + coeffs[1] * (len(dates) + days_in_future - 1)
    

    
    return prediction; 


tickers = input("Enter a list of tickers separated by commas: ")
ticker_array = tickers.split(', ')


    

    
start_date = input("Enter the start date (MM-DD-YYYY): ")
start_date_strip = datetime.strptime(start_date, '%m-%d-%Y')

days_in_future = int(input("Enter days in future: "))

for ticker in ticker_array:

    prediction = predictLinear(ticker, start_date_strip, days_in_future)

    print(ticker + " price in " + str(days_in_future) + " days will be $" 
          + str(round(prediction, 2)) + " according to this model")
    
    






    










