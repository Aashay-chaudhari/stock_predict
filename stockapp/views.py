
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from .models import Stock

import pandas_datareader as web
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
# Create your views here.

def watchlist(request):
    stock_list = Stock.objects.all()
    message = 'In home'
    return render(request, 'watchlist.html', {'message': message, 'stock_list': stock_list})

def chart(request):
    return render(request, 'charts.html')

def home(request):
    df = web.DataReader('^NSEI', data_source='yahoo', start='2005-01-3', end='2020-05-25')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    dataframe = df.reset_index()
    test_dates = df.filter(['Date'])
    test_dates = test_dates[-10:]
    print(dataframe)
    data = df.filter(['Close'])
    dataset1 = data.values
    # type(dataset), type(data)
    training_data_len1 = math.ceil(len(dataset1) * .8)
    dataset1.shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data1 = scaler.fit_transform(dataset1)
    scaled_data1.shape
    train_data1 = scaled_data1[0:training_data_len1, :]
    x_train1 = []
    y_train1 = []
    for i in range(60, len(train_data1)):
        x_train1.append(train_data1[i - 60:i, 0])
        y_train1.append(train_data1[i, 0])
    x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
    x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
    print(x_train1.shape, y_train1.shape)
    model1 = Sequential()
    model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train1.shape[1], 1)))
    model1.add(LSTM(units=50, return_sequences=False))
    model1.add(Dense(units=25))
    model1.add(Dense(units=1))

    model1.compile(optimizer='adam', loss='mean_squared_error')
    ## epochs = 9
    model1.fit(x_train1, y_train1, batch_size=1, epochs=1)
    df_for_testing1 = dataframe[training_data_len1:]
    df_for_testing1.shape
    df_for_testing_scaled1 = scaled_data1[training_data_len1:]
    df_for_testing_scaled1.shape
    xtest1 = []
    ytest1 = []

    for i in range(60, len(df_for_testing1)):
        xtest1.append(df_for_testing_scaled1[i - 60:i, 0])
    xtest1, ytest1 = np.array(xtest1), np.array(ytest1)

    xtest1 = np.reshape(xtest1, (xtest1.shape[0], xtest1.shape[1], 1))
    xtest1.shape

    prediction_list1 = []

    for i in xtest1:
        i = np.reshape(i, (1, 60, 1))
        price1 = model1.predict(i)
        prediction_list1.append(price1)

    prediction_list1 = np.array(prediction_list1)
    print("shape of prediction_list : ", prediction_list1.shape)
    prediction_list1 = np.reshape(prediction_list1, (prediction_list1.shape[0], prediction_list1.shape[1]))
    prediction_list1 = scaler.inverse_transform(prediction_list1)
    print("shape of prediction_list after reshape: ",prediction_list1.shape)
    actual_values = dataframe.filter(['Close'])
    actual_values = actual_values.values
    actual_values = actual_values[-558:]
    buffer = []
    for i in actual_values:
        buffer.append(str(i)[1:-1])
    actual_values = buffer
    actual_values_30 = actual_values[-30:]
    predicted_values = prediction_list1[:]
    buffer1 = []
    for i in predicted_values:
        buffer1.append(str(i)[1:-1])
    predicted_values = buffer1
    predicted_values_30 = predicted_values[-30:]
    a_list = list(range(1, 588))
    a_list_30 = list(range(1,30))
    message = 'In home message'
    print("Type of prediction_list1 is", type(prediction_list1))
    print("Type of actual_values is", type(actual_values))
    # stock_list = Stock.objects.all()
    stock_list = ['RELIANCE.NS']

    for stock in stock_list:
        stock_data = web.DataReader(stock, data_source='yahoo', start='2020-01-3', end='2021-05-10')
        print("Stock data is: ", stock_data)
        column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
        stock_data = stock_data.reindex(columns=column_names)
        last_60_days = stock_data.filter(['Close'])
        last_60_days = last_60_days[-60:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(last_60_days)
        x_train = np.array(scaled_data)
        x_train = np.reshape(x_train, (1, 60, 1))
        print("xtrain is: ", x_train)
        price_array = []
        price = model1.predict(x_train)
        price_array.append(price)
        price_array = np.array(price_array)
        print("shape of prediction_list : ", prediction_list1.shape)
        prediction_list1 = np.reshape(price_array, (1, 1))
        prediction_list1 = scaler.inverse_transform(prediction_list1)
        print("Price is: ", prediction_list1)
        stock_list = Stock.objects.all()
        print("stock_list is: ", stock_list)
    return render(request, 'home.html', { 'actual_values': actual_values,'predicted_values': predicted_values,'a_list': a_list,
                                          'actual_values_30': actual_values_30,'predicted_values_30': predicted_values_30,'a_list_30': a_list_30,
                                          'stock_list': stock_list})
