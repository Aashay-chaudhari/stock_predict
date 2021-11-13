# Create your views here.
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
import pandas_datareader as web
from .models import Stock

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
# import seaborn as sns

from .models import Stock
##LSTM 2 IS MULTIVARIATE INPUT
##LSTM 1 IS SINGLE INPUT BASED PREDICTION


def calculatecalls(request):
    # #Data fir LSTM1
    #
    # n_past = 60
    # df1 = web.DataReader('HINDUNILVR.NS', data_source='yahoo', start='2001-10-1', end='2021-05-7')
    # dataframe1 = df1.reset_index()
    #
    # data1 = df1.filter(['Close'])
    # # .values function converts dataframe to np array
    #
    # dataset1 = data1.values
    # training_data_len1 = math.ceil(len(dataset1) * .8)
    # dataset1.shape
    # scaler1 = MinMaxScaler(feature_range=(0, 1))
    # scaled_data1 = scaler1.fit_transform(dataset1)
    # scaled_data1.shape
    # train_data1 = scaled_data1[0:training_data_len1, :]
    # x_train1 = []
    # y_train1 = []
    # for i in range(60, len(train_data1)):
    #     x_train1.append(train_data1[i - 60:i, 0])
    #     y_train1.append(train_data1[i, 0])
    #
    # x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
    # x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
    # x_train1.shape, y_train1.shape
    #
    # #Compile First LSTM1(Single Input)
    # model1 = Sequential()
    # model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train1.shape[1], 1)))
    # model1.add(LSTM(units=50, return_sequences=False))
    # model1.add(Dense(units=25))
    # model1.add(Dense(units=1))
    #
    # model1.compile(optimizer='adam', loss='mean_squared_error')
    # ## epochs = 9, batch_size = 10
    # model1.fit(x_train1, y_train1, batch_size=50, epochs=1)
    #
    # df_for_testing1 = dataframe1[training_data_len1:]
    # df_for_testing_scaled1 = scaled_data1[training_data_len1:]
    # xtest1 = []
    # ytest1 = []
    #
    # for i in range(60, len(df_for_testing1)):
    #     xtest1.append(df_for_testing_scaled1[i - 60:i, 0])
    # xtest1, ytest1 = np.array(xtest1), np.array(ytest1)
    #
    # xtest1 = np.reshape(xtest1, (xtest1.shape[0], xtest1.shape[1], 1))
    # prediction_list1 = []
    #
    # for i in xtest1:
    #     i = np.reshape(i, (1, 60, 1))
    #     price1 = model1.predict(i)
    #     prediction_list1.append(price1)
    # prediction_list1 = np.array(prediction_list1)
    # prediction_list1 = np.reshape(prediction_list1, (prediction_list1.shape[0], prediction_list1.shape[1]))
    # prediction_list1 = scaler1.inverse_transform(prediction_list1)
    # test_close_prices1 = df_for_testing1.filter(["Close"])
    #
    # test_close_prices1
    # test_dates1 = df_for_testing1.filter(["Date"])
    #
    # prediction_list1 = pd.DataFrame(prediction_list1)
    #
    # print("Predict List of lstm1: ")
    #
    #
    #
    # print(prediction_list1)
    # diff1 = []
    # sum1 = 0
    #
    # for i in range(0, len(prediction_list1)):
    #     value1 = abs(prediction_list1.iloc[i][0] - test_close_prices1.iloc[i]['Close'])
    #     diff1.append(value1)
    #     sum1 = sum1 + value1
    # average_error1 = sum1 / len(prediction_list1)
    # print("Average error in lstm1(Single Input is: "+ str(average_error1))
    # stock_list = Stock.objects.all()
    #
    # prediction_list1 = prediction_list1.astype(int)
    # test_close_prices1 = test_close_prices1.astype(int)
    # # test_dates = df_for_testing.filter(["Date"])
    # # test_dates = test_dates[60:]
    # test_dates1 = test_dates1.to_numpy()
    # buffer = []
    # for i in test_dates1:
    #     buffer.append(str(i)[2:12])
    #
    # test_dates1 = buffer
    #
    # test_close_prices1 = test_close_prices1.to_numpy()
    # buffer = []
    # for i in test_close_prices1:
    #     buffer.append(str(i)[1:-1])
    #
    # test_close_prices1 = buffer
    #
    # prediction_list1 = prediction_list1.to_numpy()
    # buffer = []
    # for i in prediction_list1:
    #     buffer.append(str(i)[1:-1])
    #
    # prediction_list1 = buffer
    #
    # test_close_prices1_180 = test_close_prices1[-240:]
    # test_dates1_180 = test_dates1[-240:]
    # prediction_list1_180 = prediction_list1[-240:]
    #
    # test_close_prices1_60 = test_close_prices1[-30:]
    # test_dates1_60 = test_dates1[-30:]
    # prediction_list1_60 = prediction_list1[-30:]
    # # test_close_prices1_180 = test_close_prices1_180.to_numpy()
    # # test_dates1_180 = test_dates1_180.to_numpy()
    # # prediction_list1_180 = prediction_list1_180.to_numpy()
    # # test_close_prices1_180 = test_close_prices1_180.tolist()
    # # test_dates1_180 = test_dates1_180.tolist()
    # # prediction_list1_180 = prediction_list1_180.tolist()
    #
    #
    # for stock in stock_list:
    #         stock_data1 = web.DataReader(stock.symbol, data_source='yahoo', start='2020-01-3', end='2021-05-10')
    #         column_names1 = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    #         stock_data1 = stock_data1.reindex(columns=column_names1)
    #
    #         # stock_data = stock_data.reset_index()
    #         cols1 = list(stock_data1)[0:5]
    #
    #         stock_data1 = stock_data1[cols1].astype(float)
    #         scaler1 = MinMaxScaler(feature_range=(0, 1))
    #         scaled_stock_data1 = scaler1.fit_transform(stock_data1)
    #
    #         scaled_stock_data1 = pd.DataFrame(scaled_stock_data1)
    #         stock_dataset1 = scaled_stock_data1.iloc[-60:, 0]
    #
    #         stock_dataset1 = np.array(stock_dataset1)
    #         stock_dataset1 = np.reshape(stock_dataset1, (1, stock_dataset1.shape[0], 1))
    #
    #         test_value1 = model1.predict(stock_dataset1)
    #
    #         test_value1 = np.repeat(test_value1, stock_data1.shape[1], axis=-1)
    #
    #         test_value1 = scaler1.inverse_transform(test_value1)
    #         test_value1 = np.average(test_value1)
    #
    #         check_close1 = stock_data1.iloc[-1:, 0:5]
    #         check_close1 = check_close1.filter(['Close'])
    #         check_close1 = np.array(check_close1)
    #         diff1 = test_value1 - check_close1
    #
    #         print("Predicted value of tomorrow is: "+str(test_value1))
    #         print("Close Price of yesterday is: "+str(check_close1))
    #         print("Difference or scope is: "+ str(diff1))
    #
    #         if diff1 > 0:
    #             stock.lstm1 = "BUY"
    #             stock.save()
    #             print("In buy of stock: " + stock.symbol)
    #         else:
    #             stock.lstm1 = "SELL"
    #             stock.save()
    #             print("In sell of stock: " + stock.symbol)

    # df = web.DataReader('HINDUNILVR.NS', data_source='yahoo', start='2001-10-1', end='2021-05-7')
    # cols = list(df)[1:6]
    # df_combined = df[cols].astype(float)
    # training_data_len = math.ceil(len(df_combined) * .8)
    # df_for_training = df_combined.iloc[:training_data_len, :]
    # df_for_testing = df_combined.iloc[training_data_len:, :]
    # scaler = StandardScaler()
    # scaler = scaler.fit(df_combined)
    # df_combined_scaled = scaler.transform(df_combined)
    # df_combined_scaled = pd.DataFrame(df_combined_scaled)
    # df_for_training_scaled = df_combined_scaled.iloc[:training_data_len, :]
    # df_for_training_scaled.head()
    # trainX = []
    # trainY = []
    # n_future = 1  # Number of days we want to predict into the future
    #
    # n_past = 60
    #
    # for i in range(n_past, len(df_for_training_scaled)):
    #     trainX.append(df_for_training_scaled.iloc[i - n_past:i, 0:df_for_training.shape[1]])
    #     trainY.append(df_for_training_scaled.iloc[i:i + n_future, 0])
    #
    # trainX, trainY = np.array(trainX), np.array(trainY)

    #Testing:
    df = web.DataReader('^NSEI', data_source='yahoo', start='2005-01-3', end='2020-05-25')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    dataframe = df.reset_index()
    print("dataframe is:", dataframe)
    date_train_list = dataframe.loc[28:,'Date']
    date_train_list
    # Separate dates for future plotting
    train_dates = df.index.values

    # Variables for training
    cols = list(df)[0:5]

    df_for_training = df[cols].astype(float)

    # df_for_plot=df_for_training.tail(5000)
    # df_for_plot.plot.line()

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    print("scaler after fit", scaler)
    print("df_for_training", df_for_training.shape)
    df_for_training_scaled = scaler.transform(df_for_training)
    print("df_for_training_scaled.shape",df_for_training_scaled.shape)

    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).
    trainX = []
    trainY = []

    n_future = 1  # Number of days we want to predict into the future
    n_past = 60  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))




    # Compile LSTM2(MultiVariate Model)
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    history = model.fit(trainX, trainY, epochs=1, batch_size=16, validation_split=0.1, verbose=1)
    # epochs are 9 in real, batch_size = 10

    xtest = []

    for i in range(len(dataframe) - 240, len(dataframe) - n_past):
        xtest.append(df_for_training_scaled[i:i + n_past, 0:df_for_training.shape[1]])

    df_for_training_scaled = pd.DataFrame(df_for_training_scaled)

    price_list = []

    for i in xtest:
        i = np.reshape(i, (1, n_past, 5))
        price = model.predict(i)
        price_list.append(price)

    # price_list = np.repeat(price_list, df_for_training.shape[1], axis=-1)
    price_list = pd.DataFrame(price_list)

    print("price_list dimension is: ", price_list.shape)
    print("(price_list)[:, 0]", (price_list)[:, 0])
    prediction_list = scaler.inverse_transform(price_list)[:, 0]

    # y_pred_future
    # price_list
    # test_prices
    test_dataset = dataframe.iloc[len(dataframe) - 180:len(dataframe), :]
    test_dates = test_dataset.filter(['Date'])
    test_dates = test_dates.reset_index(drop=True)
    test_close_prices = test_dataset.filter(['Close'])
    test_close_prices = test_close_prices.reset_index(drop=True)
    prediction_list = pd.DataFrame(prediction_list)
    prediction_list = prediction_list.reset_index(drop=True)
    prediction_list = prediction_list.filter([3])



    # df_for_testing_scaled = df_combined_scaled.iloc[training_data_len:, :]
    # df_for_testing_scaled.tail()
    # testX = []
    #
    # for i in range(n_past, len(df_for_testing_scaled)):
    #     testX.append(df_for_testing_scaled.iloc[i - n_past:i, 0:df_for_training.shape[1]])
    # testX = np.array(testX)
    # prediction_list = []
    #
    # for i in testX:
    #     i = np.reshape(i, (1, n_past, df_for_training.shape[1]))
    #     price = model.predict(i)
    #     prediction_list.append(price)
    #
    # prediction_list = np.array(prediction_list)
    # prediction_list = np.repeat(prediction_list, df_for_training_scaled.shape[1], axis=-1)
    #
    # prediction_list = scaler.inverse_transform(prediction_list)[:, 0]
    #
    # prediction_list = np.array(prediction_list)
    # # prediction_list = prediction_list.astype(int)
    # prediction_list = pd.DataFrame(prediction_list)
    # # prediction_list = prediction_list.filter(["2"])
    # prediction_list = prediction_list.filter([2])
    # df_for_testing = df_for_testing.reset_index()
    # df_for_testing
    # df_for_testing = df_for_testing.iloc[n_past:len(df_for_testing), :]
    #
    # test_close_prices = df_for_testing.filter(["Close"])
    #
    # test_close_prices = test_close_prices.filter(['Close'])
    #
    # test_dates = df_for_testing.filter(["Date"])
    # test_dates = test_dates.reset_index()
    # test_dates = test_dates.filter(['Date'])
    # test_dates

    diff = []
    sum2 = 0

    # for i in range(0, len(prediction_list)):
    #     value = abs(prediction_list.iloc[i][2] - test_close_prices.iloc[i]['Close'])
    #     diff.append(value)
    #     sum2 = sum2 + value
    #
    # average_error2 = (sum2 / len(prediction_list))
    #
    # print("average error of lstm2 is: " + str(average_error2))

    test_dates = test_dates.to_numpy()

    test_close_prices = test_close_prices.to_numpy()

    prediction_list = prediction_list.to_numpy()

    buffer = []
    for i in test_dates:
        buffer.append(str(i)[2:12])

    test_dates = buffer

    buffer = []
    for i in test_close_prices:
        buffer.append(str(i)[1:-1])

    test_close_prices = buffer
    buffer = []
    for i in prediction_list:
        buffer.append(str(i)[1:-1])

    prediction_list = buffer

    test_close_prices_180 = test_close_prices[-60:]
    test_dates_180 = test_dates[-60:]
    prediction_list_180 = prediction_list[-60:]

    test_close_prices_60 = test_close_prices[-30:]
    test_dates_60 = test_dates[-30:]
    prediction_list_60 = prediction_list[-30:]
    stock_list = Stock.objects.all()

    for stock in stock_list:
        stock_data = web.DataReader(stock.symbol, data_source='yahoo', start='2020-01-3', end='2021-05-10')
        column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
        stock_data = stock_data.reindex(columns=column_names)

        cols = list(stock_data)[0:5]

        stock_data = stock_data[cols].astype(float)
        stock_data
        scaler = StandardScaler()
        scaler = scaler.fit(stock_data)
        stock_data_scaled = scaler.transform(stock_data)
        stock_data_scaled.shape

        stock_data_scaled = pd.DataFrame(stock_data_scaled)
        stock_dataset = stock_data_scaled.iloc[-60:, 0:5]

        stock_dataset = np.array(stock_dataset)
        stock_dataset = np.reshape(stock_dataset, (1, stock_dataset.shape[0], stock_dataset.shape[1]))

        test_value = model.predict(stock_dataset)

        test_value = np.repeat(test_value, df_for_training.shape[1], axis=-1)

        test_value = scaler.inverse_transform(test_value)
        test_value = np.average(test_value)  # New changes check
        test_value

        check_close = stock_data.iloc[-1:, 0:5]
        check_close = check_close.filter(['Close'])
        check_close = np.array(check_close)
        check_close
        diff = test_value - check_close
        diff

        print("LStm2(multivariate) results:")
        print("Predicted price for tomorrow :" + str(test_value))
        print("Actual close of yesterday: "+ str(check_close))

        if diff > 0:
            stock.lstm1 = "BUY"
            stock.save()
            print("In buy of stock: " + stock.symbol)
        else:
            stock.lstm1 = "SELL"
            stock.save()
            print("In sell of stock: " + stock.symbol)

    stock_list = Stock.objects.all()
    context = {
        'stock_list':stock_list,
        'test_close_prices_60':test_close_prices_60, 'test_dates_60':test_dates_60, 'prediction_list_60': prediction_list_60,
        # 'test_close_prices1_60':test_close_prices1_60, 'test_dates1_60':test_dates1_60, 'prediction_list1_60': prediction_list1_60,
        'test_close_prices_180': test_close_prices_180, 'test_dates_180': test_dates_180,
        'prediction_list_180': prediction_list_180,
        # 'test_dates1_180': test_dates1_180, 'prediction_list1_180': prediction_list1_180,'test_close_prices1_180': test_close_prices1_180,
    }



    return render(request, 'show_calls.html', context)



def symbol_data(request, symbol):

    return HttpResponse("In symbol of "+ str(symbol))


def calculatecalls1(request):
    ##LSTM1 multiple input
    df = web.DataReader('^NSEI', data_source='yahoo', start='2000-01-3', end='2021-05-09')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    df
    dataframe = df.reset_index()
    dataframe


    ##CHECK IF THIS IS USEFUL
    date_train_list = dataframe.loc[28:, 'Date']

    date_train_list
    # Separate dates for future plotting
    train_dates = df.index.values
    ##

    # Variables for training
    cols = list(df)[0:5]

    df_for_training = df[cols].astype(float)

    # df_for_plot=df_for_training.tail(5000)
    # df_for_plot.plot.line()

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).
    trainX = []
    trainY = []

    n_future = 1  # Number of days we want to predict into the future
    n_past = 60  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    model1 = Sequential()
    model1.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model1.add(LSTM(32, activation='relu', return_sequences=False))
    model1.add(Dropout(0.2))
    model1.add(Dense(trainY.shape[1]))

    model1.compile(optimizer='adam', loss='mse')
    model1.summary()

    # fit model
    model1.fit(trainX, trainY, epochs=1, batch_size=16, validation_split=0.1, verbose=1)  ##epochs were 10

    x_test1 = []

    for i in range(len(dataframe) - 240, len(dataframe) - n_past):
        x_test1.append(df_for_training_scaled[i:i + n_past, 0:df_for_training.shape[1]])
    price_list1 = []
    for i in x_test1:
        i = np.reshape(i, (1, n_past, 5))
        price = model1.predict(i)
        price_list1.append(price)

    price_list1 = np.repeat(price_list1, df_for_training.shape[1], axis=-1)
    test_prices1 = scaler.inverse_transform(price_list1)[:, 0]
    test_dataset1 = dataframe.iloc[len(dataframe) - 180:len(dataframe), :]
    test_dates1 = test_dataset1.filter(['Date'])
    test_dates1 = test_dates1.reset_index(drop=True)
    close_price1 = test_dataset1.filter(['Close'])
    close_price1 = close_price1.reset_index(drop=True)
    test_prices1 = pd.DataFrame(test_prices1)
    test_prices1 = test_prices1.reset_index(drop=True)
    test_prices1 = test_prices1.filter([3])



    stock_list = Stock.objects.all()
    print('Number of items in First iteration of lstm1 Stock_list is : '+ str(len(stock_list)))
    for stock in stock_list:
            print("In loop of stocklist of stock"+ str(stock.symbol))
            stock_data1 = web.DataReader(stock.symbol, data_source='yahoo', start='2020-01-3', end='2021-05-10')
            column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
            stock_data1 = stock_data1.reindex(columns=column_names)

            # stock_data = stock_data.reset_index()
            cols = list(stock_data1)[0:5]

            stock_data1 = stock_data1[cols].astype(float)
            stock_data1
            scaler = StandardScaler()
            scaler = scaler.fit(stock_data1)
            stock_data_scaled1 = scaler.transform(stock_data1)
            stock_data_scaled1.shape
            type(stock_data_scaled1)

            stock_data_scaled1 = pd.DataFrame(stock_data_scaled1)
            stock_dataset1 = stock_data_scaled1.iloc[-n_past:, 0:5]

            stock_dataset1 = np.array(stock_dataset1)
            stock_dataset1 = np.reshape(stock_dataset1, (1, stock_dataset1.shape[0], stock_dataset1.shape[1]))
            print(stock_dataset1.shape)

            test_value1 = model1.predict(stock_dataset1)

            test_value1 = np.repeat(test_value1, df_for_training.shape[1], axis=-1)

            test_value1 = scaler.inverse_transform(test_value1)
            test_value1 = np.average(test_value1)             #New changes check
            test_value1

            check_close1 = stock_data1.iloc[-1:, 0:5]
            check_close1 = check_close1.filter(['Close'])
            check_close1 = np.array(check_close1)
            check_close1
            diff1 = test_value1 - check_close1
            diff1
            scope_lstm1 = diff1/check_close1 * 100
            scope_lstm1 = abs(scope_lstm1)
            print("Scope_lstm1 is : "+ str(scope_lstm1)+stock.symbol)
            stock1 = get_object_or_404(Stock, symbol=stock.symbol)
            stock1.lstm1_scope = scope_lstm1
            if diff1 > 0:
                stock1.lstm1 = "BUY"
                stock1.save()
                print("In buy of stock: " + stock.symbol)
            else:
                stock1.lstm1 = "SELL"
                stock1.save()
                print("In sell of stock: " + stock.symbol)


    df2 = web.DataReader('^NSEI', data_source='yahoo', start='2001-10-1', end='2021-05-7')
    # Show the data
    df2
    data2 = df2.filter(['Close'])
    # .values function converts dataframe to np array

    dataset2 = data2.values
    # type(dataset), type(data)
    training_data_len2 = np.math.ceil(len(dataset2) * .8)
    training_data_len2
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data2 = scaler.fit_transform(dataset2)
    scaled_data2
    train_data2 = scaled_data2[0:training_data_len2, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data2)):
        x_train.append(train_data2[i - 60:i, 0])
        y_train.append(train_data2[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape, y_train.shape
    model2 = Sequential()
    model2.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model2.add(LSTM(units=50, return_sequences=False))
    model2.add(Dense(units=25))
    model2.add(Dense(units=1))

    model2.compile(optimizer='adam', loss='mean_squared_error')

    model2.fit(x_train, y_train, batch_size=10, epochs=1)  ##epochs were 9 originally


    stock_list = Stock.objects.all()
    print('Number of items in First iteration of lstm1 Stock_list is : ' + str(len(stock_list)))
    for stock in stock_list:
            print("In loop of stocklist of stock"+ str(stock.symbol))
            stock_data = web.DataReader(stock.symbol, data_source='yahoo', start='2020-01-3', end='2021-05-10')
            column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
            stock_data = stock_data.reindex(columns=column_names)

            # stock_data = stock_data.reset_index()
            cols = list(stock_data)[0:5]

            stock_data = stock_data[cols].astype(float)
            stock_data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_stock_data = scaler.fit_transform(stock_data)
            scaled_stock_data

            scaled_stock_data = pd.DataFrame(scaled_stock_data)
            stock_dataset = scaled_stock_data.iloc[-60:, 0]

            stock_dataset = np.array(stock_dataset)
            stock_dataset.shape
            stock_dataset = np.reshape(stock_dataset, (1, stock_dataset.shape[0], 1))
            # print(stock_dataset.shape)

            test_value = model2.predict(stock_dataset)

            test_value = np.repeat(test_value, stock_data.shape[1], axis=-1)

            test_value = scaler.inverse_transform(test_value)
            test_value = np.average(test_value)
            test_value

            check_close = stock_data.iloc[-1:, 0:5]
            check_close = check_close.filter(['Close'])
            check_close = np.array(check_close)
            diff2 = test_value - check_close
            diff2
            stock2 = get_object_or_404(Stock, symbol=stock.symbol)
            scope_lstm2 = diff2 / check_close * 100
            scope_lstm2 = abs(scope_lstm2)
            stock2.lstm2_scope = scope_lstm2
            print("Scope_lstm2 is : " + str(scope_lstm2) + stock.symbol)
            if diff2 > 0:
                stock2.lstm2 = "BUY"
                stock2.save()
                print("In buy of stock: " + stock.symbol)
            else:
                stock2.lstm2 = "SELL"
                stock2.save()
                print("In sell of stock: " + stock.symbol)

    stock_list = Stock.objects.all()
    test_dates1 = test_dates1.values
    buffer = []
    for i in test_dates1:
        buffer.append(str(i)[2:12])

    test_dates1 = buffer
    ##test_dates1 = test_dates1.values.to_list()
    ##Try converting these pandas dataframe to lists
    #test_dates1 = test_dates1.to_numpy()
    close_price1 = close_price1.to_numpy()
    test_prices1 = test_prices1.to_numpy()
    print(type(test_dates1))
    print(type(close_price1))
    print(type(test_prices1))
    #test_dates1 = test_dates1.tolist()
    close_price1 = close_price1.tolist()
    test_prices1 = test_prices1.tolist()
    print(type(test_dates1))
    print(type(close_price1))
    print(type(test_prices1))

    buffer = []
    for i in close_price1:
        buffer.append(str(i)[1:-1])

    close_price1 = buffer
    buffer = []
    for i in test_prices1:
        buffer.append(str(i)[1:-1])

    test_prices1 = buffer
    print("final type passed to html is :" + str(type(test_dates1)))


    print(type(stock_list))
    return render(request, 'show_calls.html', {'stock_list':stock_list, 'test_dates1':test_dates1,
                                               'close_price1':close_price1,'test_prices1':test_prices1})










def home(request):

    stock_list = Stock.objects.all()
    message = 'In home'
    return render(request, 'home.html', {'message': message, 'stock_list': stock_list})



def index(request):
    df = web.DataReader('Reliance.NS', data_source='yahoo', start='2005-01-3', end='2021-05-09')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    df
    dataframe = df.reset_index()
    dataframe

    date_train_list = dataframe.loc[28:, 'Date']
    date_train_list
    # Separate dates for future plotting
    train_dates = df.index.values

    # Variables for training
    cols = list(df)[0:5]

    df_for_training = df[cols].astype(float)

    # df_for_plot=df_for_training.tail(5000)
    # df_for_plot.plot.line()

    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    # As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).
    trainX = []
    trainY = []

    n_future = 1  # Number of days we want to predict into the future
    n_past = 60  # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    xtest = []

    for i in range(len(dataframe) - 240, len(dataframe) - n_past):
        xtest.append(df_for_training_scaled[i:i + n_past, 0:df_for_training.shape[1]])

    df_for_training_scaled = pd.DataFrame(df_for_training_scaled)
    X_test = df_for_training_scaled.iloc[len(trainX):, 0:5]
    print(X_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
    X_test.shape
    xtest
    n_future = 90

    test_value = model.predict(X_test)  # forecast
    price_list = []

    for i in xtest:
        i = np.reshape(i, (1, n_past, 5))
        price = model.predict(i)
        price_list.append(price)

    # Perform inverse transformation to rescale back to original range
    # Since we used 5 variables for transform, the inverse expects same dimensions
    # Therefore, let us copy our values 5 times and discard them after inv

    test_value = np.repeat(test_value, df_for_training.shape[1], axis=-1)

    test_value = scaler.inverse_transform(test_value)[:, 0]

    price_list = np.repeat(price_list, df_for_training.shape[1], axis=-1)

    test_prices = scaler.inverse_transform(price_list)[:, 0]

    # y_pred_future
    # price_list
    # test_prices
    test_dataset = dataframe.iloc[len(dataframe) - 180:len(dataframe), :]
    test_dates = test_dataset.filter(['Date'])
    test_dates = test_dates.reset_index(drop=True)
    test_prices = pd.DataFrame(test_prices)
    test_prices = test_prices.reset_index(drop=True)
    test_prices1 = test_prices.filter([3])
    close_price = test_dataset.filter(['Close'])
    close_price = close_price.reset_index(drop=True)

    test_data = pd.concat([test_dates, test_prices1, close_price], axis=1)
    test_data

    dates = list(test_data["Date"])
    ytrain = list(test_data[3])
    close_price = list(test_data['Close'])



    # original = dataframe[['Date', 'Open']]
    # original['Date'] = pd.to_datetime(original['Date'])
    # original = original.loc[original['Date'] >= '2019-1-1']
    #
    # plt.plot(test_data['Date'], test_data[3], color='blue')
    # plt.plot(original['Date'], original['Open'], color='black')
    # plt.xticks(rotation=45, fontsize=8)
    # # plt.show()
    # type(original)
    print("cHECK HTML")
    return render(request, 'test.html', {'test_value':test_value, 'dates':dates,'ytrain':ytrain,'close_price':close_price})

def chart(request):

    default_items = [20, 15, 60, 60, 65, 30, 70]
    labels = ["0s", "10s", "20s", "30s", "40s", "50s", "60s"]
    data = {
        'labels':labels,
        'default_items':default_items

    }
    return render(request, 'test.html',{'labels': labels, 'default_items': default_items})

