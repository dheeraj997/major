import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf  
class StockPricePredictor:
    def __init__(self):
        pass
    
    def get_historical_data(self, tinker,start_date,end_date):
   
        headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Token 0b4623cf02c29229cfa1f8790ccba6d0bd04983c'
        }

        url = f"https://api.tiingo.com/tiingo/daily/{tinker}/prices"

        params = {
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq':'daily'
        }

        try:
            requestResponse = requests.get(url,
                                        headers=headers,
                                        params=params)
            data = pd.DataFrame(requestResponse.json())
            requestResponse.raise_for_status()  # Raise an error for bad responses
            print(requestResponse.json())
        except requests.exceptions.RequestException as e:
            print("Error fetching data:", e)

        return data
        
    
    def preprocess_data(self, df):
        
        try:
    # Attempt to convert 'date' to datetime format if it exists
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ')

        except KeyError:
    # If 'date' column is missing, print DataFrame columns
            print("Dataframe does not contain a 'date' column. Existing columns are:")
            print(df.head)
    # You can potentially raise an error here if 'date' is strictly required
        # Slicing data to four main features open,close,high,low with date-time as index of dataframe
        gstock_data = df [['date','open','close','high','low']]
        gstock_data .set_index('date',drop=True,inplace=True)

        #Making scsasling values between 0 and 1
        scaler=MinMaxScaler()
        gstock_data [gstock_data .columns] = scaler.fit_transform(gstock_data )
        #creating trainig ad testing data
        training_size=round(len(gstock_data)*0.80)

        train_data = gstock_data [:training_size]
        test_data  = gstock_data [training_size:]

        #Creating trianing and test sequences with a past look back of 50 days
        def create_sequence(df:pd.DataFrame):
            sequences=[]
            labels=[]
            start_index=0
            for stop_index in range(50,len(df)):
                sequences.append(df.iloc[start_index:stop_index])
                labels.append(df.iloc[stop_index])
                start_index +=1
            return (np.array(sequences), np.array(labels))
        train_seq, train_label = create_sequence(train_data)
        test_seq, test_label = create_sequence(test_data)

        return train_seq, train_label,test_seq, test_label, scaler, gstock_data, training_size
    
    
    def preprocess_yfinance_data(self, ticker, years=10):
        # Retrieve data from yfinance for the specified years
        today = pd.to_datetime('today')
        end = today
        start = today - pd.DateOffset(years=years)
        df = yf.download(ticker, start=start, end=end)


        # Rest of your preprocessing steps
        gstock_data = df[['Open', 'Close', 'High', 'Low']]

        self.scaler = MinMaxScaler()  # Fit scaler on the entire data
        gstock_data [gstock_data .columns] = self.scaler.fit_transform(gstock_data )

        # Splitting data into training and testing sets
        training_size = round(len(gstock_data) * 0.8)
        train_data = gstock_data[:training_size]
        test_data = gstock_data[training_size:]

        # Creating training and testing sequences (assuming create_sequence is a class method)
        #Creating trianing and test sequences with a past look back of 50 days
        def create_sequence(df:pd.DataFrame):
            sequences=[]
            labels=[]
            start_index=0
            for stop_index in range(50,len(df)):
                sequences.append(df.iloc[start_index:stop_index])
                labels.append(df.iloc[stop_index])
                start_index +=1
            return (np.array(sequences), np.array(labels))
        train_seq, train_label = create_sequence(train_data)
        test_seq, test_label = create_sequence(test_data)

        return train_seq, train_label, test_seq, test_label, self.scaler, gstock_data, training_size

    
    def create_lstm_model(self, train_seq):
        
        model=Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        

        return model
    
    def predictions_data_analysis(self, test_predicted, gstock_data, scaler):
        
        test_inverse_predicted = scaler.inverse_transform(test_predicted)

        gstock_subset = gstock_data.iloc[-test_predicted.shape[0]:].copy()

        # Creating a DataFrame from the predicted values with appropriate columns
        predicted_df = pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted','high predicted','low predictied'])

        # Aligning the index of predicted_df with the index of gstock_subset
        predicted_df.index = gstock_subset.index

        # Concatenating the two DataFrames along the columns axis
        gs_slice_data = pd.concat([gstock_subset, predicted_df], axis=1)

        gs_slice_data[['Open','Close','High','Low']] = scaler.inverse_transform(gs_slice_data[['Open','Close','High','Low']])

        fig = go.Figure()

        # Plot actual 'open' and 'open_predicted'
        fig.add_trace(go.Scatter(x=gs_slice_data.index, y=gs_slice_data['Open'], mode='lines', name='Actual Open'))
        fig.add_trace(go.Scatter(x=gs_slice_data.index, y=gs_slice_data['open_predicted'], mode='lines', name='Predicted Open', line=dict(dash='dash')))

        # Plot actual 'close' and 'close_predicted'
        fig.add_trace(go.Scatter(x=gs_slice_data.index, y=gs_slice_data['Close'], mode='lines', name='Actual Close'))
        fig.add_trace(go.Scatter(x=gs_slice_data.index, y=gs_slice_data['close_predicted'], mode='lines', name='Predicted Close', line=dict(dash='dash')))

        fig.update_layout(
            title='Actual vs Predicted Stock Prices',
            xaxis=dict(title='Date', tickangle=45),
            yaxis=dict(title='Stock Price'),
            legend=dict(x=0, y=1.2),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            width=800,
            title_x=0.6,  # Adjust the x position of the title to move it towards the right
            title_y=0.9    # Adjust the y position of the title to center it vertically
        )

        st.plotly_chart(fig)             
        return gs_slice_data
    
    def forecasting(self, temp_input,model):
       
        lst_output = []
        n_steps = 50  # Number of timesteps
        n_features = 4  # Number of features
        i = 0

        while i < 30:
            if len(temp_input) > n_steps:
                x_input = temp_input[-n_steps:] # Select the last n_steps elements
                yhat = model.predict(x_input, verbose=0)  # Predict next value
                temp_input = np.concatenate((temp_input, yhat.reshape(1, -1, n_features)), axis=0) # Append prediction to temp_input
                lst_output.append(yhat[0])  # Append prediction to lst_output
                i += 1
            else:
                x_input = temp_input  # Use all available elements
                yhat = model.predict(x_input, verbose=0)  # Predict next value
                temp_input = np.concatenate((temp_input, yhat.reshape(1, -1, n_features)), axis=0) # Append prediction to temp_input
                lst_output.append(yhat[0])  # Append prediction to lst_output
                i += 1
        return lst_output        
    def plot_predictions(self, gs_slice_data, scaler, test_predicted, lst_output):
        
        day_new = np.arange(len(gs_slice_data) - len(test_predicted), len(gs_slice_data))
        day_pred = np.arange(len(gs_slice_data), len(gs_slice_data) + len(lst_output))

        fig = go.Figure()

        # Plotting the actual predicted values
        fig.add_trace(go.Scatter(x=day_new, y=scaler.inverse_transform(test_predicted).flatten(), mode='lines', name='Actual Predicted'))

        # Plotting the forecasted values
        fig.add_trace(go.Scatter(x=day_pred, y=scaler.inverse_transform(lst_output).flatten(), mode='lines', name='Forecast'))

        fig.update_layout(
            title='Actual Predicted vs Forecasted Stock Prices',
            xaxis=dict(title='Date'),  # Removed 'size' property here
            yaxis=dict(title='Stock Price'),
            legend=dict(x=0, y=1.2),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            width=800,
            title_x=0.6,  # Adjust the x position of the title to move it towards the right
            title_y=0.9    # Adjust the y position of the title to center it vertically
        )

        st.plotly_chart(fig)

        # Forecast plotting
        forecast_dates = pd.date_range(start=gs_slice_data.index[-1] + pd.Timedelta(days=1), periods=len(lst_output))

        # Creating a DataFrame for the forecasted values with the forecast_dates as index
        forecast_df = pd.DataFrame(lst_output, index=forecast_dates, columns=['future_close_predicted', 'future_open_predicted', 'future_high_predicted', 'future_low_predicted'])

        # Concatenating the forecast_df with gs_slice_data
        combined_data = pd.concat([gs_slice_data, forecast_df])

                
        combined_data[['future_open_predicted', 'future_close_predicted', 'future_high_predicted', 'future_low_predicted']] = scaler.inverse_transform(combined_data[['future_open_predicted', 'future_close_predicted', 'future_high_predicted', 'future_low_predicted']])


        # Plotting the data
        fig = px.line(combined_data, x=combined_data.index, y=['future_open_predicted', 'future_close_predicted', 'future_high_predicted', 'future_low_predicted'],
                    labels={'value': 'Stock Price', 'variable': 'Stock Type'},
                    title='Forecasted Stock Prices')

        fig.update_layout(
            xaxis=dict(title='Date'),  # Removed 'size' property here
            yaxis=dict(title='Stock Price'),
            legend_title='Stock Type',
            legend=dict(x=0, y=1.2),
            height=400,
            width=800,
            title_x=0.6,  # Adjust the x position of the title to move it towards the right
            title_y=0.9    # Adjust the y position of the title to center it vertically
        )

        st.plotly_chart(fig)
    
