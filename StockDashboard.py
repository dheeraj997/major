import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from functions import StockPricePredictor
from stocknews import StockNews

class StockDashboard:
    def __init__(self):
        self.predictor = StockPricePredictor()
        self.tinker = 'AAPL'
        self.start_date = pd.to_datetime('2015-01-01')
        self.end_date = pd.to_datetime('2024-01-01')

    def select_stock_ticker(self):
        st.sidebar.title("Select Stock Ticker")
        self.tinker = st.sidebar.selectbox('Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'F'])

    def select_dates(self):
        st.sidebar.title("Select Dates")
        self.start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
        self.end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-01-01'))

    def fetch_historical_data(self):
        self.df = self.predictor.get_historical_data(self.tinker, self.start_date, self.end_date)
                        # making date into date-time object
        self.df['date']= pd.to_datetime(self.df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ')
        # Slicing data to four main features open,close,high,low with date-time as index of dataframe

        gstock_data = self.df [['date','open','close','high','low']]
        gstock_data .set_index('date',drop=True,inplace=True)


        fig=px.line(gstock_data,x=gstock_data.index,y=gstock_data['close'], title =self.tinker)

        st.plotly_chart(fig)

        

    def display_price_movement(self):
        st.header("Pricing Movements")
        data2 = self.df
        data2['% Change'] = self.df['adjClose'] / self.df['adjClose'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write('Annual Return is', annual_return, "%")
        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.write('Standard Deviation  is', stdev * 100, "%")
        st.write('Risk ADj. return is', annual_return / (stdev * 100))

    def fetch_financial_data(self):
        api_key = 'ITYlXpnp_ltg_tWq11hu1mp5ucO_w2fe'  # Replace 'YOUR_API_KEY' with your actual API key from Polygon.io

        try:
            # Fetch balance sheet data
            balance_sheet_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={self.tinker}'
            balance_sheet_response = requests.get(balance_sheet_endpoint)
            balance_sheet_response.raise_for_status()
            self.balance_sheet_data = balance_sheet_response.json()['results']

            # Fetch income statement data
            income_statement_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={self.tinker}&type=income_statement'
            income_statement_response = requests.get(income_statement_endpoint)
            income_statement_response.raise_for_status()
            self.income_statement_data = income_statement_response.json()['results']

            # Fetch cash flow statement data
            cash_flow_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={self.tinker}&type=cash_flow_statement'
            cash_flow_response = requests.get(cash_flow_endpoint)
            cash_flow_response.raise_for_status()
            self.cash_flow_data = cash_flow_response.json()['results']

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching financial data: {e}")

    def display_fundamental_data(self):
        st.header("Fundamental Data")

        st.subheader('Balance Sheet')
        st.write(pd.DataFrame(self.balance_sheet_data))

        st.subheader('Income Statement')
        st.write(pd.DataFrame(self.income_statement_data))

        st.subheader('Cash Flow Statement')
        st.write(pd.DataFrame(self.cash_flow_data))

    def display_news(self):
        st.header(f'News of {self.tinker}')
        sn = StockNews(self.tinker, save_news=False)
        df_news = sn.read_rss()
        for i in range(10):
            st.subheader(f'News {i + 1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment {news_sentiment}')

    def run(self):
        st.title("Stock Dashboard")

        self.select_stock_ticker()
        self.select_dates()
        self.fetch_historical_data()
        pricing_data,fundamental_data,news=st.tabs(["Pricing Data","Fundamental Data","Top 10 News"])
        
        with pricing_data:
            st.title("Pricing Data")
            self.display_price_movement()

        with fundamental_data:
            st.title("Fundamental Data")
            self.fetch_financial_data()  # Fetch financial data first
            if hasattr(self, 'balance_sheet_data'):  # Check if balance_sheet_data is available
                self.display_fundamental_data()  # Display fundamental data if available
            else:
                st.warning("Financial data is not available.")

        with news:
            st.title("Top 10 News")
            self.display_news()

if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run()
