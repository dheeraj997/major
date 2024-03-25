import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from functions import StockPricePredictor
from multipage_streamlit import State
from stocknews import StockNews


# Function to fetch financial data from Polygon.io
def get_financial_data(symbol):
    api_key = 'ITYlXpnp_ltg_tWq11hu1mp5ucO_w2fe'  # Replace 'YOUR_API_KEY' with your actual API key from Polygon.io
    
    try:
        # Fetch balance sheet data
        balance_sheet_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={symbol}'
        balance_sheet_response = requests.get(balance_sheet_endpoint)
        balance_sheet_response.raise_for_status()
        balance_sheet_data = balance_sheet_response.json()['results']

        # Fetch income statement data
        income_statement_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={symbol}&type=income_statement'
        income_statement_response = requests.get(income_statement_endpoint)
        income_statement_response.raise_for_status()
        income_statement_data = income_statement_response.json()['results']

        # Fetch cash flow statement data
        cash_flow_endpoint = f'https://api.polygon.io/vX/reference/financials?apiKey={api_key}&ticker={symbol}&type=cash_flow_statement'
        cash_flow_response = requests.get(cash_flow_endpoint)
        cash_flow_response.raise_for_status()
        cash_flow_data = cash_flow_response.json()['results']

        return balance_sheet_data, income_statement_data, cash_flow_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching financial data: {e}")

# Main code

predictor = StockPricePredictor()
global tinker, start_date, end_date
tinker = 'AAPL'
start_date = '2015-01-01'
end_date = '2024-01-01'


#-----------------------------------------------------------------------------------------------------------------------------------------
state = State(__name__)
st.title("Stock DashBoard")


tinker = st.sidebar.selectbox('Select Stock Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA','F'])
start_date=st.sidebar.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
end_date=st.sidebar.date_input('End Date', value=pd.to_datetime('2024-01-01'))
state.save()
df=predictor.get_historical_data(tinker, start_date, end_date)
#-----------------------------------------------------------------------------------------------------------------------------------------


# making date into date-time object
df['date']= pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ')
# Slicing data to four main features open,close,high,low with date-time as index of dataframe

gstock_data = df [['date','open','close','high','low']]
gstock_data .set_index('date',drop=True,inplace=True)


fig=px.line(gstock_data,x=gstock_data.index,y=gstock_data['close'], title =tinker)

st.plotly_chart(fig)

pricing_data,fundamental_data,news=st.tabs(["Pricing Data","Fundamental Data","Top 10 News"])

# Fetch financial data
balance_sheet_data, income_statement_data, cash_flow_data = get_financial_data(tinker)



with pricing_data:
    st.header("Pricing Movements")
    data2=df
    data2['% Change']=df['adjClose'] / df['adjClose'].shift(1)-1
    data2.dropna(inplace=True)
    st.write(df)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is', annual_return,"%")
    stdev=np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation  is', stdev*100,"%")
    st.write('Risk ADj. return is',annual_return/(stdev*100))

with fundamental_data:
    st.header("Fundamental Data")

    st.subheader('Balance Sheet')
    st.write(pd.DataFrame(balance_sheet_data))

    st.subheader('Income Statement')
    st.write(pd.DataFrame(income_statement_data))

    st.subheader('Cash Flow Statement')
    st.write(pd.DataFrame(cash_flow_data))

with news:
    st.header(f'News of {tinker}')
    sn = StockNews(tinker, save_news=False)
    df_news= sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment=df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
