import streamlit as st
import streamlit as st  # Assuming you're using streamlit-multipage

st.title("StockX: Stock Price Prediction with Deep Learning")
st.write("StockX is a web application that utilizes the power of Deep Learning and Long Short-Term Memory (LSTM) models to predict future stock prices. This allows you to make informed decisions based on data-driven insights.")
st.sidebar.header("Navigation")
st.write("Welcome to StockX! Click a button above to navigate.")
# Add pages using the correct method (streamlit-multipage)
st.page_link("StockDashboard.py",label="Stock Dashboard",icon="🏠")
st.page_link("model.py",label="model_training",icon="🌎")

# Add creator names and contact information
st.write("Created by:")
st.write("- Tudimilla Dheeraj Kummar Chary")
st.write("- Nagarjuna Reddy")
st.write("- Kotika Venkata Kavya")

# Add contact information
st.write("Contact us:")
st.write("Phone: 7893334349")

st.sidebar.write("Jane Smith (20eg107150@anurag.edu.in)")

# Disclaimer (optional)
st.write("**Disclaimer:** Stock price predictions are not guaranteed to be accurate. Please conduct your own research and due diligence before making any investment decisions.")