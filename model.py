import streamlit as st
from functions import StockPricePredictor

class Train:
    def __init__(self):
        self.tinker = "AAPL"  # Store user-selected tinker symbol
        self.start_date = None  # Store user-selected start date
        self.end_date = None  # Store user-selected end date
        self.model = None  # Store trained model
        self.scaler = None  # Store data scaler
        self.gstock_data = None  # Store preprocessed stock data

    def run(self):
        """
        Main entry point for training and forecasting functionality.
        """
        # Get user input for tinker, start date, and end date from Streamlit UI elements
        self.get_user_input()

        if self.tinker and self.start_date and self.end_date:
            # Separate buttons for training and forecasting
            if st.button('Get results'):
                self.train_model()
                predictor = StockPricePredictor()
                train_seq, train_label, test_seq, test_label,scaler,gstock_data,training_size = predictor.preprocess_yfinance_data("AAPL", years=5)  # Get test sequence input from user
                self.forecasting(test_seq,gstock_data,scaler,self.model)
        else:
            st.error("Please select a tinker, start date, and end date.")

    def get_user_input(self):
        """
        Retrieves user input for tinker, start date, and end date from Streamlit UI.
        """
        self.tinker = st.selectbox('Ticker', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'F'])
        self.start_date = st.date_input("Select Start Date:")
        self.end_date = st.date_input("Select End Date:")

    def train_model(self):
        """
        Trains the model with user-selected parameters.
        """
        predictor = StockPricePredictor()
        train_seq, train_label, test_seq, test_label, self.scaler,self.gstock_data,training_size = predictor.preprocess_yfinance_data("AAPL", years=5)
        self.model = predictor.create_lstm_model(train_seq)
        with st.spinner("Training the model...This may take a while!"):
            self.model.fit(train_seq, train_label, epochs=10, validation_data=(test_seq, test_label), verbose=1)
        st.success("Model trained successfully!")

    def forecasting(self, test_seq, gstock_data, scaler, model):
        try:
            if model is None:
                st.error("Please train the model first.")
                return

            test_predicted = model.predict(test_seq)
            # Model data analysis
            gs_slice_data = StockPricePredictor().predictions_data_analysis(test_predicted, gstock_data, scaler)
            # Forecasting
            lst_output = StockPricePredictor().forecasting(test_seq, model)
            # Plotting the forecast predictions
            StockPricePredictor().plot_predictions(gs_slice_data, scaler, test_predicted, lst_output)
            st.success("polted successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def runn():
    """
    Entry point for the entire application.
    """
    train_instance = Train()
    train_instance.run()

if __name__ == "__main__":
    runn()
