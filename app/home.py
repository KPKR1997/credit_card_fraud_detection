import os
import sys
import streamlit as st
import pandas as pd
from src.utils import load_object
from src.exception import CustomException

st.title("Credit card fraud detection")

st.header('Input values')


day_of_week = st.selectbox('Day of week transaction done', ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

time = st.number_input('Time of transaction: (in 0-23)', min_value=0, max_value=23, step=1)

card_type = st.selectbox('Card Type', ['Visa', 'MasterCard'])

entry_mode = st.selectbox('Entry Mode', ['Tap', 'PIN', 'CVC'])

amount = st.number_input('Transaction Amount', min_value=0.0, step=0.01)

transaction_type = st.selectbox('Transaction Type', ['POS', 'Online', 'ATM'])

merchant = st.selectbox('Merchant', ['Entertainment', 'Services', 'Restaurant', 'Electronics', 'Children',
 'Fashion', 'Food', 'Products', 'Subscription', 'Gaming'])

country_of_transaction = st.selectbox('Country of Transaction', ['United Kingdom', 'USA', 'India', 'Russia', 'China'])

shipping_address = st.selectbox('Shipping Address Country', ['United Kingdom', 'USA', 'India', 'Russia', 'China'])

residence =  st.selectbox('Country of Rseidence', ['United Kingdom', 'USA', 'India', 'Russia', 'China'])

gender = st.radio('Gender', ['M', 'F'])

age = st.number_input('Age', min_value=0, max_value=100, step=1)

bank = st.selectbox('Bank', ['RBS', 'Lloyds', 'Barclays', 'Halifax', 'Monzo', 'HSBC', 'Metro', 'Barlcays'])

input_data = {
    'day': [day_of_week],
    'time': [time],
    'card_type': [card_type],
    'entry_mode': [entry_mode],
    'amount': [amount],
    'transaction_type': [transaction_type],
    'merchant': [merchant],
    'country_of_transaction': [country_of_transaction],
    'shipping_address': [shipping_address],
    'residence': [residence],
    'gender': [gender],
    'age': [age],
    'bank': [bank],
}

df = pd.DataFrame(input_data)


def predict():
    try:
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model = load_object(file_path = model_path)
        preprocessor = load_object(file_path=preprocessor_path)

        data = preprocessor.transform(df)

        prediction = model.predict(data)

        return prediction

    except Exception as e:
        raise CustomException(e,sys)
    


if st.button(label = 'Predict'):
    val = predict()
    if val == 1:
        st.warning("Transaction seems to be Fraud")
    else:
        st.success('Transaction seems to be Legit')
    

    