import streamlit as st
import requests
import json

# Set the API url (replace with your own if different)
API_URL = "http://localhost:8000/predict"

st.title('Text Classification App')

# Input text box for user to input text
user_input = st.text_input("Please enter text")

# 'Predict' button
if st.button('Predict'):
    # Make sure user has input something
    if user_input:
        # Data to send to the API
        data = {'title': user_input}

        # Send POST request to API
        response = requests.post(API_URL, json=data)

        # If request was successful
        if response.status_code == 200:
            # Extract prediction from response JSON
            prediction = response.json()['prediction']

            # Display prediction
            st.write(f'Prediction: {prediction}')

        # If request was not successful, display error message
        else:
            st.write(f'Error: {response.text}')

    else:
        st.write('Please enter some text')
