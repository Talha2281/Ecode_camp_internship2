import streamlit as st
import joblib
import numpy as np
import requests  # For making HTTP requests to GIMNI API

# Load the KNN model
model_path = 'diabates 2.pkl'  # Ensure this is the correct path for your trained model
model = joblib.load(model_path)

# GIMNI API integration
GIMNI_API_KEY = 'AIzaSyBZ-MtjPu2rtQhUyruMphzuwL87_lKuDRk'  # Your GIMNI API Key
GIMNI_API_URL = 'https://api.gimni.com/advice'  # Replace with actual GIMNI API URL

# Title and creator information
st.title('Diabetes Prediction App')
st.write("This app is created by TALHA KHAN")

# Input fields for user data
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Age', min_value=10, max_value=100, step=1)

# Sidebar for GIMNI Query
st.sidebar.header("Ask GIMNI for Advice")
user_query = st.sidebar.text_input("Enter your query:")

# Function to get advice from GIMNI API
def get_advice(query):
    headers = {'Authorization': f'Bearer {GIMNI_API_KEY}'}
    response = requests.post(GIMNI_API_URL, json={'query': query}, headers=headers)

    if response.status_code == 200:
        advice = response.json().get('advice', 'Sorry, no advice available.')
        return advice
    else:
        return 'Error fetching advice. Please try again later.'

# Display GIMNI advice if user entered a query
if user_query:
    advice = get_advice(user_query)
    st.sidebar.write(f"**GIMNI Advice:** {advice}")

# Check if all fields are filled before allowing prediction
if st.button('Predict'):
    if not (pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and dpf and age):
        st.warning('Please enter all the required values before making a prediction.')
    else:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

        # Debugging: Print input data
        st.write("Input data:", input_data)

        # Make prediction without scaling
        try:
            prediction = model.predict(input_data)
            
            # Debugging: Print prediction
            st.write("Prediction:", prediction)

            # Display the prediction outcome
            if prediction[0] == 1:
                st.write('The model predicts that you have diabetes.')
            else:
                st.write('The model predicts that you do not have diabetes.')

        except Exception as e:
            st.write("An error occurred:", str(e))





     
 



     



