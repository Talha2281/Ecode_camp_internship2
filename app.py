import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the KNN model
model_path = 'diabates 2.pkl'  # Make sure this path is correct
model = joblib.load(model_path)

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

# Debugging: Checking if the input fields are working
if pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and dpf and age:
    st.write("Input fields are working!")
else:
    st.write("Some fields may not be properly initialized.")

# Prediction button
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Standardize input data using predefined mean and std values
    scaler = StandardScaler()
    scaler.mean_ = np.array([3.8, 120.9, 69.1, 20.5, 80.5, 32.0, 0.5, 33.2])  # Replace with actual mean values
    scaler.scale_ = np.array([3.2, 32.0, 19.4, 15.9, 115.2, 7.9, 0.3, 11.8])   # Replace with actual std values
    input_data = scaler.transform(input_data)

    # Make prediction
    try:
        prediction = model.predict(input_data)

        # Display the prediction outcome
        if prediction[0] == 1:
            st.write('The model predicts that you have diabetes.')
        else:
            st.write('The model predicts that you do not have diabetes.')

    except Exception as e:
        st.write("An error occurred:", str(e))


 



     



