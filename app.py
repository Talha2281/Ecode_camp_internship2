import streamlit as st
import pickle
import numpy as np

# Load the trained  model
model_path = 'diabates 2.pkl'  # Name of your model file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of the web app
st.title('Diabetes Prediction App')
st.write("This app is created by TALHA KHAN")  # Replace [Your Name] with your actual name

# Input fields for user data
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Age', min_value=10, max_value=100, step=1)

# Prediction function
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    st.write("Input data shape:", input_data.shape)  # Check the shape of the input data
    prediction = model.predict(input_data)
    
    if prediction == 1:
        st.write('The model predicts that you have diabetes.')
    else:
        st.write('The model predicts that you do not have diabetes.')


