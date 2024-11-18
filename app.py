import streamlit as st
import joblib
import numpy as np
import os
import requests

# Load your diabetes prediction model
model_path = 'diabetes 2.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'diabetes 2.pkl' is in the project directory.")

# Function to simulate advice fetching (replace with actual API call)
def get_llama_advice(query):
    try:
        # Simulated response for demo
        advice = f"Advice for your query '{query}': Stay hydrated and exercise regularly."
        return advice
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit interface
st.title("Diabetes Prediction App with Expert Advice")
st.sidebar.title("Expert Advice")

# Input fields
inputs = {
    'pregnancies': st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1),
    'glucose': st.number_input('Glucose Level', min_value=0, max_value=200, step=1),
    'blood_pressure': st.number_input('Blood Pressure', min_value=0, max_value=150, step=1),
    'skin_thickness': st.number_input('Skin Thickness', min_value=0, max_value=100, step=1),
    'insulin': st.number_input('Insulin Level', min_value=0, max_value=900, step=1),
    'bmi': st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1),
    'dpf': st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01),
    'age': st.number_input('Age', min_value=10, max_value=100, step=1),
}

if st.button("Predict Diabetes"):
    if any(v == 0 for v in inputs.values()):
        st.warning("Please fill in all fields before predicting.")
    else:
        input_data = np.array([list(inputs.values())])
        prediction = model.predict(input_data)
        result = "You have diabetes." if prediction[0] == 1 else "You do not have diabetes."
        st.write(result)

# Sidebar advice section
st.sidebar.subheader("Ask for Advice")
user_query = st.sidebar.text_input("Enter your question (e.g., advice on diet)")
if user_query:
    advice = get_llama_advice(user_query)
    st.sidebar.write("Advice:", advice)






     
 



     



