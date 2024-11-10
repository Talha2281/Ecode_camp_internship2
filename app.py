import streamlit as st
import joblib
import numpy as np
import os
import requests
from groq import Groq

# Load your diabetes prediction model
model_path = 'diabates 2.pkl'  # Update with your model path if necessary
model = joblib.load(model_path)

# Initialize Groq API client for Llama 3 8B model
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to get advice using Llama 3 8B model via Groq API
def get_llama_advice(query):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama3-8b-8192",
        )
        advice = response.choices[0].message.content
        return advice
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit interface for the diabetes prediction and advice app
st.title("Diabates Prediction App with Expert Advice")

st.sidebar.title("Expert Advice")

# Diabetes prediction input fields
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input('Age', min_value=10, max_value=100, step=1)

# Button for prediction
if st.button("Predict Diabetes"):
    # Check if all fields have been filled
    if pregnancies == 0 or glucose == 0 or blood_pressure == 0 or skin_thickness == 0 or insulin == 0 or bmi == 0 or dpf == 0 or age == 0:
        st.warning("Please fill in all fields before predicting.")
    else:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)

        # Display prediction result
        if prediction[0] == 1:
            st.write("The model predicts that you have diabetes.")
        else:
            st.write("The model predicts that you do not have diabetes.")

# Sidebar for user query to Llama model
st.sidebar.subheader("Ask for Advice")
user_query = st.sidebar.text_input("Enter your question (e.g., advice on diet)")

if user_query:
    with st.sidebar:
        st.write("Generating advice, please wait...")
        advice = get_llama_advice(user_query)
        st.write("Advice:", advice)






     
 



     



