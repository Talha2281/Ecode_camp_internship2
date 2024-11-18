import streamlit as st
import joblib
import numpy as np
import os
from groq import Groq

# Load your diabetes prediction model
model_path = 'diabates 2.pkl'  # Ensure the model file is in the project directory
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please upload 'diabetes 2.pkl' to the project directory.")
    st.stop()

# Retrieve the API key from environment variables
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable is not set. Please configure it as a GitHub secret or Streamlit secret.")
    st.stop()

# Initialize the Groq client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq API client: {e}")
    st.stop()

# Function to get advice using the Llama model via Groq API
def get_llama_advice(query):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="llama3-8b-8192",
        )
        advice = chat_completion.choices[0].message.content
        return advice
    except Exception as e:
        return f"Error in fetching advice: {str(e)}"

# Streamlit interface
st.title("Diabetes Prediction App with Expert Advice")
st.sidebar.title("Expert Advice")
st.write("This application is created by TALHA KHAN")

# Input fields for diabetes prediction
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

# Predict Diabetes
if st.button("Predict Diabetes"):
    if any(v == 0 for v in inputs.values()):
        st.warning("Please fill in all fields before predicting.")
    else:
        input_data = np.array([list(inputs.values())])
        prediction = model.predict(input_data)
        result = "You have diabetes." if prediction[0] == 1 else "You do not have diabetes."
        st.write(result)

# Sidebar for Llama model advice
st.sidebar.subheader("Ask for Advice")
user_query = st.sidebar.text_input("Enter your question (e.g., advice on diet)")

if user_query:
    st.sidebar.write("Fetching advice, please wait...")
    advice = get_llama_advice(user_query)
    st.sidebar.write("Advice:", advice)






     
 



     



