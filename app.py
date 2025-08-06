import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/emotion_model.pkl")

# Title
st.title("🧠 MindPulse: Emotion Detector")

# Text input
user_input = st.text_input("Enter a sentence tstreamlit run app.pyo detect emotion:")

# Predict button
if st.button("Predict Emotion"):
        prediction = model.predict([user_input])[0]
        st.success(f"Predicted Emotion: {prediction}")
