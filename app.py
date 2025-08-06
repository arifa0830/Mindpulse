import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/emotion_model.pkl")

# Map each emotion to an emoji
emoji_map = {
    'happy': '😊',
    'angry': '😡',
    'sad': '😢',
    'love': '❤️',
    'calm': '😌',
    'fear': '😱',
    'nervous': '😬',
    'lonely': '🥺',
    'surprised': '😲',
    'disgusted': '🤢',
    'embarrassed': '😳',
    'confused': '🤔'
}

# Title
st.title("🧠 MindPulse: Emotion Detector")

# Text input
user_input = st.text_input("Enter a sentence to detect emotion:")

# Predict button
if st.button("Predict Emotion"):
    prediction = model.predict([user_input])[0]
    emoji = emoji_map.get(prediction, '')  # Get emoji or empty if not found
    st.success(f"Predicted Emotion: {prediction} {emoji}")


