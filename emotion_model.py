import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load your data
df = pd.read_csv("data/train.txt", sep=';', names=['text', 'label'], header=None)  # or your actual data file

# Features and labels
X = df["text"]      # Text column
y = df["label"]     # Emotion label column

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# Build model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/emotion_model.pkl")

print("✅ Model trained and saved!")