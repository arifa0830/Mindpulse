import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("✅ Starting training process...")

# Create model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")
    print("📁 Created 'model/' folder")

# Extended training dataset with emojis
data = {
    'text': [
        'I am so happy today!',
        'This is terrible, I hate it.',
        'I feel awesome and excited!',
        'I am very sad and depressed.',
        'You make me angry!',
        'I love this!',
        'I am frustrated and annoyed.',
        'I am calm and relaxed.',
        'I am scared of the dark.',
        'This makes me joyful and cheerful.',
        'Why is everything so hard?',
        'Leave me alone!',
        'I feel nervous before the exam.',
        'That was disappointing.',
        'I am very angry with you!',
        'What a peaceful moment.',
        'I’m terrified!',
        'You broke my heart.',
        'Wow, this is amazing!',
        'I’m having a panic attack.'
    ],
    'emotion': [
        'happy',    # 4 total
        'angry',
        'happy',
        'sad',
        'angry',
        'happy',
        'angry',
        'calm',
        'fear',
        'happy',
        'sad',      # 3 total
        'angry',
        'fear',
        'sad',
        'angry',
        'calm',
        'fear',     # 3 total
        'sad',
        'happy',
        'fear'
    ]
}


df = pd.DataFrame(data)
print("📄 Data loaded with", len(df), "samples.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)
print("🔀 Split into training and test sets.")

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)
print("✅ Model trained.")

# Save model
joblib.dump(pipeline, 'model/emotion_model.pkl')
print("✅ Model saved to model/emotion_model.pkl")
