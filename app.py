from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from cleaner import clean_text
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load model components
model = joblib.load('goemotions_model.pkl')
vectorizer = joblib.load('goemotions_vectorizer.pkl')
mlb = joblib.load('goemotions_label_encoder.pkl')

# Flask setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    emotions = []
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    if request.method == 'POST':
        journal_text = request.form.get('entry', '')
        cleaned = clean_text(journal_text)

        if cleaned.strip():
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)
            prediction_probs = model.predict_proba(vectorized)

            predicted_emotions = mlb.inverse_transform(prediction)[0]

            emotion_scores = {}
            for i, label in enumerate(mlb.classes_):
                score = prediction_probs[0][i]
                if score > 0.3:
                    emotion_scores[label] = round(score, 2)

            emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

            # Count sentiment groups (basic mapping)
            positive_emotions = {
                'admiration', 'amusement', 'approval', 'caring', 'desire',
                'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
            }
            negative_emotions = {
                'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
            }

            for emotion, score in emotion_scores.items():
                if emotion in positive_emotions:
                    counts['positive'] += score
                elif emotion in negative_emotions:
                    counts['negative'] += score
                else:
                    counts['neutral'] += score

    EMOJI_MAP = {
        'admiration': '🌟', 'amusement': '😄', 'anger': '😠', 'annoyance': '😒',
        'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🧐',
        'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
        'embarrassment': '😳', 'excitement': '🤩', 'fear': '😱', 'gratitude': '🙏',
        'grief': '😭', 'joy': '😄', 'love': '❤️', 'nervousness': '😬',
        'neutral': '😐', 'optimism': '🌈', 'pride': '🏆', 'realization': '💡',
        'relief': '😌', 'remorse': '😔', 'sadness': '😢', 'surprise': '😲'
    }

    return render_template('index.html', emotions=emotions, emotion_emoji=EMOJI_MAP, counts=counts)

if __name__ == '__main__':
    app.run(debug=True)
