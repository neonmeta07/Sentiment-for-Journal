import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from cleaner import clean_text

# Load emotion labels
with open("goemotions/emotions.txt", "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

# Total columns: text + id + 27 emotions = 29
columns = ['text', 'id'] + emotion_labels

# Load all 3 datasets
df1 = pd.read_csv("goemotions/goemotions_1.csv", header=None, names=columns, low_memory=False)
df2 = pd.read_csv("goemotions/goemotions_2.csv", header=None, names=columns, low_memory=False)
df3 = pd.read_csv("goemotions/goemotions_3.csv", header=None, names=columns, low_memory=False)

# Combine them
df = pd.concat([df1, df2, df3], ignore_index=True)

for label in emotion_labels:
    df[label] = pd.to_numeric(df[label], errors='coerce').fillna(0).astype(int)

# Keep rows with at least 1 emotion
df['label_count'] = df[emotion_labels].sum(axis=1)
df = df[df['label_count'] > 0]

# Clean text
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.strip() != ""]

# Create multi-label list
df['emotions'] = df[emotion_labels].apply(
    lambda row: [emo for emo, present in zip(emotion_labels, row) if present == 1],
    axis=1
)
label_counts = df['emotions'].explode().value_counts()
rare_labels = label_counts[label_counts < 100].index.tolist()

df['emotions'] = df['emotions'].apply(lambda labels: [l for l in labels if l not in rare_labels])
df = df[df['emotions'].map(len) > 0]
print(f"✅ Data shape after cleaning: {df.shape}")
print("🧠 Sample cleaned text:\n", df['clean_text'].sample(5).to_string(index=False))

# Vectorize
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])

# Label encode
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['emotions'])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save everything
joblib.dump(model, 'goemotions_model.pkl')
joblib.dump(vectorizer, 'goemotions_vectorizer.pkl')
joblib.dump(mlb, 'goemotions_label_encoder.pkl')

print("✅ Model, vectorizer, and label encoder saved successfully!")
