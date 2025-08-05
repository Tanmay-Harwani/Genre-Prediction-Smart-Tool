import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
from spacy.cli import download as spacy_download

df = pd.read_csv('lyrics_dataset.csv')

df = df[['Lyric', 'Genre']].dropna()

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

df['cleaned_lyrics'] = df['Lyric'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_lyrics'])

y = df['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'saved_model/model.pkl')
joblib.dump(vectorizer, 'saved_model/vectorizer.pkl')

print("Model and vectorizer saved successfully!")
