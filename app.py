import streamlit as st
import joblib
import spacy

model = joblib.load('saved_model/model.pkl')
vectorizer = joblib.load('saved_model/vectorizer.pkl')

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0f1117;
    }
    .stTextArea textarea {
        font-size: 16px;
        background-color: #1e1e1e;
        color: white;
        border-radius: 10px;
    }
    .stButton > button {
        color: white;
        background-color: #7c3aed;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #a78bfa;
        color: black;
    }
    .stMarkdown h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("## 🎵 Music Genre Predictor 🎵")
    st.markdown("Paste your song lyrics below and predict the genre!")

    lyrics_input = st.text_area(
        'Enter Lyrics:',
        placeholder="E.g.\nI'm too hot (hot damn), call the police and the fireman...",
        height=300
    )

    if st.button('Predict Genre'):
        if lyrics_input.strip():
            cleaned = preprocess(lyrics_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized).max()

            st.success(f'🎧 Predicted Genre: **{prediction}**')
            st.caption(f'Confidence: `{confidence:.2f}`')
        else:
            st.warning('Please paste some lyrics first!')
