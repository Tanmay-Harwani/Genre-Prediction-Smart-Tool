import streamlit as st
import joblib
import re
import time
from pathlib import Path


# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('saved_model/model.pkl')
        vectorizer = joblib.load('saved_model/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure 'saved_model/model.pkl' and 'saved_model/vectorizer.pkl' exist")
        return None, None


# Simple text preprocessing without spaCy dependency
def preprocess(text):
    """
    Simple text preprocessing function that doesn't require spaCy
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Simple stop word removal (basic English stop words)
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
        'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }

    # Filter out stop words
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]

    return ' '.join(words)


# Genre color mapping
def get_genre_color(genre):
    colors = {
        'Pop': '#FF6B6B',  # Coral red
        'Hip-Hop': '#4ECDC4',  # Teal
        'Rock': '#45B7D1',  # Sky blue
        'R&B': '#96CEB4',  # Mint green
        'Alternative': '#FFEAA7',  # Soft yellow
        'K-pop': '#DDA0DD',  # Plum
        'Rap': '#98D8C8',  # Aqua mint
    }
    return colors.get(genre, '#6C5CE7')  # Default purple


# Streamlit app configuration
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }

    .stTextArea textarea {
        font-size: 16px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
    }

    .stButton > button {
        color: white;
        background: linear-gradient(45deg, #7c3aed, #a855f7);
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
        background: linear-gradient(45deg, #a855f7, #c084fc);
    }

    .genre-result {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        height: 25px;
        border-radius: 15px;
        overflow: hidden;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        transition: width 1s ease-out;
        border-radius: 15px;
    }

    .genre-chip {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        color: white;
        animation: fadeIn 0.3s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

    .main-title {
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        to { text-shadow: 0 0 30px rgba(118, 75, 162, 0.8); }
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .sidebar .stMarkdown h2 {
        color: #7c3aed;
        border-bottom: 2px solid #7c3aed;
        padding-bottom: 5px;
    }

    .stProgress .st-bo {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    .example-lyrics {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #7c3aed;
        font-style: italic;
    }

    .stats-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    </style>
""", unsafe_allow_html=True)

# Load models
model, vectorizer = load_models()

# Header
st.markdown('<h1 class="main-title">🎵 Music Genre Predictor 🎵</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover the genre of any song using AI-powered lyric analysis</p>',
            unsafe_allow_html=True)

# Main content area
col1, col2, col3 = st.columns([1, 10, 1])

with col2:
    # Genre chips display
    genres = ['Pop', 'Hip-Hop', 'Rock', 'R&B', 'Alternative', 'K-pop', 'Rap']
    st.markdown("### 🎯 Supported Genres")

    # Create genre chips
    genre_html = ""
    for genre in genres:
        color = get_genre_color(genre)
        genre_html += f'<span class="genre-chip" style="background-color: {color};">{genre}</span>'

    st.markdown(genre_html, unsafe_allow_html=True)
    st.markdown("---")

    # Input area with better styling
    st.markdown("### 📝 Enter Your Lyrics")

    # Example lyrics button
    col_example1, col_example2, col_example3 = st.columns(3)

    example_lyrics = {
        "Pop Example": "I'm walking on sunshine, everything's alright\nDancing through the night, feeling so bright\nLove is in the air, nothing can compare",
        "Hip-Hop Example": "Started from the bottom now we here\nGrinding every day, making it clear\nMoney on my mind, success is near",
        "Rock Example": "Break these chains that hold me down\nScream it loud, that thunderous sound\nRebel heart won't be bound"
    }

    with col_example1:
        if st.button("📻 Pop Example", help="Load pop lyrics example"):
            st.session_state.lyrics_input = example_lyrics["Pop Example"]

    with col_example2:
        if st.button("🎤 Hip-Hop Example", help="Load hip-hop lyrics example"):
            st.session_state.lyrics_input = example_lyrics["Hip-Hop Example"]

    with col_example3:
        if st.button("🎸 Rock Example", help="Load rock lyrics example"):
            st.session_state.lyrics_input = example_lyrics["Rock Example"]

    # Text input
    lyrics_input = st.text_area(
        'Paste or type your lyrics here:',
        value=st.session_state.get('lyrics_input', ''),
        placeholder="🎵 Paste song lyrics here...\n\nExample:\n'I'm feeling good tonight\nDancing in the moonlight\nEverything's alright...'\n\nTip: The more lyrics you provide, the better the prediction!",
        height=200,
        help="Enter at least 3-4 lines of lyrics for better accuracy",
        key="lyrics_text"
    )

    # Button area with better layout
    st.markdown("### 🚀 Make Prediction")
    button_col1, button_col2, button_col3, button_col4 = st.columns([2, 2, 2, 2])

    with button_col2:
        predict_button = st.button('🎯 Predict Genre', type="primary", help="Analyze lyrics and predict genre")

    with button_col3:
        if st.button('🗑️ Clear Text', help="Clear the input area"):
            st.session_state.lyrics_input = ""
            st.rerun()

    # Prediction results
    if predict_button:
        if lyrics_input.strip() and model is not None and vectorizer is not None:
            # Add a progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text('🔄 Preprocessing lyrics...')
            progress_bar.progress(25)
            time.sleep(0.3)

            try:
                # Preprocess the lyrics
                cleaned = preprocess(lyrics_input)

                status_text.text('🧠 Analyzing with AI model...')
                progress_bar.progress(75)
                time.sleep(0.3)

                if not cleaned:
                    st.warning('⚠️ The lyrics contain mostly common words. Please try with more descriptive lyrics.')
                else:
                    # Vectorize and predict
                    vectorized = vectorizer.transform([cleaned])
                    prediction = model.predict(vectorized)[0]
                    probabilities = model.predict_proba(vectorized)[0]
                    confidence = probabilities.max()

                    status_text.text('✅ Prediction complete!')
                    progress_bar.progress(100)
                    time.sleep(0.5)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Get all class probabilities
                    classes = model.classes_
                    prob_dict = dict(zip(classes, probabilities))

                    # Main result with enhanced styling
                    genre_color = get_genre_color(prediction)
                    st.markdown(f"""
                    <div class="genre-result">
                        <h2>🎧 Predicted Genre: <span style="color: {genre_color}; font-weight: 800;">{prediction}</span></h2>
                        <p style="font-size: 1.1rem; margin-top: 10px;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Enhanced confidence bar
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Genre probabilities with better styling
                    st.markdown("### 📊 Detailed Analysis")

                    # Sort probabilities in descending order
                    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

                    # Create two columns for better layout
                    prob_col1, prob_col2 = st.columns(2)

                    for i, (genre, prob) in enumerate(sorted_probs):
                        col = prob_col1 if i % 2 == 0 else prob_col2
                        with col:
                            genre_color = get_genre_color(genre)
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid {genre_color};">
                                <strong style="color: {genre_color};">{genre}</strong>: {prob:.1%}
                            </div>
                            """, unsafe_allow_html=True)

                    # Additional insights
                    st.markdown("---")
                    col_insight1, col_insight2 = st.columns(2)

                    with col_insight1:
                        st.markdown(f"""
                        <div class="stats-card">
                            <h4>📈 Prediction Strength</h4>
                            <p>{'🔥 Very Confident' if confidence > 0.8 else '✅ Confident' if confidence > 0.6 else '⚠️ Moderate' if confidence > 0.4 else '❓ Uncertain'}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_insight2:
                        second_best = sorted_probs[1] if len(sorted_probs) > 1 else None
                        if second_best:
                            st.markdown(f"""
                            <div class="stats-card">
                                <h4>🎯 Alternative Possibility</h4>
                                <p>{second_best[0]}: {second_best[1]:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Debug info with better styling
                    with st.expander("🔍 Technical Details", expanded=False):
                        st.markdown(f"""
                        **Input Statistics:**
                        - Original length: {len(lyrics_input)} characters
                        - Processed length: {len(cleaned)} characters
                        - Words analyzed: {len(cleaned.split())} words

                        **Processed Text Preview:**
                        ```
                        {cleaned[:300]}{'...' if len(cleaned) > 300 else ''}
                        ```
                        """)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🚨 An error occurred during prediction: {str(e)}")
                st.error("Please check that your model files are compatible with the current setup.")

        elif not lyrics_input.strip():
            st.warning('🎤 Please paste some lyrics first!')
        else:
            st.error('❌ Model files could not be loaded. Please check the file paths.')

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 📚 How to Use")
    st.markdown("""
    1. **📝 Enter Lyrics:** Paste song lyrics in the text area
    2. **🎯 Predict:** Click 'Predict Genre' to analyze
    3. **📊 Review:** Check the predicted genre and confidence scores
    4. **🔍 Explore:** View all genre probabilities and insights
    """)

    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This AI-powered tool uses **machine learning** to analyze song lyrics and predict music genres.

    **🤖 Technology Stack:**
    - Random Forest Classifier
    - TF-IDF Text Vectorization  
    - Natural Language Processing
    - Trained on 5,900+ songs

    **🎯 Accuracy:** ~81% on test data
    """)

    st.markdown("## 🔧 Model Status")
    if model is not None and vectorizer is not None:
        st.success("✅ AI Model: Loaded")
        st.success("✅ Vectorizer: Loaded")
        st.info("🚀 Ready for predictions!")
    else:
        st.error("❌ Models not loaded")

    st.markdown("## 🎵 Quick Examples")
    st.markdown("""
    **Pop:** Love songs, upbeat themes, mainstream appeal

    **Hip-Hop:** Urban culture, rhythmic speech, social themes

    **Rock:** Rebellion, energy, guitar-driven themes

    **R&B:** Soulful, romantic, smooth vocals themes
    """)

    st.markdown("---")
    st.markdown("*Made with ❤️ using Streamlit*")