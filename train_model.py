import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import os


def preprocess_text(text):
    """
    Simple text preprocessing function
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Basic stop word removal
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


def train_model():
    """
    Train the genre prediction model
    """
    print("🔄 Loading dataset...")

    # Load the dataset
    try:
        df = pd.read_csv('lyrics_dataset.csv')
        print(f"✅ Dataset loaded with {len(df)} samples")
    except FileNotFoundError:
        print("❌ lyrics_dataset.csv not found. Please run the combine_csvs.py first.")
        return False

    # Check the data
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Genre distribution:")
    print(df['Genre'].value_counts())

    # Preprocess the lyrics
    print("🔄 Preprocessing lyrics...")
    df['Processed_Lyrics'] = df['Lyric'].apply(preprocess_text)

    # Remove empty processed lyrics
    df = df[df['Processed_Lyrics'].str.len() > 10]  # Keep only lyrics with more than 10 characters
    print(f"✅ After preprocessing: {len(df)} samples")

    # Prepare features and target
    X = df['Processed_Lyrics']
    y = df['Genre']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")

    # Create and train the model
    print("🔄 Training model...")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    # Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=20,
        min_samples_split=5
    )

    # Fit vectorizer and model
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    print("🔄 Evaluating model...")
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.3f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create saved_model directory if it doesn't exist
    os.makedirs('saved_model', exist_ok=True)

    # Save the model and vectorizer
    print("💾 Saving model and vectorizer...")
    joblib.dump(model, 'saved_model/model.pkl')
    joblib.dump(vectorizer, 'saved_model/vectorizer.pkl')

    print("✅ Model and vectorizer saved successfully!")
    print("🚀 You can now run the Streamlit app!")

    return True


if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n🎉 Model training completed successfully!")
        print("📁 Files saved in 'saved_model/' directory")
        print("🚀 Run: streamlit run app.py")
    else:
        print("\n❌ Model training failed!")