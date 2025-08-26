# 🎵 Music Genre Predictor

A machine learning web application that predicts music genres from song lyrics using AI.
Overview
This project analyzes song lyrics and predicts which genre a song belongs to from 7 categories: Pop, Hip-Hop, Rock, R&B, Alternative, K-pop, and Rap. Simply paste lyrics into the web interface and get instant predictions with confidence scores.
Features

AI-powered genre classification - Uses Random Forest machine learning model
7 genre categories - Pop, Hip-Hop, Rock, R&B, Alternative, K-pop, Rap
Confidence scoring - Shows how certain the prediction is
Web interface - Easy-to-use Streamlit app
Trained on real data - 5,900+ songs from 21 different artists

Quick Start
Requirements

Python 3.7 or higher
pip package manager

Installation
bash# Install required packages
pip install streamlit scikit-learn pandas numpy nltk

## Run the application
streamlit run app.py
Usage

Open your browser to http://localhost:8501
Paste song lyrics into the text area
Click "Predict Genre"
View your results with confidence scores

Example Output
🎯 Prediction: Hip-Hop (78% confidence)

Alternative possibilities:
- Rap: 15%
- Pop: 4% 
- R&B: 2%

## How It Works
The app uses natural language processing to:

Clean and preprocess the lyrics text
Convert text to numerical features using TF-IDF. 
Apply a Random Forest classifier to predict genre
Return prediction with confidence percentages

## Model Performance

Accuracy: 81.2% on test data,

Dataset: 5,900+ songs across 21 artists,

#### Algorithm: Random Forest with TF-IDF vectorization

# Technologies Used

Python - Core programming language,

Streamlit - Web application framework,

scikit-learn - Machine learning library,

NLTK - Natural language processing,

Pandas/NumPy - Data manipulation