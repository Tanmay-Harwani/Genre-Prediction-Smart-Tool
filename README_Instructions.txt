
Music Genre Prediction from Lyrics - Instructions

This project predicts the genre of a song using only its lyrics.

Project Folder Structure:
--------------------------
- archive/               -> Folder containing original CSV files (lyrics by artist)
- saved_model/           -> Stores the trained model and TF-IDF vectorizer
- app.py                 -> Streamlit web application
- model_training.py      -> Script to clean data, train model, and save it
- combine_csv's.py       -> Combines multiple CSVs into one dataset
- lyrics_dataset.csv     -> Output of the combined and cleaned lyrics data

How to Run the Project:
--------------------------
Step 1: Combine the CSV Files
    Run:
    python combine_csv's.py

Step 2: Train the Model
    Run:
    python model_training.py
    This will create and save the model and vectorizer inside saved_model/

Step 3: Run the Web App
    Run:
    streamlit run app.py
    This launches the web interface for genre prediction

Dependencies:
---------------
- Python
- spaCy
- scikit-learn
- Streamlit
- joblib

Make sure to activate your virtual environment if using one before running these commands.
