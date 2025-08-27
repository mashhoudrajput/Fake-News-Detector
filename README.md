Fake News Detector
==================

A Flask-based web application that uses machine learning to classify news articles as "Real" or "Fake." Built using Python, scikit-learn, and TF-IDF vectorization, this project aims to provide an accessible interface for detecting misinformation.

Features
--------

- Web Interface: Simple and intuitive UI built with Flask and HTML.
- Machine Learning Model: Logistic Regression classifier trained on a balanced dataset of real and fake news.
- TF-IDF Vectorization: Utilizes n-grams and stopword removal for effective text representation.
- Threshold-Based Prediction: Labels news as "Fake" if the model's confidence is below 90% for "Real."
- Model and Vectorizer: Pre-trained `model.joblib` and `vectorizer.joblib` files included for immediate use.

Installation
------------

Clone the repository:

    git clone https://github.com/mashhoudrajput/Fake-News-Detector.git
    cd Fake-News-Detector

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

    pip install -r requirements.txt

Run the application:

    python app.py

The app will be accessible at http://localhost:5000.

Usage
-----

1. Open the application in your browser.
2. Paste a news article into the text area.
3. Click "Check" to classify the news as "Real" or "Fake."

Example Input:

"On November 5, 2024, the United States held its presidential election, with voter turnout reaching one of the highest levels in recent decades."

Expected Output:

- Prediction: Real
- Probability of being Fake: 12.34%
- Probability of being Real: 87.66%

Model Training
--------------

To retrain the model with your own dataset:

1. Prepare two CSV files: `fake.csv` and `real.csv` in the `data/` directory, each containing a `text` column with news articles.
2. Run the training script:

    python train.py

This will generate `model.joblib` and `vectorizer.joblib` files for use in the application.

Project Structure
-----------------

Fake-News-Detector/
├── app.py                # Flask application
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── model.joblib          # Pre-trained model
├── vectorizer.joblib     # TF-IDF vectorizer
├── data/                 # Dataset directory
│   ├── fake.csv          # Fake news samples
│   └── real.csv          # Real news samples
└── templates/
    └── index.html        # Frontend template

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.
