# train.py
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# ---------- Config ----------
DATA_DIR = "data"
MODEL_PATH = "model.joblib"
VECT_PATH = "vectorizer.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 100000
# ----------------------------

def load_dataset(data_dir=DATA_DIR):
    fake_path = os.path.join(data_dir, "fake.csv")
    real_path = os.path.join(data_dir, "real.csv")
    if os.path.exists(fake_path) and os.path.exists(real_path):
        fake = pd.read_csv(fake_path)[['text']].dropna().copy()
        fake['label'] = 'fake'
        real = pd.read_csv(real_path)[['text']].dropna().copy()
        real['label'] = 'real'

        df = pd.concat([fake, real], ignore_index=True)
        print("Before balancing:\n", df['label'].value_counts())

        # Balance dataset (downsample larger class)
        fake_df = df[df['label'] == 'fake']
        real_df = df[df['label'] == 'real']
        min_size = min(len(fake_df), len(real_df))
        fake_bal = resample(fake_df, replace=False, n_samples=min_size, random_state=RANDOM_STATE)
        real_bal = resample(real_df, replace=False, n_samples=min_size, random_state=RANDOM_STATE)
        df = pd.concat([fake_bal, real_bal]).sample(frac=1, random_state=RANDOM_STATE)

        print("After balancing:\n", df['label'].value_counts())
        return df
    else:
        raise FileNotFoundError("Provide data/fake.csv and data/real.csv")

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)  # ✅ only keep letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("Loading dataset...")
    df = load_dataset()
    df['text'] = df['text'].astype(str).apply(simple_preprocess)
    X_text = df['text'].values
    y = df['label'].values

    print(f"Dataset loaded: {len(df)} rows")
    print("Label distribution:\n", df['label'].value_counts())

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=(1,2), stop_words='english'
    )
    X = vectorizer.fit_transform(X_text)
    print(f"TF-IDF matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred))

    print(f"Saving model to {MODEL_PATH} and vectorizer to {VECT_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()

