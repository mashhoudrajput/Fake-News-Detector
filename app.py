# app.py
from flask import Flask, request, render_template
import joblib
import re
import os

MODEL_PATH = "model.joblib"
VECT_PATH = "vectorizer.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError("model.joblib and vectorizer.joblib must exist. Run train.py first.")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

app = Flask(__name__)

# Preprocessing function (same as train.py ✅)
def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, prob_fake=None, prob_real=None, text="", warning=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news", "")
    if not text.strip():
        return render_template(
            "index.html",
            prediction="(no input)",
            prob_fake=None,
            prob_real=None,
            text=text,
            warning="⚠️ Please enter some text for analysis."
        )

    clean = simple_preprocess(text)
    vect = vectorizer.transform([clean])

    # Default prediction
    prediction = model.predict(vect)[0]

    prob_fake, prob_real = None, None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vect)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probs))
        prob_fake = round(prob_dict.get("fake", 0) * 100, 2)
        prob_real = round(prob_dict.get("real", 0) * 100, 2)

        # ✅ fair rule: pick whichever is higher
        prediction = "Real" if prob_real > prob_fake else "Fake"

    # Warning for very short text
    warning = None
    if len(text.split()) < 5:
        warning = "⚠️ Text is very short, results may not be accurate."

    return render_template(
        "index.html",
        prediction=prediction,
        prob_fake=prob_fake,
        prob_real=prob_real,
        text=text,
        warning=warning
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

