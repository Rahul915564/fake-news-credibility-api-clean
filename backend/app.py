import os
import urllib.request
import zipfile
import joblib
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

# ================== APP ==================
app = Flask(__name__)
CORS(app)

# ================== PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "ml")
MODEL_PATH = os.path.join(MODEL_DIR, "model_v2.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer_v2.pkl")
ZIP_PATH = os.path.join(MODEL_DIR, "model_files.zip")

# ================== DOWNLOAD MODEL IF NOT EXISTS ==================
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("⬇️ Downloading ML model files...")

    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=10kF4Fbm2-zdEmY2FfZyzb5e7cHYkRKRp",
        ZIP_PATH
    )

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    print("✅ ML model files downloaded & extracted")

# ================== LOAD MODEL ==================
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ================== HELPERS ==================
def get_top_words(text, vectorizer, model, top_n=5):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    indices = vec.nonzero()[1]

    word_scores = [(feature_names[i], coef[i]) for i in indices]
    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)

    return [w for w, _ in word_scores[:top_n]]

# ================== API ==================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    title = data.get("title", "")
    text = data.get("text", "")

    if not title or not text:
        return jsonify({"error": "Title and text required"}), 400

    combined = title + " " + text
    vec = vectorizer.transform([combined])

    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    confidence = round(max(proba) * 100, 2)
    label = "REAL" if pred == 1 else "FAKE"

    if confidence >= 80:
        risk = "Low Risk"
    elif confidence >= 60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return jsonify({
        "label": label,
        "confidence": confidence,
        "risk": risk,
        "top_words": get_top_words(combined, vectorizer, model)
    })

# ================== RUN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
