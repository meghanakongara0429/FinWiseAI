import joblib
from prediction.preprocessing import clean_text, tokenize
import json

# Load model
model = joblib.load("saved_model/classifier.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")

# Load taxonomy
taxonomy = json.load(open("taxonomy.json"))

def predict_category(text):
    text_clean = tokenize(clean_text(text))
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)[0]

    confidence = max(model.predict_proba(vector)[0])
    return prediction, float(confidence)
