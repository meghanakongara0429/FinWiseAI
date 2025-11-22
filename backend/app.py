from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load model & vectorizer
model = joblib.load("saved_model/classifier.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")

app = FastAPI(title="FinWiseAI Backend")

# Allow all origins (frontend can connect easily)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FinWiseAI API is running!"}

@app.post("/predict")
async def predict_category(text: str):
    """
    Predict category for a single transaction sentence
    """
    transformed = vectorizer.transform([text])
    pred = model.predict(transformed)[0]
    return {"transaction": text, "predicted_category": pred}


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict categories for CSV file containing a 'transaction' column
    """
    df = pd.read_csv(file.file)

    if "transaction" not in df.columns:
        return {"error": "CSV must contain 'transaction' column"}

    transformed = vectorizer.transform(df["transaction"])
    df["predicted_category"] = model.predict(transformed)

    return df.to_dict(orient="records")
