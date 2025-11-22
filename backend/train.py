import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import json

# Load dataset
df = pd.read_csv("../database/transactions_dataset.csv")

# Clean column names
df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

# Rename for consistency
if "description" not in df.columns:
    df.rename(columns={"transaction": "description"}, inplace=True)

if "category" not in df.columns:
    raise ValueError("Dataset must contain a `Category` column")

# Features & Labels
X = df["description"].astype(str)
y = df["category"].astype(str)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "saved_model/classifier.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")

print("\n🎉 Model training completed!")
print("Saved: saved_model/classifier.pkl & vectorizer.pkl")
