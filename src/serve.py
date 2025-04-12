from fastapi import FastAPI
import pickle
import numpy as np
import os

app = FastAPI()
model = pickle.load(open(os.path.join("models", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join("models", "vectorizer.pkl"), "rb"))

@app.post("/predict")
async def predict(text: str):
    X = vectorizer.transform([text]).toarray()
    X_array = np.array(X)  # Ensure array format
    prediction = model.predict(X_array)[0]
    return {"sentiment": "positive" if prediction == 1 else "negative"}