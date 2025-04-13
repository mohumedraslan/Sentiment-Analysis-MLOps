import pickle
import numpy as np

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

texts = ["i love this movie!", "i hate this movie"]
for text in texts:
    X = vectorizer.transform([text]).toarray()
    X_array = np.array(X)
    prediction = model.predict(X_array)[0]
    print(f"Text: {text}, Prediction: {'positive' if prediction == 1 else 'negative'}")