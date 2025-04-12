import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os

def train_model(features_path, labels_path, model_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).values.ravel()  # Ensure y is 1D
    model = LogisticRegression()
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pd.to_pickle(model, model_path)

if __name__ == "__main__":
    train_model(
        features_path="data/features.csv",
        labels_path="data/labels.csv",
        model_path="models/model.pkl"
    )