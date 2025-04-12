import pandas as pd
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

nltk.download('stopwords')

def preprocess_data(input_path, output_path):
    try:
        # Read input CSV
        print(f"Reading {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Columns found: {df.columns.tolist()}")
        
        # Verify expected columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Expected 'text' and 'label' columns in CSV")

        # Vectorize text
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(df['text']).toarray()
        y = df['label']

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        model_dir = os.path.join(output_path, '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Save features and labels
        print("Saving features and labels...")
        feature_path = os.path.join(output_path, 'features.csv')
        label_path = os.path.join(output_path, 'labels.csv')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        pd.DataFrame(X).to_csv(feature_path, index=False)
        pd.Series(y).to_csv(label_path, index=False)
        pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

        print(f"Saved: {feature_path}, {label_path}, {vectorizer_path}")
        return vectorizer
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data("data/imdb_small.csv", "data")