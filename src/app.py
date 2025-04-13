import streamlit as st
import requests
import urllib.parse
import os

st.title("Sentiment Analysis API")
st.write("Enter a movie review to predict its sentiment!")

text = st.text_area("Review", "I love this movie!")
if st.button("Predict"):
    # Check if running on Streamlit Cloud
    if "STREAMLIT_CLOUD" in os.environ:
        st.warning("This is a demo on Streamlit Cloud. The FastAPI backend is not available.")
        # Mock prediction for demo purposes
        mock_sentiment = "positive" if "love" in text.lower() else "negative"
        st.success(f"Sentiment: {mock_sentiment.capitalize()} (Mock prediction)")
    else:
        try:
            encoded_text = urllib.parse.quote(text)
            url = f"http://localhost:8000/predict?text={encoded_text}"
            response = requests.post(url)
            response.raise_for_status()
            result = response.json()
            if "sentiment" in result:
                st.success(f"Sentiment: {result['sentiment'].capitalize()}")
            else:
                st.error(f"Unexpected response: {result}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {str(e)}")

st.markdown("Built by [mohamed_raslan](https://github.com/moraslan202) | Powered by FastAPI & Docker")