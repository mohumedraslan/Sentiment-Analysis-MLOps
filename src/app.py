import streamlit as st
import requests
import urllib.parse

st.title("Sentiment Analysis API")
st.write("Enter a movie review to predict its sentiment!")

text = st.text_area("Review", "I love this movie!")
if st.button("Predict"):
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