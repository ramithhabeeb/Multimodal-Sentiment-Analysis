import streamlit as st
from transformers import pipeline

st.title("Review Classification")

classifier = pipeline('text-classification', model='distilbert-base-uncased-sentiment-model')

text = st.text_area("Enter Your Review Here")

if st.button("Predict"):
        result = classifier(text)[0]  # Get the first prediction (it's a single text input)
        label = result['label']
        score = result['score']

        # Display the results in a more readable format
        st.subheader("Prediction Result")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence Score:** {score:.2f}")