
import streamlit as st
import pickle
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model(
    "models/bilstm_fake_news_model.keras"
)


with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LEN = 300

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 AI-Based Fake News Detection System")
st.write("Enter a news article to check whether it is Fake or Real.")

news = st.text_area("Enter News Article:", height=200)

if st.button("Detect News"):
    if news.strip():
        cleaned = clean_text(news)

        st.write("Cleaned text:", cleaned[:300])

        sequence = tokenizer.texts_to_sequences([cleaned])

        st.write("Token sequence:", sequence[0][:30])

        padded = pad_sequences(
            sequence,
            maxlen=300,
            padding="post",
            truncating="post"
        )

        probability = float(
            model.predict(padded, verbose=0)[0][0]
        )

        st.write("Raw prediction probability:", probability)

        if probability >= 0.5:
            st.success("REAL NEWS")
        else:
            st.error("FAKE NEWS")

    else:
        st.warning("Please enter a news article.")