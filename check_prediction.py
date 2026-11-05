
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model(
    "models/bilstm_fake_news_model.keras"
)

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load a genuine article from processed dataset
data = pd.read_csv("dataset/processed_news.csv")
real_news = data[data["label"] == 1]["content"].iloc[0]


sequence = tokenizer.texts_to_sequences([real_news])
padded = pad_sequences(
    sequence,
    maxlen=300,
    padding="post",
    truncating="post"
)

# Predict
prediction = float(model.predict(padded, verbose=0)[0][0])

print("Actual label: REAL")
print("Model probability:", prediction)
print("Predicted label:", "REAL" if prediction >= 0.5 else "FAKE")
