
import pandas as pd
import numpy as np
import os

os.makedirs("models", exist_ok=True)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load processed dataset
data = pd.read_csv("dataset/processed_news.csv")

# Input text and labels
X = data["content"].astype(str)
y = data["label"]

# Tokenization
max_words = 20000
max_len = 300

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

# Convert text into numerical sequences
sequences = tokenizer.texts_to_sequences(X)

# Apply padding
X_padded = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save tokenizer
import pickle

with open("models/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Save processed arrays
np.save("models/X_train.npy", X_train)
np.save("models/X_test.npy", X_test)
np.save("models/y_train.npy", y_train)
np.save("models/y_test.npy", y_test)

print("NLP TOKENIZATION COMPLETED!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input shape:", X_train.shape)
print("Vocabulary size:", len(tokenizer.word_index))
