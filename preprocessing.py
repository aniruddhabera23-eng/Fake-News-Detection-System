
import pandas as pd
import re

# Load datasets
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Assign labels
true_news["label"] = 1
fake_news["label"] = 0

# Combine datasets
data = pd.concat([true_news, fake_news], ignore_index=True)

print("Original Dataset Size:", len(data))

# Remove duplicate news articles
data.drop_duplicates(subset=["title", "text"], inplace=True)

# Remove missing values
data.dropna(subset=["title", "text"], inplace=True)

print("After Removing Duplicates:", len(data))

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply text cleaning
data["content"] = data["content"].apply(clean_text)

# Remove empty articles
data = data[data["content"].str.strip() != ""]

# Save processed dataset
data.to_csv("dataset/processed_news.csv", index=False)

print("\nTEXT PREPROCESSING COMPLETED!")
print("Final Dataset Size:", len(data))

print("\nSample Cleaned News:")
print(data["content"].iloc[0][:500])

print("\nFinal Label Distribution:")
print(data["label"].value_counts())

print("\nProcessed dataset saved successfully!")
