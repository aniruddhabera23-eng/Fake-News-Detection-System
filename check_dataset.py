
import pandas as pd

# Load real and fake news datasets
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Display dataset information
print("REAL NEWS DATASET")
print(true_news.head())

print("\nFAKE NEWS DATASET")
print(fake_news.head())

print("\nReal News Shape:", true_news.shape)
print("Fake News Shape:", fake_news.shape)

print("\nReal News Columns:", true_news.columns.tolist())
print("Fake News Columns:", fake_news.columns.tolist())
