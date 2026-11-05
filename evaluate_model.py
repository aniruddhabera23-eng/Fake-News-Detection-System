
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")


model = load_model("models/bilstm_fake_news_model.keras")

print("Evaluating BiLSTM Model...")


y_prob = model.predict(X_test)


y_pred = (y_prob >= 0.5).astype(int).flatten()


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Fake News", "Real News"],
    zero_division=0
))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("BiLSTM Fake News Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved successfully!")
