import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")

vectorizer = TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 2),
    stop_words="english",
)

X_train = vectorizer.fit_transform(train_df["narrative"])
X_val = vectorizer.transform(val_df["narrative"])

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

model.fit(X_train, train_df["product"])

val_predictions = model.predict(X_val)

print(classification_report(val_df["product"], val_predictions))

print("Training complete")

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")