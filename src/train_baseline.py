import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from check_duplicates import flag_duplicates

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

test_df = pd.read_csv("data/processed/test.csv")
X_test = vectorizer.transform(test_df["narrative"])
test_predictions = model.predict(X_test)

clean = ~flag_duplicates(train_df, test_df).values

report_all = classification_report(test_df["product"], test_predictions)
report_clean = classification_report(test_df["product"][clean], test_predictions[clean])

print("TEST, all rows")
print(report_all)
print("TEST, rows not seen in train")
print(report_clean)

with open("data/processed/baseline_test_report.txt", "w") as f:
    f.write("TEST, all rows\n" + report_all + "\nTEST, rows not seen in train\n" + report_clean)