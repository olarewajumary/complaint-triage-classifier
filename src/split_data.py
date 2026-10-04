import pandas as pd
from sklearn.model_selection import train_test_split
from data_cleaning import load_clean_data

df = load_clean_data("data/raw/complaints.csv")

print(f"Clean dataset: {len(df)} rows")

print(f"Clean dataset: {len(df)} rows")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["product"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["product"],
    random_state=42,
)

print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)