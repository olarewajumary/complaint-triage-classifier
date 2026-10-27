import pandas as pd


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(f"data/processed/{name}.csv")


def normalize(texts: pd.Series) -> pd.Series:
    return texts.str.strip().str.lower()


def flag_duplicates(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    train_texts = set(normalize(train["narrative"]))
    return normalize(test["narrative"]).isin(train_texts)


if __name__ == "__main__":
    train = load_split("train")
    test = load_split("test")
    flags = flag_duplicates(train, test)
    print("Test rows:", len(test))
    print("Test rows whose text also appears in train:", flags.sum())
    print("Share:", round(flags.mean() * 100, 2), "%")