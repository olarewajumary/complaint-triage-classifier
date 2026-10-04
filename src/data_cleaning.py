import pandas as pd

CATEGORY_CONSOLIDATION = {
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Bank account or service": "Checking or savings account",
}

MIN_CATEGORY_SIZE = 1000
MIN_WORD_COUNT = 10


def load_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["product"] = df["product"].replace(CATEGORY_CONSOLIDATION)

    category_counts = df["product"].value_counts()
    rare_categories = category_counts[category_counts < MIN_CATEGORY_SIZE].index.tolist()
    df = df[~df["product"].isin(rare_categories)]

    df["narrative_word_count"] = df["narrative"].str.split().str.len()
    df = df[df["narrative_word_count"] >= MIN_WORD_COUNT]

    return df