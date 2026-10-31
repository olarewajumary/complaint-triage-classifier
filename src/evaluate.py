from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from check_duplicates import flag_duplicates, load_split

REPORT_DIR = Path("reports")


def load_results(split="test"):
    data = load_split(split)
    probs = np.load(f"data/processed/{split}_probs.npy")
    labels = np.array(sorted(data["product"].unique()))
    predicted = labels[probs.argmax(axis=1)]
    return data, predicted


def summarize(y_true, y_pred, name):
    weighted = f1_score(y_true, y_pred, average="weighted")
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"{name}: weighted F1 {weighted:.4f}, macro F1 {macro:.4f}, rows {len(y_true)}")


if __name__ == "__main__":
    test, predicted = load_results()
    train = load_split("train")
    clean = ~flag_duplicates(train, test).values

    summarize(test["product"], predicted, "All test rows")
    summarize(test["product"][clean], predicted[clean], "Rows not seen in train")

    report = classification_report(test["product"], predicted, output_dict=True)
    table = pd.DataFrame(report).T.round(3)
    REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORT_DIR / "per_class_metrics.csv")
    print(table)