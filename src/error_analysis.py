import pandas as pd
import numpy as np
from confusion_matrix_plot import SHORT_NAMES
from check_duplicates import load_split

from evaluate import load_results


def main_mistake_per_class(y_true, y_pred) -> pd.DataFrame:
    pairs = pd.DataFrame({"actual": y_true.values, "predicted": y_pred})
    wrong = pairs[pairs["actual"] != pairs["predicted"]]
    counts = wrong.value_counts().reset_index(name="count")
    totals = pairs["actual"].value_counts()
    counts["share_of_actual"] = (counts["count"] / counts["actual"].map(totals)).round(3)
    top = counts.drop_duplicates("actual")
    return top.sort_values("share_of_actual", ascending=False)


def show_examples(test, predicted, actual, guessed, n=4):
    confidence = np.load("data/processed/test_probs.npy").max(axis=1)
    mask = (test["product"].values == actual) & (predicted == guessed)
    rows = test[mask].assign(confidence=confidence[mask])
    for _, row in rows.sample(min(n, len(rows)), random_state=42).iterrows():
        print(f"[{row['date_received']}] sub_product={row['sub_product']} confidence={row['confidence']:.2f}")
        print(row["narrative"][:400], "\n")


def date_ranges(df) -> pd.DataFrame:
    dates = pd.to_datetime(df["date_received"])
    summary = dates.groupby(df["product"]).agg(["min", "max", "count"])
    return summary.rename(index=SHORT_NAMES).sort_values("min")


def subproduct_mix(df, product) -> pd.Series:
    rows = df.loc[df["product"] == product, "sub_product"]
    return rows.value_counts(dropna=False)


if __name__ == "__main__":
    test, predicted = load_results()
    table = main_mistake_per_class(test["product"], predicted)
    table["actual"] = table["actual"].map(SHORT_NAMES)
    table["predicted"] = table["predicted"].map(SHORT_NAMES)
    print(table.to_string(index=False))
    print()
    show_examples(test, predicted, "Vehicle loan or lease", "Consumer Loan")
    print()
    print(date_ranges(load_split("train")).to_string())
    print()
    print(subproduct_mix(load_split("train"), "Consumer Loan").to_string())