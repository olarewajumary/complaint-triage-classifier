import pandas as pd
from confusion_matrix_plot import SHORT_NAMES

from evaluate import load_results


def main_mistake_per_class(y_true, y_pred) -> pd.DataFrame:
    pairs = pd.DataFrame({"actual": y_true.values, "predicted": y_pred})
    wrong = pairs[pairs["actual"] != pairs["predicted"]]
    counts = wrong.value_counts().reset_index(name="count")
    totals = pairs["actual"].value_counts()
    counts["share_of_actual"] = (counts["count"] / counts["actual"].map(totals)).round(3)
    top = counts.drop_duplicates("actual")
    return top.sort_values("share_of_actual", ascending=False)


if __name__ == "__main__":
    test, predicted = load_results()
    table = main_mistake_per_class(test["product"], predicted)
    table["actual"] = table["actual"].map(SHORT_NAMES)
    table["predicted"] = table["predicted"].map(SHORT_NAMES)
    print(table.to_string(index=False))