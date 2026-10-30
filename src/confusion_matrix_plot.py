import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from evaluate import REPORT_DIR, load_results

SHORT_NAMES = {
    "Checking or savings account": "Checking/savings",
    "Consumer Loan": "Consumer loan",
    "Credit card or prepaid card": "Credit/prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Mortgage": "Mortgage",
    "Payday loan, title loan, or personal loan": "Payday/title loan",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan/lease",
}


def plot_confusion(y_true, y_pred) -> None:
    labels = sorted(SHORT_NAMES)
    names = [SHORT_NAMES[label] for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Share of each actual category predicted as each category")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    print("Saved", REPORT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    test, predicted = load_results()
    plot_confusion(test["product"], predicted)