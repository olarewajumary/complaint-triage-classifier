import matplotlib.pyplot as plt
import numpy as np

from evaluate import REPORT_DIR, load_results

TARGET_ACCURACY = 0.95
THRESHOLDS = np.arange(0.0, 1.0, 0.01)


def load_scores(split):
    data, predicted = load_results(split)
    confidence = np.load(f"data/processed/{split}_probs.npy").max(axis=1)
    correct = predicted == data["product"].values
    return confidence, correct


def at_threshold(confidence, correct, t):
    auto = confidence >= t
    human_share = 1 - auto.mean()
    auto_acc = correct[auto].mean() if auto.any() else np.nan
    return human_share, auto_acc


def sweep(confidence, correct):
    return [(t, *at_threshold(confidence, correct, t)) for t in THRESHOLDS]


def pick(rows):
    for t, human_share, auto_acc in rows:
        if auto_acc >= TARGET_ACCURACY:
            return t, human_share, auto_acc
    return None


def plot(rows, chosen):
    human = [r[1] * 100 for r in rows]
    acc = [r[2] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(human, acc)
    ax.scatter(rows[50][1] * 100, rows[50][2] * 100, color="red", zorder=3, label="threshold 0.50")
    if chosen:
        ax.scatter(chosen[1] * 100, chosen[2] * 100, color="green", zorder=3, label=f"chosen {chosen[0]:.2f}")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Complaints sent to a human (%)")
    ax.set_ylabel("Accuracy on the rest (%)")
    ax.set_title("Accuracy vs human workload (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "threshold_tradeoff.png", dpi=150)
    print("Saved", REPORT_DIR / "threshold_tradeoff.png")


if __name__ == "__main__":
    val_conf, val_correct = load_scores("val")
    print("Validation accuracy, no threshold:", round(val_correct.mean(), 4))

    rows = sweep(val_conf, val_correct)
    _, human50, acc50 = rows[50]
    print(f"Threshold 0.50: {human50:.1%} to a human, {acc50:.1%} accurate on the rest")

    chosen = pick(rows)
    if chosen is None:
        print("No threshold reaches the target on validation")
    else:
        t, human, acc = chosen
        print(f"Threshold {t:.2f}: {human:.1%} to a human, {acc:.1%} accurate on the rest")
        test_conf, test_correct = load_scores("test")
        human_t, acc_t = at_threshold(test_conf, test_correct, t)
        print(f"Same threshold on test: {human_t:.1%} to a human, {acc_t:.1%} accurate on the rest")
    plot(rows, chosen)