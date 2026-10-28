import numpy as np

from check_duplicates import load_split


def check() -> None:
    probs = np.load("data/processed/test_probs.npy")
    test = load_split("test")
    labels = np.array(sorted(test["product"].unique()))
    predicted = labels[probs.argmax(axis=1)]

    print("Shape:", probs.shape)
    print("Rows match test.csv:", len(probs) == len(test))
    print("Rows are probabilities:", np.allclose(probs.sum(axis=1), 1, atol=1e-3))
    print("Accuracy:", round((predicted == test["product"].values).mean(), 4))


if __name__ == "__main__":
    check()