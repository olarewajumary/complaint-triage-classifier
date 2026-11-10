import numpy as np
import pandas as pd
import torch

from src.predict import load_model

tokenizer, model, device = load_model()

TEST_CSV = "data/processed/test.csv"
TEST_PROBS = "data/processed/test_probs.npy"
SAMPLE_SIZE = 100
SEED = 0


def live_probs(text):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def main():
    saved = np.load(TEST_PROBS)
    texts = pd.read_csv(TEST_CSV, usecols=["narrative"])["narrative"]
    print(f"Saved probabilities: shape {saved.shape}, dtype {saved.dtype}")
    print(f"test.csv rows: {len(texts)}")

    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(texts), size=SAMPLE_SIZE, replace=False))

    live = np.array([live_probs(str(texts.iloc[i])) for i in rows])
    old = saved[rows]

    agree = int((live.argmax(axis=1) == old.argmax(axis=1)).sum())
    diff = np.abs(live - old)
    print(f"Same top class: {agree} of {SAMPLE_SIZE}")
    print(f"Mean abs probability difference: {diff.mean():.5f}")
    print(f"Max abs probability difference: {diff.max():.5f}")


if __name__ == "__main__":
    main()