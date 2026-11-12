import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer

MODEL_DIR = "models/final_model"
TEST_CSV = "data/processed/test.csv"
MAX_LENGTH = 512
BATCH_SIZE = 1000


def token_counts(texts, tokenizer):
    counts = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        ids = tokenizer(batch, truncation=False)["input_ids"]
        counts.extend(len(item) for item in ids)
    return np.array(counts)


def main():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    narratives = pd.read_csv(TEST_CSV, usecols=["narrative"])["narrative"]
    counts = token_counts(narratives.astype(str).tolist(), tokenizer)
    over = int((counts > MAX_LENGTH).sum())
    print(f"Test complaints: {len(counts)}")
    print(f"Over {MAX_LENGTH} tokens: {over} ({over / len(counts) * 100:.1f}%)")
    print(f"Median length: {int(np.median(counts))}, longest: {counts.max()}")


if __name__ == "__main__":
    main()