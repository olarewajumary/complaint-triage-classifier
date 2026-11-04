import os

import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.path.join("models", "final_model")
MAX_LENGTH = 512


def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    print(f"After imports: {memory_mb():.0f} MB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"After tokenizer: {memory_mb():.0f} MB")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    print(f"After model load: {memory_mb():.0f} MB")

    long_text = "my credit report has a mistake and nobody will fix it " * 60
    inputs = tokenizer(
        long_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    print(f"Token count fed to model: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        model(**inputs)
    print(f"After one prediction: {memory_mb():.0f} MB")

    info = psutil.Process(os.getpid()).memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak:
        print(f"Peak during the run: {peak / (1024 * 1024):.0f} MB")


if __name__ == "__main__":
    main()