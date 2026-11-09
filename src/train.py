import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
MAX_LENGTH = 512
BASE_MODEL = "distilbert-base-uncased"


def build_label_maps(train_path):
    products = pd.read_csv(train_path, usecols=["product"])["product"]
    labels = sorted(products.unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def load_split(path, label2id, sample_rows=None):
    df = pd.read_csv(path, usecols=["narrative", "product"])
    if sample_rows:
        df = df.sample(n=sample_rows, random_state=SEED)
    df["label"] = df["product"].map(label2id)
    return df[["narrative", "label"]]


def make_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["narrative"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    return dataset.map(tokenize_batch, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    return {"weighted_f1": weighted_f1}


def build_training_args(output_dir, epochs, batch_size):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted_f1",
        seed=SEED,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the complaint classifier.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model or checkpoint folder to start training from.",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models/train_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16, help="Per device.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Use only this many rows per split (a quick test run).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = f"{args.data_dir}/train.csv"
    val_path = f"{args.data_dir}/val.csv"

    label2id, _ = build_label_maps(train_path)
    train_df = load_split(train_path, label2id, args.sample_rows)
    val_df = load_split(val_path, label2id, args.sample_rows)

    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    train_ds = make_dataset(train_df, tokenizer)
    val_ds = make_dataset(val_df, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(label2id)
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(args.output_dir, args.epochs, args.batch_size),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("GPU used:", torch.cuda.is_available())
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print([x for x in trainer.state.log_history if "eval_weighted_f1" in x])


if __name__ == "__main__":
    main()