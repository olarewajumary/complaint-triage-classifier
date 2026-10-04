# Consumer Complaint Triage Classifier

Automatically classifies consumer financial complaints into one of 10 product categories, using a fine-tuned DistilBERT model, built to route real support tickets to the right team faster than manual sorting.

## Problem

Financial institutions receive large volumes of consumer complaints that need to be routed to the correct team quickly. Misrouted or delayed complaints carry real compliance risk. This project builds and compares two approaches to automatic classification.

## Data

~380K real consumer complaints from the CFPB Consumer Complaint Database (public, U.S. Government Work). Note: CFPB discontinued publishing complaint narratives in their live database on August 14, 2026. This project uses a historical snapshot (2011–2019) from a well-established public mirror, since the narrative field is essential to the task and is no longer available from the live source.

## Approach

1. **Baseline:** TF-IDF + Logistic Regression, class-weighted for severe category imbalance (424:1 ratio between largest and smallest category).
2. **Fine-tuned model:** DistilBERT, fine-tuned for 3 epochs, with the final model selected by validation performance rather than simply the last epoch, after confirming epoch 3 showed early signs of overfitting.

## Results

| Model | Weighted F1 (test set) |
|---|---|
| TF-IDF + Logistic Regression | 0.82 |
| Fine-tuned DistilBERT | **0.87** |

## Limitations

- Performance is weakest on the three smallest categories by data volume (Consumer Loan, Payday/title loan, Vehicle loan/lease), F1 in the 0.58–0.59 range, directly reflecting limited training data for those classes.
- ~11.6% of complaints exceed the model's 512-token input limit and are truncated; information near the end of very long complaints may not be seen by the model.
- Confidence threshold (0.5) for the human-review fallback is a reasonable starting point, not independently tuned/validated.

## Usage

```python
from src.predict import predict

result = predict("My mortgage servicer applied my payment to the wrong month...")
# {'prediction': 'Mortgage', 'confidence': 0.658}
```

Low-confidence predictions return `needs_human_review` instead of a forced guess. Invalid input (empty, too short, wrong type) returns a clear error rather than crashing.

## Reproducing the model

Trained weights aren't committed to this repo (GitHub's 100MB file limit, and they're fully reproducible from the pipeline below).

1. Run `src/download_data.py`, then `src/split_data.py` to prepare the data
2. Fine-tune `distilbert-base-uncased` on `data/processed/train.csv`
3. Save the resulting model to `models/final_model/`