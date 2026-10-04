from pathlib import Path

import pandas as pd

CHUNK_SIZE = 50_000

COLUMN_MAP = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Submitted via": "submitted_via",
    "Complaint ID": "complaint_id",
}

def filter_and_save(csv_path: Path, date_from: str, date_to: str, out_path: Path) -> None:
    reader = pd.read_csv(
        csv_path,
        usecols=list(COLUMN_MAP.keys()),
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    kept_chunks = []
    total_seen = 0
    for chunk in reader:
        total_seen += len(chunk)
        chunk = chunk.rename(columns=COLUMN_MAP)
        chunk = chunk[chunk["narrative"].notna()]
        chunk["date_received"] = pd.to_datetime(chunk["date_received"], format="%m/%d/%Y")
        chunk = chunk[
            (chunk["date_received"] >= date_from)
            & (chunk["date_received"] <= date_to)
        ]
        if len(chunk):
            kept_chunks.append(chunk)
        print(f"  scanned {total_seen:,} rows, kept {sum(len(c) for c in kept_chunks):,}", end="\r")

    print()
    df = pd.concat(kept_chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} filtered complaints to {out_path}")

if __name__ == "__main__":
    date_from = "2011-01-01"
    date_to = "2019-12-31"
    csv_path = Path("data/raw/consumer_complaints_kaggle.csv")
    out_path = Path("data/raw/complaints.csv")

    filter_and_save(csv_path, date_from, date_to, out_path)