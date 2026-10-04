import pandas as pd
from data_cleaning import load_clean_data

df = load_clean_data("data/raw/complaints.csv")
df["narrative_length"] = df["narrative"].str.len()

summary_lines = []
summary_lines.append(f"Total complaints: {len(df)}")
summary_lines.append("")
summary_lines.append("Complaints per product:")
summary_lines.append(df["product"].value_counts().to_string())
summary_lines.append("")
summary_lines.append("Narrative length (characters):")
summary_lines.append(df["narrative_length"].describe().to_string())

summary_text = "\n".join(summary_lines)
print(summary_text)

print()
print("Shortest narratives (checking for junk/placeholder text):")
print(df.nsmallest(10, "narrative_length")[["narrative", "narrative_length"]].to_string())

with open("data/processed/eda_summary.txt", "w") as f:
    f.write(summary_text)