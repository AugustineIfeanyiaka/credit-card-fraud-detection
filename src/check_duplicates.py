
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "creditcard.csv"

df = pd.read_csv(data_path)

# Identify repeated records beyond the first occurrence
duplicates = df[df.duplicated()]

print("===== DUPLICATE ANALYSIS =====")
print("Total duplicate rows:", len(duplicates))

print("\nDuplicate rows by class:")
print(duplicates["Class"].value_counts())

print("\nOriginal class distribution:")
print(df["Class"].value_counts())

# Preview what removing exact duplicates would change
deduplicated = df.drop_duplicates()

print("\nClass distribution after removing duplicates:")
print(deduplicated["Class"].value_counts())

print("\nDataset size after removing duplicates:")
print(deduplicated.shape)
