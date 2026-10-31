
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "creditcard.csv"
OUTPUT = ROOT / "data" / "demo_transactions.csv"

df = pd.read_csv(SOURCE)

# Select ten legitimate and ten fraudulent examples
legitimate = df[df["Class"] == 0].head(10)
fraudulent = df[df["Class"] == 1].head(10)

demo = pd.concat(
    [legitimate, fraudulent],
    ignore_index=True
)

demo.to_csv(OUTPUT, index=False)

print("Demo dataset created:", OUTPUT)
print("Shape:", demo.shape)
print("Class distribution:")
print(demo["Class"].value_counts())
