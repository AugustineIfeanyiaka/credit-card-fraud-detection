
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("After deduplication:", df.shape)

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Stratified split preserves the rare fraud class proportion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\n===== TRAINING CLASS DISTRIBUTION =====")
print(y_train.value_counts())

print("\n===== TEST CLASS DISTRIBUTION =====")
print(y_test.value_counts())

print("\nTraining fraud rate:", f"{y_train.mean() * 100:.4f}%")
print("Testing fraud rate:", f"{y_test.mean() * 100:.4f}%")
