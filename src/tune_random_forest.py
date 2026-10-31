
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Use the same preprocessing and training partition
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("Generating Random Forest out-of-fold scores...")

scores = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

thresholds = [
    0.10, 0.20, 0.30, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.70, 0.80, 0.90
]

print("\n===== RANDOM FOREST THRESHOLD COMPARISON =====")
print("Threshold | Precision | Recall | F1")

best_threshold = None
best_f1 = -1

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)

    precision = precision_score(
        y_train, predictions, zero_division=0
    )
    recall = recall_score(
        y_train, predictions, zero_division=0
    )
    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n===== SELECTED THRESHOLD =====")
print(f"Threshold: {best_threshold:.2f}")
print(f"Cross-validated F1: {best_f1:.4f}")
print(f"Average precision: {average_precision_score(y_train, scores):.4f}")
