
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Class-weighted Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "SMOTE + Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            sampling_strategy=0.1,
            random_state=42,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ]),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "average_precision": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

print("===== IMBALANCE HANDLING COMPARISON =====")

for name, model in models.items():
    print(f"\nTraining: {name}")

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    for metric in scoring:
        value = results[f"test_{metric}"].mean()
        print(f"{metric}: {value:.4f}")
