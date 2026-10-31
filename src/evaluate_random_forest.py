
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_fraud_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "random_forest_metrics.json"

THRESHOLD = 0.45

df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

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

print("Training final Random Forest model...")
model.fit(X_train, y_train)

scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

metrics = {
    "model": "Random Forest",
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    "average_precision": float(average_precision_score(y_test, scores)),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

print("\n===== RANDOM FOREST TEST RESULTS =====")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\n===== CONFUSION MATRIX =====")
print([[int(tn), int(fp)], [int(fn), int(tp)]])

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)
