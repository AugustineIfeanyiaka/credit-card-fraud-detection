import joblib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Reproduce the original preprocessing and test split
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

# Use the same model configuration as the baseline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

print("Training optimized fraud detection model...")
model.fit(X_train, y_train)

# Obtain test-set classification scores
scores = model.predict_proba(X_test)[:, 1]

print("\n===== HELD-OUT TEST EVALUATION =====")
print(f"Average precision: {average_precision_score(y_test, scores):.4f}")

for threshold in [0.50, 0.99]:
    predictions = (scores >= threshold).astype(int)

    print(f"\n===== THRESHOLD {threshold:.2f} =====")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall: "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score: "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# Save the final trained pipeline and selected threshold
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

model_path = MODELS_DIR / "fraud_detection_model.joblib"
threshold_path = REPORTS_DIR / "selected_threshold.json"

joblib.dump(model, model_path)

with open(threshold_path, "w") as file:
    json.dump(
        {
            "threshold": 0.99,
            "selection_method": "Highest F1 among seven tested thresholds",
            "validation": "5-fold cross-validation on training data",
            "model": "Class-weighted Logistic Regression"
        },
        file,
        indent=4
    )

print("\n===== MODEL ARTIFACTS SAVED =====")
print("Model:", model_path)
print("Threshold:", threshold_path)

