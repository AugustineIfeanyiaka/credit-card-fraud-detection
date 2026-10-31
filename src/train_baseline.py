
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

# Load and prepare the dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

# Reproduce the same train/test split as Step 9
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Scaling is fitted on training data only, preventing leakage
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n===== BASELINE MODEL RESULTS =====")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"Average Precision (PR-AUC summary): {average_precision_score(y_test, probabilities):.4f}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, predictions))
