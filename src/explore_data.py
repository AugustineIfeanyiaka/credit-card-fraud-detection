
import pandas as pd
from pathlib import Path

# Locate the dataset relative to the project directory
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "creditcard.csv"

# Load dataset
df = pd.read_csv(data_path)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== FRAUD DISTRIBUTION =====")
print(df["Class"].value_counts())

fraud_percentage = df["Class"].mean() * 100
print(f"\nFraud percentage: {fraud_percentage:.4f}%")

print("\n===== TRANSACTION AMOUNT SUMMARY =====")
print(df["Amount"].describe())
