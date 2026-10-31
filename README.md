# Credit Card Fraud Detection — Imbalanced Machine Learning

## Project Overview

This project investigates fraud detection using an anonymized credit-card transaction dataset containing 284,807 transactions and 492 labeled fraudulent transactions.

The project compares Logistic Regression, Random Forest, class weighting, SMOTE and classification threshold optimization to identify fraud while controlling false-positive alerts.

## Technologies

Python 3.12, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn, Joblib and Pytest.

## Dataset

Source: [Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Original transactions: 284,807
- Original fraudulent transactions: 492
- Duplicate rows identified: 1,081
- Records after exact deduplication: 283,726
- Fraudulent records after deduplication: 473
- Training records: 226,980
- Test records: 56,746

The dataset is excluded from GitHub because of its size. Identical records were removed as an experimental assumption; they cannot conclusively be identified as duplicate underlying transactions.

## Experiments

1. Baseline class-weighted Logistic Regression
2. Five-fold cross-validation
3. Logistic Regression threshold optimization
4. Class-weighted Random Forest
5. Random Forest threshold optimization
6. SMOTE versus class weighting

SMOTE was applied inside training folds using an imbalanced-learn pipeline to avoid data leakage.

## Held-Out Test Results

| Metric | Logistic Regression (0.99) | Random Forest (0.45) |
|---|---:|---:|
| Precision | 65.79% | 88.75% |
| Recall | 78.95% | 74.74% |
| F1-score | 71.77% | 81.14% |
| Average precision | 0.6719 | 0.8069 |
| False positives | 39 | 9 |
| False negatives | 20 | 24 |
| True positives | 75 | 71 |

Random Forest produced fewer false alarms and a higher F1-score, while Logistic Regression detected slightly more fraud.

## SMOTE Experiment

Five-fold training cross-validation at the default 0.50 threshold:

| Metric | Class-weighted LR | SMOTE + LR |
|---|---:|---:|
| Average precision | 0.7538 | 0.7554 |
| Precision | 6.05% | 40.98% |
| Recall | 91.79% | 86.51% |
| F1-score | 11.35% | 55.54% |

## Reproduction

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download `creditcard.csv` from the dataset source into `data/`.
4. Run the experiment scripts from the repository root.

Example:

```bash
python src/explore_data.py
python src/prepare_data.py
python src/train_baseline.py
python src/compare_models.py
python src/tune_random_forest.py
python src/evaluate_random_forest.py
python src/compare_imbalance_methods.py
```

## Limitations

- Exact duplicate removal is an experimental preprocessing decision.
- The held-out test set was consulted during iterative model development; results should not be described as a completely untouched final benchmark.
- Prediction scores have not been calibrated as real-world fraud probabilities.
- Transaction costs and operational fraud investigation requirements are unknown.
- Performance on newer or different transaction populations remains untested.

## Author

Augustine Ifeanyiaka

Python | Machine Learning | AI Engineering

Educational portfolio project — not a production banking fraud detection system.