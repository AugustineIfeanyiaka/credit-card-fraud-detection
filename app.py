
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_fraud_model.joblib"
DATA_PATH = ROOT / "data" / "demo_transactions.csv"

THRESHOLD = 0.45

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_samples():
    df = pd.read_csv(DATA_PATH)

    legitimate = df[df["Class"] == 0].head(5)
    fraudulent = df[df["Class"] == 1].head(5)

    return pd.concat([legitimate, fraudulent])

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Portfolio Project — Random Forest")

st.caption(
    "Educational demonstration using anonymized credit-card transactions."
)

st.divider()

model = load_model()
samples = load_samples()

st.subheader("Select a Transaction")

transaction_type = st.selectbox(
    "Transaction Category",
    ["Legitimate", "Fraudulent"],
)

if transaction_type == "Legitimate":
    available = samples[samples["Class"] == 0]
else:
    available = samples[samples["Class"] == 1]

sample_number = st.selectbox(
    "Sample Number",
    list(range(1, len(available) + 1)),
)

selected = available.iloc[sample_number - 1]

st.write("**Transaction amount:**", selected["Amount"])
st.write("**Transaction time:**", selected["Time"])

if st.button("Analyze Transaction", type="primary"):
    features = selected.drop(labels=["Class"])
    X = pd.DataFrame([features])

    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    score = model.predict_proba(X)[0, 1]
    prediction = int(score >= THRESHOLD)

    st.divider()
    st.subheader("Fraud Detection Results")

    st.metric("Model Fraud Score", f"{score:.2%}")
    st.progress(float(score))

    if prediction == 1:
        st.error("Potential Fraud Detected")
    else:
        st.success("Transaction Classified as Legitimate")

    st.write(f"Classification threshold: {THRESHOLD:.2f}")

    st.caption(
        "Model scores are not guaranteed to be calibrated "
        "probabilities of fraud."
    )
