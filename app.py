import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_detection_model.pkl")
st.write(model.feature_names_in_)

st.title("Fraud Detection Prediction App")

st.markdown("Enter transaction details to check if it is fraudulent.")

st.divider()

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)

newbalanceOrg = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)

newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=0.0)


if st.button("Predict"):

    input_data = pd.DataFrame({
"type": [transaction_type],
"amount": [amount],
"oldbalanceOrg": [oldbalanceOrg],
"newbalanceOrig": [newbalanceOrig],
"oldbalanceDest": [oldbalanceDest],
"newbalanceDest": [newbalanceDest]
})

    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction: {prediction}")
    if prediction == 1:
        st.error("The transaction is likely fraudulent.")
    else:
        st.success("The transaction is likely legitimate.")

if st.button("Predict"):

    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrg": [newbalanceOrg],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Fraud Risk Score")

    st.progress(probability)

    st.write(f"Fraud Probability: {probability:.2%}")

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction Appears Legitimate")

import matplotlib.pyplot as plt

st.subheader("Example Fraud Distribution")

labels = ["Legitimate", "Fraud"]
values = [95, 5]

fig, ax = plt.subplots()
ax.bar(labels, values)

st.pyplot(fig)