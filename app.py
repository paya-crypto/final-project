# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_xgb_paysim.pkl")

st.title("💳 Fraud Detection App (PaySim)")

# Input form
st.subheader("Enter Transaction Details")
type_map = {'CASH_OUT': 1, 'TRANSFER': 4, 'CASH_IN': 0, 'DEBIT': 2, 'PAYMENT': 3}
trans_type = st.selectbox("Transaction Type", list(type_map.keys()))
step = st.number_input("Step (time)", min_value=0)
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

# Scale inputs manually (must match training preprocessing)
scaled_input = pd.DataFrame([[step, 
                              type_map[trans_type], 
                              amount, 
                              oldbalanceOrg, newbalanceOrig, 
                              oldbalanceDest, newbalanceDest]],
                            columns=['step', 'type', 'amount', 
                                     'oldbalanceOrg', 'newbalanceOrig', 
                                     'oldbalanceDest', 'newbalanceDest'])

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected with {proba*100:.2f}% confidence")
    else:
        st.success(f"✅ Transaction is Legitimate with {100-proba*100:.2f}% confidence")
