import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgb_model.pkl")

# UI
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to check if it's fraudulent:")

amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
normalized_amount = (amount - 88.0) / 250.0  # Replace with mean & std from your dataset

# Predict button
if st.button("Predict Fraud"):
    # Construct input (match number of features used in training!)
    # If you trained on ['normalizedAmount'] only:
    input_data = np.array([[normalized_amount]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Transaction is Fraudulent!")
    else:
        st.success("✅ Transaction is Legitimate.")
