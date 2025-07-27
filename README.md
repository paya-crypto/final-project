```markdown
# 💳 Credit Card Fraud Detection using PaySim Data

This project detects fraudulent transactions using **XGBoost** and visualizes results through a **Streamlit web app**. The dataset is based on **PaySim**, a realistic financial simulation dataset.

---

## 📁 Project Structure

.
├── app.py                         # Streamlit app for prediction  
├── model_training.py             # Script for training the model  
├── fraud_xgb_paysim.pkl          # Trained XGBoost model  
├── visuals/  
│   ├── confusion_matrix.png  
│   └── roc_curve.png  
├── models/  
│   └── (optional saved models)  
├── dataset/  
│   └── PS_20174392719_1491204439457_log.csv  
├── README.md

---

## ✅ Features Used

- `step`: Time step of the transaction  
- `type`: Transaction type (CASH_OUT, TRANSFER, etc.)  
- `amount`: Transaction amount  
- `oldbalanceOrg`: Sender’s old balance  
- `newbalanceOrig`: Sender’s new balance  
- `oldbalanceDest`: Receiver’s old balance  
- `newbalanceDest`: Receiver’s new balance  

---

## 🧠 Model Training

### Steps:

1. **Preprocessing**  
   - Encode `type` using `LabelEncoder`  
   - Drop `nameOrig`, `nameDest`  
   - Standard scale numerical features  

2. **Anomaly Detection (Optional)**  
   - Use `Isolation Forest` and `Local Outlier Factor`  
   - Combine predictions if desired  

3. **Model Training**  
   - Use `XGBClassifier` from XGBoost  
   - Handle imbalance with `scale_pos_weight`  

4. **Evaluation**  
   - Print classification report  
   - Plot confusion matrix and ROC curve  

5. **Saving**  
   - Model saved as `fraud_xgb_paysim.pkl`  

---

## 📊 Visual Output

- 📈 ROC Curve: `visuals/roc_curve.png`  
- 🔲 Confusion Matrix: `visuals/confusion_matrix.png`  

---

## 🌐 Streamlit Web App

### How to Run

```bash
streamlit run app.py
```

### App Features

- Input form for:
  - Transaction type  
  - Step  
  - Amount  
  - Sender and Receiver balances  
- Displays prediction:
  - ✅ Legitimate Transaction  
  - ⚠️ Fraudulent Transaction  
- Shows prediction confidence  

---

## 💾 Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
```

---

## 📌 Notes

- Ensure `fraud_xgb_paysim.pkl` is in the same directory as `app.py`.  
- Make sure feature preprocessing during prediction matches the training pipeline.  

---

## 📚 Dataset Reference

- [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)
```
