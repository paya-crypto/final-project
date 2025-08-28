import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
print("hello")
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from xgboost import XGBClassifier

# Create output folders if not exist
os.makedirs("models", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\PAYAL MAHARANA\\OneDrive\\Documents\\\\python\\PS_20174392719_1491204439457_log.csv")
# 2. Preprocessing
# Encode 'type' feature
df['type'] = LabelEncoder().fit_transform(df['type'])

# Drop unnecessary columns
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Features and Labels
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# 3. Anomaly Detection (optional)
iso = IsolationForest(contamination=0.001, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)

iso_pred = pd.Series(np.where(iso.fit_predict(X) == -1, 1, 0))
lof_pred = pd.Series(np.where(lof.fit_predict(X) == -1, 1, 0))

ensemble_pred = ((iso_pred + lof_pred) >= 1).astype(int)
# Optional: uncomment to use ensemble predictions
# y = ensemble_pred

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 5. Train XGBoost with scale_pos_weight
weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# 7. Visualizations
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.savefig("visuals/confusion_matrix.png")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="XGBoost (AUC = {:.3f})".format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("visuals/roc_curve.png")
plt.show()

# 8. Save Model
joblib.dump(model, "python/fraud_xgb_paysim.pkl")
print("✅ Model saved as 'python/fraud_xgb_paysim.pkl'")

