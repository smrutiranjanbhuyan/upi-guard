import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

from preprocess import build_features


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv("../data/upi_100k_ultra_realistic.csv")
df = build_features(df)

drop_cols = [
    "fraud_flag",
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id"
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["fraud_flag"]

# ----------------------------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------
# HANDLE IMBALANCE
# ----------------------------------------------------------

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=10,
    eval_metric="auc"
)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

print("\n================ MODEL EVALUATION ================\n")

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"ROC-AUC Score       : {roc_auc:.4f}")
print(f"PR-AUC Score        : {pr_auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDetailed Breakdown:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

# ----------------------------------------------------------
# COST-SENSITIVE LOSS (RESEARCH LEVEL)
# ----------------------------------------------------------

C_FN = 5000   # Fraud loss cost
C_FP = 200    # Customer friction cost

financial_loss = (C_FN * fn) + (C_FP * fp)

print(f"\nEstimated Financial Loss: ₹{financial_loss:,}")

print("\n==================================================\n")

# ----------------------------------------------------------
# SAVE MODEL
# ----------------------------------------------------------

pickle.dump(model, open("../model/fraud_model.pkl", "wb"))
pickle.dump(X.columns, open("../model/feature_columns.pkl", "wb"))

print("Model Trained & Saved Successfully")
