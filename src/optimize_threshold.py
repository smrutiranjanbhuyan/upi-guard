# ==========================================================
# UPI-Guard++ Threshold Optimization
# Cost-Sensitive Decision Boundary
# ==========================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocess import build_features

# ----------------------------------------------------------
# 1️⃣ LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv("../data/upi_100k_ultra_realistic.csv")
df = build_features(df)

# Drop non-numeric identifiers
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
# 2️⃣ TRAIN / TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------
# 3️⃣ LOAD TRAINED MODEL
# ----------------------------------------------------------

model = pickle.load(open("../model/fraud_model.pkl", "rb"))

# Get probabilities on test set
y_proba = model.predict_proba(X_test)[:, 1]
y_true = y_test.values

# ----------------------------------------------------------
# 4️⃣ COST-SENSITIVE OPTIMIZATION
# ----------------------------------------------------------

thresholds = np.arange(0.1, 0.9, 0.01)

C_FN = 5000   # Fraud loss cost
C_FP = 200    # Customer friction cost

best_threshold = 0
lowest_loss = float("inf")

for t in thresholds:

    y_pred = (y_proba > t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    loss = (C_FN * fn) + (C_FP * fp)

    if loss < lowest_loss:
        lowest_loss = loss
        best_threshold = t

print("Best Threshold:", round(best_threshold, 3))
print("Minimum Expected Loss:", lowest_loss)

# Save threshold
with open("../model/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("Threshold saved to model/threshold.txt")
