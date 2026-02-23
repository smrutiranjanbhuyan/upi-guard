import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

from preprocess import build_features


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv("../data/upi_100k_ultra_realistic.csv")

print("Building transaction graph...")
df = build_features(df)
print("Feature engineering completed.")

drop_cols = [
    "fraud_flag",
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id"
]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["fraud_flag"]

print(f"Total Features: {X.shape[1]}")


# ----------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT (NO LEAKAGE)
# ----------------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.2,
    stratify=y_temp,
    random_state=42
)

print("Split completed:")
print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")


# ----------------------------------------------------------
# HANDLE IMBALANCE
# ----------------------------------------------------------

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")


# ----------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.02,
    min_child_weight=5,
    gamma=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)


# ----------------------------------------------------------
# THRESHOLD TUNING ON VALIDATION SET
# ----------------------------------------------------------

C_FN = 5000
C_FP = 200

val_proba = model.predict_proba(X_val)[:, 1]

def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    lowest_loss = float("inf")

    for t in np.arange(0.01, 0.5, 0.01):
        y_pred_temp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
        loss = (fn * C_FN) + (fp * C_FP)

        if loss < lowest_loss:
            lowest_loss = loss
            best_threshold = t

    return best_threshold, lowest_loss


best_threshold, val_loss = find_best_threshold(y_val, val_proba)

print(f"\nOptimal Threshold (Validation): {best_threshold:.2f}")
print(f"Validation Financial Loss: ₹{val_loss:,}")


# ----------------------------------------------------------
# FINAL TEST EVALUATION (UNSEEN DATA)
# ----------------------------------------------------------

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\n================ FINAL TEST EVALUATION ================\n")

roc_auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

print(f"ROC-AUC Score       : {roc_auc:.4f}")
print(f"PR-AUC Score        : {pr_auc:.4f}")
print(f"Final Threshold     : {best_threshold:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

cm = confusion_matrix(y_test, test_pred)
print("Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDetailed Breakdown:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

financial_loss = (C_FN * fn) + (C_FP * fp)
print(f"\nFinal Test Financial Loss: ₹{financial_loss:,}")

print("\n========================================================\n")


# ----------------------------------------------------------
# SAVE MODEL + THRESHOLD
# ----------------------------------------------------------

pickle.dump(model, open("../model/fraud_model.pkl", "wb"))
pickle.dump(X.columns, open("../model/feature_columns.pkl", "wb"))
pickle.dump(best_threshold, open("../model/threshold.pkl", "wb"))

print("Model, Features, Threshold Saved Successfully")