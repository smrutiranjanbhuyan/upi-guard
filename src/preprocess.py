# ==========================================================
# UPI-Guard++ Preprocessing & Feature Engineering
# 100K Ultra-Realistic Indian UPI Dataset
# ==========================================================

import pandas as pd
import numpy as np
import networkx as nx


def build_features(df):

    # ------------------------------------------------------
    # 1️⃣ BASIC CLEANING
    # ------------------------------------------------------

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df.sort_values(["sender_id", "timestamp"])


    # ------------------------------------------------------
    # 2️⃣ TEMPORAL FEATURES
    # ------------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month

    # Night risk
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)

    # Salary week (1st 7 days)
    df["is_salary_week"] = (df["timestamp"].dt.day <= 7).astype(int)

    # ------------------------------------------------------
    # 3️⃣ TRANSACTION VELOCITY (FIXED VERSION)
    # ------------------------------------------------------

    df = df.set_index("timestamp")

    df["txn_velocity_1h"] = (
        df.groupby("sender_id")["amount"]
        .rolling("1h")
        .count()
        .reset_index(level=0, drop=True)
    )

    df = df.reset_index()

    # Fill missing velocities
    df["txn_velocity_1h"] = df["txn_velocity_1h"].fillna(1)


    # ------------------------------------------------------
    # 4️⃣ BEHAVIORAL FEATURES
    # ------------------------------------------------------

    df["sender_mean_amt"] = (
        df.groupby("sender_id")["amount"].transform("mean")
    )

    df["sender_std_amt"] = (
        df.groupby("sender_id")["amount"].transform("std")
    )

    df["sender_txn_count"] = (
        df.groupby("sender_id")["amount"].transform("count")
    )

    df["unique_receivers"] = (
        df.groupby("sender_id")["receiver_id"].transform("nunique")
    )

    # Z-score anomaly
    df["amount_zscore"] = (
        (df["amount"] - df["sender_mean_amt"]) /
        (df["sender_std_amt"] + 1e-5)
    )


    # ------------------------------------------------------
    # 5️⃣ GRAPH FEATURES
    # ------------------------------------------------------

    print("Building transaction graph...")

    G = nx.from_pandas_edgelist(
        df,
        source="sender_id",
        target="receiver_id",
        create_using=nx.DiGraph()
    )

    pagerank = nx.pagerank(G)
    degree = nx.degree_centrality(G)

    df["sender_pagerank"] = df["sender_id"].map(pagerank)
    df["sender_degree"] = df["sender_id"].map(degree)

    # Fill missing graph values
    df["sender_pagerank"] = df["sender_pagerank"].fillna(0)
    df["sender_degree"] = df["sender_degree"].fillna(0)


    # ------------------------------------------------------
    # 6️⃣ CROSS-STATE FLAG
    # ------------------------------------------------------

    if "sender_state" in df.columns and "receiver_state" in df.columns:
        df["cross_state"] = (
            df["sender_state"] != df["receiver_state"]
        ).astype(int)


    # ------------------------------------------------------
    # 7️⃣ ENCODING CATEGORICAL FEATURES
    # ------------------------------------------------------

    categorical_cols = [
        "transaction_type",
        "merchant_category",
        "sender_state",
        "receiver_state",
        "sender_bank",
        "receiver_bank",
        "device_type",
        "network_type"
    ]

    existing_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)


    # ------------------------------------------------------
    # 8️⃣ FINAL CLEANUP
    # ------------------------------------------------------

    df = df.fillna(0)

    print("Feature engineering completed.")
    print("Total Features:", len(df.columns))

    return df
