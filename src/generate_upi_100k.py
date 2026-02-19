# ==========================================================
# UPI-Guard++ Strong Signal 100K Dataset Generator
# ==========================================================

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 100000
start_date = datetime(2024, 1, 1)

states = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu",
    "Gujarat", "Uttar Pradesh", "Telangana",
    "West Bengal", "Rajasthan", "Odisha"
]

banks = [
    "SBI", "HDFC", "ICICI", "Axis",
    "PNB", "Bank of Baroda", "Kotak"
]

merchant_categories = [
    "Grocery", "Retail", "Travel",
    "Entertainment", "Food", "Bills"
]

device_types = ["Android", "iOS"]
network_types = ["4G", "5G", "WiFi"]

data = []

# Create mule accounts
mule_accounts = [f"R{random.randint(1,5000)}" for _ in range(100)]

for i in range(N):

    timestamp = start_date + timedelta(
        minutes=random.randint(0, 365*24*60)
    )

    sender_id = f"S{random.randint(1,10000)}"
    receiver_id = f"R{random.randint(1,10000)}"

    hour = timestamp.hour
    is_night = 1 if hour <= 5 else 0
    is_salary_week = 1 if timestamp.day <= 7 else 0

    # Normal transaction amount
    amount = np.random.lognormal(mean=7, sigma=0.8)

    transaction_type = random.choice(["P2P", "P2M"])
    merchant = random.choice(merchant_categories)
    sender_state = random.choice(states)
    receiver_state = random.choice(states)
    sender_bank = random.choice(banks)
    receiver_bank = random.choice(banks)
    device = random.choice(device_types)
    network = random.choice(network_types)

    account_age = random.randint(10, 1500)
    txn_velocity_1h = np.random.poisson(1)

    fraud_flag = 0

    # -------------------------------
    # STRONG FRAUD RULES
    # -------------------------------

    # Rule 1: Mule network fraud
    if receiver_id in mule_accounts:
        fraud_flag = 1
        amount *= random.uniform(3, 6)
        txn_velocity_1h = random.randint(4, 8)
        account_age = random.randint(1, 20)

    # Rule 2: Night + high amount + new account
    elif (
        amount > 20000 and
        is_night == 1 and
        account_age < 30
    ):
        fraud_flag = 1

    # Rule 3: Burst behavior
    elif txn_velocity_1h > 5:
        fraud_flag = 1
        amount *= random.uniform(2, 4)

    # Keep fraud rate around 2-3%
    if fraud_flag == 1 and random.random() > 0.7:
        fraud_flag = 0

    data.append([
        f"TXN{i}",
        timestamp,
        sender_id,
        receiver_id,
        round(amount, 2),
        transaction_type,
        merchant,
        sender_state,
        receiver_state,
        sender_bank,
        receiver_bank,
        device,
        network,
        account_age,
        is_salary_week,
        is_night,
        txn_velocity_1h,
        fraud_flag
    ])

columns = [
    "transaction_id", "timestamp",
    "sender_id", "receiver_id",
    "amount", "transaction_type",
    "merchant_category",
    "sender_state", "receiver_state",
    "sender_bank", "receiver_bank",
    "device_type", "network_type",
    "account_age_days",
    "is_salary_week",
    "is_night",
    "txn_velocity_1h",
    "fraud_flag"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("../data/upi_100k_ultra_realistic.csv", index=False)

print("100K Strong-Signal UPI Dataset Generated")
print("Fraud Rate:", round(df["fraud_flag"].mean(), 4))
