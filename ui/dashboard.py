# ==========================================================
# UPI-Guard++ | ML-Aligned Fraud Detection Console
# ==========================================================

import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import math
import random

st.set_page_config(
    page_title="UPI-Guard++",
    page_icon="🛡",
    layout="wide"
)

# ──────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────
for k in ["prob","decision","contributors"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ──────────────────────────────────────────────────────────
# Constants (Dataset-Aligned)
# ──────────────────────────────────────────────────────────
STATES = ["Delhi","Maharashtra","Karnataka","Tamil Nadu",
          "Gujarat","Uttar Pradesh","Telangana",
          "West Bengal","Rajasthan","Odisha"]

BANKS = ["SBI","HDFC","ICICI","Axis","PNB","Bank of Baroda","Kotak"]
MERCHANTS = ["Grocery","Retail","Travel","Entertainment","Food","Bills"]
DEVICES  = ["Android","iOS"]
NETWORKS = ["4G","5G","WiFi"]
TXN_TYPES = ["P2P","P2M"]

# ──────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────
st.markdown("## 🛡 UPI-GUARD++  |  Fraud Risk Console")
st.caption("Strictly aligned with trained ML model features")
st.divider()

left, right = st.columns([3,2], gap="large")

# ══════════════════════════════════════════════════════════
# LEFT — INPUT PANEL
# ══════════════════════════════════════════════════════════
with left:

    st.subheader("🧾 Transaction Identity")

    col1, col2 = st.columns(2)
    with col1:
        transaction_id = st.text_input(
            "Transaction ID",
            f"TXN{random.randint(10000,99999)}"
        )
        sender_id = st.text_input("Sender ID", "S320")

    with col2:
        receiver_id = st.text_input("Receiver ID", "R420")

    st.divider()

    # ── Proper Date + Time (NO manual typing)
    st.subheader("🕒 Transaction Time")

    col_d, col_t = st.columns(2)
    with col_d:
        txn_date = st.date_input("Transaction Date")
    with col_t:
        txn_time = st.time_input("Transaction Time")

    timestamp = datetime.combine(txn_date, txn_time)

    is_night = 1 if timestamp.hour <= 5 else 0
    is_salary_week = 1 if txn_date.day <= 7 else 0

    st.caption(
        f"Derived → Night: {is_night} | Salary Week: {is_salary_week}"
    )

    st.divider()

    st.subheader("💰 Transaction Details")

    amount = st.number_input(
        "Amount (INR)",
        min_value=1.0,
        value=20000.0,
        step=1000.0
    )

    # Log-normal percentile reference
    percentile = min(int((math.log(amount+1)/10)*100),100)
    st.progress(percentile/100)
    st.caption(f"Dataset Amount Percentile ≈ {percentile}%")

    transaction_type = st.selectbox("Transaction Type", TXN_TYPES)
    merchant_category = st.selectbox("Merchant Category", MERCHANTS)

    st.divider()

    st.subheader("🌍 Geography & Banking")

    col3, col4 = st.columns(2)
    with col3:
        sender_state = st.selectbox("Sender State", STATES, index=3)
        sender_bank  = st.selectbox("Sender Bank", BANKS, index=1)
    with col4:
        receiver_state = st.selectbox("Receiver State", STATES, index=6)
        receiver_bank  = st.selectbox("Receiver Bank", BANKS, index=0)

    cross_state = sender_state != receiver_state
    inter_bank  = sender_bank != receiver_bank

    if cross_state:
        st.warning("⚡ Cross-State Transaction")

    if inter_bank:
        st.info("🏦 Inter-Bank Transfer")

    st.divider()

    st.subheader("📱 Device & Network")

    device_type = st.selectbox("Device Type", DEVICES, index=1)
    network_type = st.selectbox("Network Type", NETWORKS, index=1)

    st.divider()

    st.subheader("📊 Behavioural Signals")

    account_age_days = st.slider("Account Age (days)", 1, 1500, 45)
    txn_velocity_1h  = st.slider("Txn Velocity (last hour)", 0, 20, 3)

    analyze_clicked = st.button(
        "🚀 Run Fraud Classification",
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════
# SIMPLE HEURISTIC (MIRRORS YOUR DATA RULES)
# ══════════════════════════════════════════════════════════
if analyze_clicked:

    contributors = {}
    prob = 0.03

    # Rule 1 — Mule
    if account_age_days <= 20 and txn_velocity_1h >= 4:
        prob += 0.55
        contributors["Mule Pattern"] = 0.55

    # Rule 2 — Night High Amount
    if amount > 20000 and is_night and account_age_days < 30:
        prob += 0.50
        contributors["Night High Amount"] = 0.50

    # Rule 3 — Burst
    if txn_velocity_1h > 5:
        prob += 0.40
        contributors["Burst Velocity"] = 0.40

    # Soft signals
    if cross_state:
        prob += 0.06
        contributors["Cross-State"] = 0.06

    if inter_bank:
        prob += 0.04
        contributors["Inter-Bank"] = 0.04

    prob = min(prob, 0.99)
    decision = "Fraud" if prob > 0.5 else "Safe"

    st.session_state.prob = prob
    st.session_state.decision = decision
    st.session_state.contributors = contributors

# ══════════════════════════════════════════════════════════
# RIGHT — RISK VISUALIZATION
# ══════════════════════════════════════════════════════════
with right:

    if st.session_state.prob is None:
        st.info("Awaiting transaction input...")
    else:
        prob = st.session_state.prob
        pct = prob * 100
        trust = 100 - pct

        st.subheader("📈 Fraud Probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            title={'text': "Fraud Risk (%)"},
            gauge={
                'axis': {'range':[0,100]},
                'steps': [
                    {'range':[0,25],'color':'#00ff9d'},
                    {'range':[25,50],'color':'#ffaa00'},
                    {'range':[50,75],'color':'#ff5733'},
                    {'range':[75,100],'color':'#ff0033'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.decision == "Fraud":
            st.error(f"🚨 FRAUD DETECTED ({pct:.2f}%)")
        else:
            st.success(f"✅ TRANSACTION SAFE ({pct:.2f}%)")

        st.metric("Trust Score", f"{trust:.1f} / 100")

        st.divider()
        st.subheader("🧠 Risk Contributors")

        if st.session_state.contributors:
            for k,v in st.session_state.contributors.items():
                st.write(f"• {k} (+{round(v*100,1)}%)")
        else:
            st.write("No major anomaly signals.")

        st.divider()

        with st.expander("🔍 Payload Sent to Model"):
            st.json({
                "transaction_id": transaction_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "merchant_category": merchant_category,
                "sender_state": sender_state,
                "receiver_state": receiver_state,
                "sender_bank": sender_bank,
                "receiver_bank": receiver_bank,
                "device_type": device_type,
                "network_type": network_type,
                "account_age_days": account_age_days,
                "is_salary_week": is_salary_week,
                "is_night": is_night,
                "txn_velocity_1h": txn_velocity_1h
            })
