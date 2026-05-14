

import streamlit as st
import requests
import uuid
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="UPI-Guard++",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0e17;
    --panel:    #111827;
    --border:   #1e2d40;
    --accent:   #00d4ff;
    --danger:   #ff3b5c;
    --safe:     #00e5a0;
    --warn:     #ffaa00;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

.stApp { background: var(--bg); }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Top header bar */
.header-bar {
    background: linear-gradient(135deg, #0a0e17 0%, #0d1b2a 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
}
.header-sub {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Card panels */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
}

/* Result verdict box */
.verdict-fraud {
    background: linear-gradient(135deg, rgba(255,59,92,0.15), rgba(255,59,92,0.05));
    border: 1px solid var(--danger);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.verdict-safe {
    background: linear-gradient(135deg, rgba(0,229,160,0.12), rgba(0,229,160,0.04));
    border: 1px solid var(--safe);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}
.verdict-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* Metric chips */
.metric-chip {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

/* API status badge */
.status-online  { color: var(--safe);   font-size: 0.75rem; }
.status-offline { color: var(--danger); font-size: 0.75rem; }

/* Risk bar */
.risk-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin: 0.4rem 0;
}

/* History table */
.hist-row-fraud { color: var(--danger); }
.hist-row-safe  { color: var(--safe); }

/* Streamlit widget overrides */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { color: var(--text) !important; }

div[data-testid="stForm"] { background: transparent; }

.stButton > button {
    background: linear-gradient(135deg, #0066cc, #00aaff) !important;
    border: none !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants (dataset-aligned) ───────────────────────────
API_BASE = os.getenv(
    "API_BASE",
    "http://127.0.0.1:8000"
)

STATES    = ["Delhi","Maharashtra","Karnataka","Tamil Nadu","Gujarat",
             "Uttar Pradesh","Telangana","West Bengal","Rajasthan","Odisha"]
BANKS     = ["SBI","HDFC","ICICI","Axis","PNB","Bank of Baroda","Kotak","IndusInd","Yes Bank"]
MERCHANTS = ["Grocery","Retail","Travel","Entertainment","Food",
             "Bills","Utilities","Healthcare","Shopping","Fuel","Education","Transport","Other"]
DEVICES   = ["Android","iOS","Web"]
NETWORKS  = ["4G","5G","WiFi"]
TXN_TYPES = ["P2P","P2M"]

# ── Session state ─────────────────────────────────────────
for k, v in [
    ("result", None),
    ("history", []),
    ("api_ok", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── API health probe ──────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            st.session_state.api_ok = True
            return r.json()
    except Exception:
        pass
    st.session_state.api_ok = False
    return None

health_data = check_api()

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <span style="font-size:1.8rem">🛡</span>
  <div>
    <div class="header-title">UPI-GUARD++</div>
    <div class="header-sub">Graph-Aware · Cost-Sensitive · XGBoost · Real-Time Fraud Detection</div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.api_ok and health_data:
    api_status_html = (
        f'<span class="status-online">● API ONLINE &nbsp;|&nbsp; '
        f'threshold={health_data.get("threshold", "?")} &nbsp;|&nbsp; '
        f'features={health_data.get("feature_count", health_data.get("features", "?"))}</span>'
    )
else:
    api_status_html = '<span class="status-offline">⚠ API OFFLINE — start uvicorn before classifying</span>'
st.markdown(api_status_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════
left, right = st.columns([5, 4], gap="large")

with left:

    # ── Timestamp ────────────────────────────────────────
    st.markdown('<div class="card-title">🕒 Transaction Time</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        txn_date = st.date_input("Date", label_visibility="collapsed")
    with c2:
        txn_time = st.time_input("Time", label_visibility="collapsed")
    timestamp = datetime.combine(txn_date, txn_time)
    is_night = timestamp.hour <= 5
    is_salary = txn_date.day <= 7
    st.caption(f"{'🌙 Night transaction' if is_night else '☀️ Daytime'} {'· 💵 Salary week' if is_salary else ''}")

    st.divider()

    # ── Amount + IDs ──────────────────────────────────────
    st.markdown('<div class="card-title">💰 Transaction Core</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        amount = st.number_input("Amount (₹)", min_value=1.0, value=20000.0, step=1000.0)
    with c2:
        txn_type = st.selectbox("Type", TXN_TYPES)
    with c3:
        merchant = st.selectbox("Merchant", MERCHANTS)

    sender_id   = st.text_input("Sender ID",   value="USER_042", placeholder="USER_xxx")
    receiver_id = st.text_input("Receiver ID", value="USER_899", placeholder="USER_xxx")

    st.divider()

    # ── Geography ─────────────────────────────────────────
    st.markdown('<div class="card-title">🌍 Geography & Banking</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        s_state = st.selectbox("Sender State",  STATES, index=9)
        s_bank  = st.selectbox("Sender Bank",   BANKS,  index=0)
    with c2:
        r_state = st.selectbox("Receiver State", STATES, index=1)
        r_bank  = st.selectbox("Receiver Bank",  BANKS,  index=1)

    cross = s_state != r_state
    interbank = s_bank != r_bank
    flags = []
    if cross:     flags.append("⚡ Cross-State")
    if interbank: flags.append("🏦 Inter-Bank")
    if flags:
        st.caption("  ·  ".join(flags))

    st.divider()

    # ── Device ────────────────────────────────────────────
    st.markdown('<div class="card-title">📱 Device & Network</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        device  = st.selectbox("Device", DEVICES)
    with c2:
        network = st.selectbox("Network", NETWORKS)

    st.divider()

    # ── Behavioural signals ───────────────────────────────
    st.markdown('<div class="card-title">📊 Behavioural Signals</div>', unsafe_allow_html=True)
    acct_age  = st.slider("Account Age (days)", 1, 1500, 45)
    velocity  = st.slider("Txn Velocity (last 1h)", 0, 20, 3)

    with st.expander("Advanced: sender historical stats (improves accuracy)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            mean_amt = st.number_input("Sender Avg Amount (₹)", value=0.0, step=100.0)
        with c2:
            std_amt  = st.number_input("Sender Std Amount (₹)", value=0.0, step=100.0)
        with c3:
            txn_count = st.number_input("Sender Total Txns",   value=1, step=1)
        c4, c5 = st.columns(2)
        with c4:
            uniq_recv = st.number_input("Unique Receivers",    value=1, step=1)
        with c5:
            pagerank  = st.number_input("Sender PageRank",     value=0.0, format="%.6f")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀  RUN FRAUD CLASSIFICATION", use_container_width=True)

# ══════════════════════════════════════════════════════════
# CLASSIFY
# ══════════════════════════════════════════════════════════
if run:
    if not st.session_state.api_ok:
        st.error("API is offline. Run: `uvicorn api.main:app --reload`")
    else:
        payload = {
            "transaction_id":    f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "timestamp":         timestamp.isoformat(),
            "sender_id":         sender_id or "USER_042",
            "receiver_id":       receiver_id or "USER_899",
            "amount":            float(amount),
            "transaction_type":  txn_type,
            "merchant_category": merchant,
            "sender_state":      s_state,
            "receiver_state":    r_state,
            "sender_bank":       s_bank,
            "receiver_bank":     r_bank,
            "device_type":       device,
            "network_type":      network,
            "account_age_days":  int(acct_age),
            "txn_velocity_1h":   float(velocity),
            "sender_mean_amt":   float(mean_amt) if mean_amt > 0 else None,
            "sender_std_amt":    float(std_amt)  if std_amt  > 0 else None,
            "sender_txn_count":  int(txn_count),
            "unique_receivers":  int(uniq_recv),
            "sender_pagerank":   float(pagerank),
            "sender_degree":     0.0,
        }

        with st.spinner("Classifying via XGBoost model…"):
            try:
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                result["_payload"] = payload
                st.session_state.result = result

                # append to history (keep last 20)
                st.session_state.history.insert(0, {
                    "id":    result["transaction_id"],
                    "amt":   f"₹{amount:,.0f}",
                    "prob":  result["fraud_probability"],
                    "dec":   result["decision"],
                    "risk":  result["risk_level"],
                    "time":  timestamp.strftime("%H:%M:%S"),
                })
                st.session_state.history = st.session_state.history[:20]
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# RIGHT PANEL — RESULTS
# ══════════════════════════════════════════════════════════
with right:

    res = st.session_state.result

    if res is None:
        st.markdown("""
        <div style="height:200px;display:flex;align-items:center;justify-content:center;
                    border:1px dashed #1e2d40;border-radius:12px;margin-top:2rem;">
            <div style="text-align:center;color:#64748b;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">🛡</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;letter-spacing:0.1em">
                    AWAITING TRANSACTION
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        prob     = res["fraud_probability"]
        decision = res["decision"]
        risk     = res["risk_level"]
        threshold_used = res["threshold_used"]

        # ── Verdict ───────────────────────────────────────
        is_fraud = decision == "Fraud"
        verdict_class = "verdict-fraud" if is_fraud else "verdict-safe"
        verdict_color = "#ff3b5c" if is_fraud else "#00e5a0"
        verdict_icon  = "🚨" if is_fraud else "✅"

        st.markdown(f"""
        <div class="{verdict_class}">
            <div class="verdict-label" style="color:{verdict_color}">
                {verdict_icon} {decision.upper()}
            </div>
            <div class="verdict-sub">
                Risk Level: <strong style="color:{verdict_color}">{risk}</strong>
                &nbsp;·&nbsp; Threshold: {threshold_used}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────
        gauge_color = "#ff3b5c" if prob > 0.70 else ("#ffaa00" if prob > threshold_used else "#00e5a0")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 2),
            number={"suffix": "%", "font": {"size": 28, "color": gauge_color, "family": "Space Mono"}},
            title={"text": "FRAUD PROBABILITY", "font": {"size": 11, "color": "#64748b", "family": "Space Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#1e2d40", "tickfont": {"color": "#64748b", "size": 9}},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#111827",
                "bordercolor": "#1e2d40",
                "steps": [
                    {"range": [0, 18],   "color": "rgba(0,229,160,0.08)"},
                    {"range": [18, 70],  "color": "rgba(255,170,0,0.08)"},
                    {"range": [70, 100], "color": "rgba(255,59,92,0.10)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 1},
                    "thickness": 0.7,
                    "value": threshold_used * 100
                }
            }
        ))
        fig.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Key metrics ───────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability",   f"{prob:.4f}")
        c2.metric("Decision",      decision)
        c3.metric("Risk Level",    risk)

        st.divider()

        # ── Transaction ID + raw payload ──────────────────
        st.markdown(f'<div class="card-title">Transaction ID: {res["transaction_id"]}</div>',
                    unsafe_allow_html=True)

        with st.expander("📦 Full Payload Sent to Model"):
            payload_display = {k: v for k, v in res["_payload"].items()
                               if not k.startswith("_")}
            st.json(payload_display)

    # ── History ───────────────────────────────────────────
    st.divider()
    st.markdown('<div class="card-title">📋 Session History</div>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.caption("No transactions classified yet.")
    else:
        df_hist = pd.DataFrame(hist)
        # Color-code decision column
        def color_dec(val):
            c = "#ff3b5c" if val == "Fraud" else "#00e5a0"
            return f"color: {c}; font-weight: 600"

        styled = (
            df_hist.rename(columns={
                "id": "Txn ID", "amt": "Amount", "prob": "Prob",
                "dec": "Decision", "risk": "Risk", "time": "Time"
            })
            .style
            .applymap(color_dec, subset=["Decision"])
            .format({"Prob": "{:.4f}"})
        )
        st.dataframe(styled, use_container_width=True, height=220)

        # Mini bar chart of probabilities
        if len(hist) >= 2:
            fig2 = px.bar(
                df_hist, x="id", y="prob",
                color="dec",
                color_discrete_map={"Fraud": "#ff3b5c", "Safe": "#00e5a0"},
                labels={"id": "", "prob": "Fraud Prob", "dec": ""},
            )
            fig2.add_hline(
                y=health_data["threshold"] if health_data else 0.18,
                line_dash="dash", line_color="white",
                annotation_text="threshold", annotation_font_size=9
            )
            fig2.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0", "size": 10},
                xaxis=dict(showticklabels=False, gridcolor="#1e2d40"),
                yaxis=dict(gridcolor="#1e2d40", range=[0, 1]),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        if st.button("🗑  Clear History", use_container_width=False):
            st.session_state.history = []
            st.rerun()