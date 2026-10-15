import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATA = os.path.join(os.path.dirname(__file__), "demo_stream.csv")

st.set_page_config(page_title="Fraud Monitor", layout="wide")
st.title("Real-time Fraud Monitor")
st.caption("Replays held-out card transactions through the fraud detection API. "
           "Fraud is oversampled to about 15% for demo purposes (real rate: 0.17%).")

# Sidebar: API status and controls
try:
    info = requests.get(f"{API_URL}/model-info", timeout=3).json()
    st.sidebar.success(f"API online: {info['model']}\n\nThreshold: {info['threshold']}")
except requests.RequestException:
    st.sidebar.error(f"API not reachable at {API_URL}")
    st.stop()

n_tx = st.sidebar.slider("Transactions to stream", 20, 500, 150, step=10)
delay = st.sidebar.slider("Delay between transactions (s)", 0.0, 1.0, 0.1, step=0.05)
start = st.sidebar.button("Start stream", type="primary")

if not start:
    st.info("Choose settings in the sidebar and press **Start stream**.")
    st.stop()

stream = pd.read_csv(DATA).head(n_tx)
threshold = info["threshold"]

# Layout placeholders
kpi = [c.empty() for c in st.columns(5)]
st.subheader("Fraud probability per transaction")
chart_ph = st.empty()
left, right = st.columns([3, 2])
left.subheader("Recent transactions")
table_ph = left.empty()
right.subheader("Alerts")
alerts = right.container(height=450)

OUTCOME_COLOURS = {
    "Caught": "background-color: #c8f7c5",
    "Missed": "background-color: #ffd6a5",
    "False alarm": "background-color: #ffadad",
    "OK": "",
}

rows = []
for i, (_, tx) in enumerate(stream.iterrows(), start=1):
    r = requests.post(f"{API_URL}/predict", json=tx.drop("Class").to_dict(), timeout=5).json()
    actual = int(tx["Class"])
    flagged = r["is_fraud"]

    if flagged and actual:
        outcome = "Caught"
    elif flagged:
        outcome = "False alarm"
    elif actual:
        outcome = "Missed"
    else:
        outcome = "OK"

    rows.append({
        "Txn": i,
        "Amount ($)": round(tx["Amount"], 2),
        "Probability": r["fraud_probability"],
        "Risk": r["risk_level"],
        "Actual": "Fraud" if actual else "Normal",
        "Outcome": outcome,
        "Latency (ms)": r["latency_ms"],
    })
    log = pd.DataFrame(rows)

    # KPIs
    caught = (log["Outcome"] == "Caught").sum()
    total_fraud = (log["Actual"] == "Fraud").sum()
    kpi[0].metric("Processed", len(log))
    kpi[1].metric("Flagged", int(log["Outcome"].isin(["Caught", "False alarm"]).sum()))
    kpi[2].metric("Frauds caught", f"{caught} / {total_fraud}")
    kpi[3].metric("False alarms", int((log["Outcome"] == "False alarm").sum()))
    kpi[4].metric("Avg latency", f"{log['Latency (ms)'].mean():.1f} ms")

    # Chart and table
    chart_ph.line_chart(
        log.set_index("Txn")[["Probability"]].assign(Threshold=threshold),
        height=220,
        color=["#1f77b4", "#ff4b4b"],
    )
    recent = log.tail(15).iloc[::-1]
    table_ph.dataframe(
        recent.style
        .map(lambda v: OUTCOME_COLOURS.get(v, ""), subset=["Outcome"])
        .format({"Amount ($)": "{:.2f}", "Probability": "{:.4f}", "Latency (ms)": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )

    # Alerts
    reasons = ", ".join(f"{f['feature']} ({f['value']})" for f in r["top_factors"][:3])
    if flagged:
        label = "Confirmed fraud" if actual else "False alarm"
        alerts.error(f"**Txn {i}** · ${tx['Amount']:.2f} · p={r['fraud_probability']:.3f} "
                     f"· {label}\n\nTop reasons: {reasons}")
    elif actual:
        alerts.warning(f"**Txn {i}** · ${tx['Amount']:.2f} · p={r['fraud_probability']:.3f} "
                       f"· Missed fraud\n\nTop factors: {reasons}")

    time.sleep(delay)

st.success(f"Stream finished: caught {caught} of {total_fraud} frauds "
           f"with {(log['Outcome'] == 'False alarm').sum()} false alarms.")