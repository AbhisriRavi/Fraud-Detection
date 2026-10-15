import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from explain import explain_transaction  # noqa: E402
from features import build_features  # noqa: E402

bundle = joblib.load(ROOT / "models" / "fraud_model.joblib")
MODEL, THRESHOLD = bundle["model"], bundle["threshold"]

fields = {"Time": (float, ...), "Amount": (float, ...)}
fields.update({f"V{i}": (float, ...) for i in range(1, 29)})
Transaction = create_model("Transaction", **fields)

app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions for fraud with SHAP explanations.",
    version="1.0",
)


def risk_level(p: float) -> str:
    if p >= THRESHOLD:
        return "high"
    if p >= THRESHOLD / 2:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": bundle["model_name"],
        "threshold": round(THRESHOLD, 4),
        "features": bundle["features"],
    }


@app.post("/predict")
def predict(tx: Transaction):
    start = time.perf_counter()
    X = build_features(pd.DataFrame([tx.model_dump()]))
    p = float(MODEL.predict_proba(X)[0, 1])
    return {
        "fraud_probability": round(p, 4),
        "is_fraud": p >= THRESHOLD,
        "risk_level": risk_level(p),
        "threshold": round(THRESHOLD, 4),
        "top_factors": explain_transaction(MODEL, X),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }