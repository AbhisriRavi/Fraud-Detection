import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
SAMPLES = json.loads((Path(__file__).parent / "sample_transactions.json").read_text())


def predict(tx):
    return client.post("/predict", json=tx)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info():
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] < 1
    assert len(body["features"]) == 30


@pytest.mark.parametrize("sample", SAMPLES)
def test_predict_response_shape(sample):
    r = predict(sample["transaction"])
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert len(body["top_factors"]) == 5


def test_normal_transactions_not_flagged():
    for s in SAMPLES:
        if s["label"] == 0:
            assert not predict(s["transaction"]).json()["is_fraud"]


def test_known_frauds_caught():
    flagged = [predict(s["transaction"]).json()["is_fraud"] for s in SAMPLES if s["label"] == 1]
    assert sum(flagged) >= 2


def test_missing_field_rejected():
    tx = dict(SAMPLES[0]["transaction"])
    tx.pop("V1")
    assert predict(tx).status_code == 422