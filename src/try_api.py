import json

import requests

with open("tests/sample_transactions.json") as f:
    samples = json.load(f)

for s in samples:
    r = requests.post("http://127.0.0.1:8000/predict", json=s["transaction"]).json()
    actual = "FRAUD " if s["label"] else "normal"
    print(f"actual={actual} | p={r['fraud_probability']:.4f} | "
          f"risk={r['risk_level']:<6} | top={r['top_factors'][0]['feature']} | "
          f"{r['latency_ms']} ms")