import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import build_features

_explainer = None


def get_explainer(model):
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    return _explainer


def explain_transaction(model, X_row: pd.DataFrame, top_k: int = 5):
    """Top features pushing one transaction towards (positive) or away from (negative) fraud."""
    values = get_explainer(model).shap_values(X_row)[0]
    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {
            "feature": X_row.columns[i],
            "value": round(float(X_row.iloc[0, i]), 4),
            "contribution": round(float(values[i]), 4),
        }
        for i in order
    ]


if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    bundle = joblib.load("models/fraud_model.joblib")
    model, thr = bundle["model"], bundle["threshold"]

    # Recreate the same test split as train.py
    df = (pd.read_csv("data/creditcard.csv")
          .drop_duplicates()
          .sort_values("Time")
          .reset_index(drop=True))
    test = df.iloc[int(0.8 * len(df)):]
    X_te, y_te = build_features(test), test["Class"]

    # Global explanation: 2,000 normal transactions plus every fraud
    sample = pd.concat([X_te[y_te == 0].sample(2000, random_state=42), X_te[y_te == 1]])
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False, max_display=12)
    plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Local explanation: the most confident caught fraud
    probs = model.predict_proba(X_te)[:, 1]
    caught = np.where((y_te.values == 1) & (probs >= thr))[0]
    i = caught[np.argmax(probs[caught])]
    row = X_te.iloc[[i]]

    print(f"Example caught fraud: probability {probs[i]:.4f}, "
          f"amount ${test['Amount'].iloc[i]:.2f}")
    for item in explain_transaction(model, row):
        print(item)

    shap.plots.waterfall(explainer(row)[0], show=False, max_display=10)
    plt.savefig("reports/figures/shap_example_fraud.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("SHAP charts saved to reports/figures/")