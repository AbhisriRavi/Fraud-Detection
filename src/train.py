import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURES, build_features

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Load, dedupe, time-ordered split
df = (pd.read_csv("data/creditcard.csv")
      .drop_duplicates()
      .sort_values("Time")
      .reset_index(drop=True))
n = len(df)
train = df.iloc[: int(0.6 * n)]
val = df.iloc[int(0.6 * n): int(0.8 * n)]
test = df.iloc[int(0.8 * n):]

X_tr, y_tr = build_features(train), train["Class"]
X_va, y_va = build_features(val), val["Class"]
X_te, y_te = build_features(test), test["Class"]

for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
    print(f"{name}: {len(y)} rows, {y.sum()} frauds")

spw = (y_tr == 0).sum() / (y_tr == 1).sum()

models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000)),
    "xgboost": XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        n_jobs=-1, random_state=42),
}


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(name, y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "model": name,
        "threshold": round(thr, 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


results, fitted = [], {}
plt.figure(figsize=(7, 5))

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    thr = best_threshold(y_va, p_va)
    p_te = model.predict_proba(X_te)[:, 1]

    res = evaluate(name, y_te, p_te, thr)
    res["val_pr_auc"] = round(average_precision_score(y_va, p_va), 4)
    results.append(res)
    fitted[name] = (model, thr)
    print(res)

    prec, rec, _ = precision_recall_curve(y_te, p_te)
    plt.plot(rec, prec, label=f"{name} (PR-AUC {res['pr_auc']})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curves (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/pr_curves.png")
plt.close()

# Choose by validation score, never by test score
best = max(results, key=lambda r: r["val_pr_auc"])
model, thr = fitted[best["model"]]
joblib.dump({"model": model, "threshold": thr, "features": FEATURES,
             "model_name": best["model"]}, "models/fraud_model.joblib")

with open("reports/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBest model: {best['model']} (saved to models/fraud_model.joblib)")