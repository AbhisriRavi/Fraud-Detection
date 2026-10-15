import json
import os

import pandas as pd

os.makedirs("tests", exist_ok=True)
df = (pd.read_csv("data/creditcard.csv")
      .drop_duplicates()
      .sort_values("Time")
      .reset_index(drop=True))
test = df.iloc[int(0.8 * len(df)):]

picked = pd.concat([
    test[test["Class"] == 0].sample(3, random_state=1),
    test[test["Class"] == 1].sample(3, random_state=1),
])

samples = [
    {"label": int(row["Class"]), "transaction": row.drop("Class").to_dict()}
    for _, row in picked.iterrows()
]
with open("tests/sample_transactions.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"Saved {len(samples)} samples to tests/sample_transactions.json")