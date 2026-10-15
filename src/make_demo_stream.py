import pandas as pd

df = (pd.read_csv("data/creditcard.csv")
      .drop_duplicates()
      .sort_values("Time")
      .reset_index(drop=True))
test = df.iloc[int(0.8 * len(df)):]

fraud = test[test["Class"] == 1]
normal = test[test["Class"] == 0].sample(426, random_state=7)
stream = pd.concat([fraud, normal]).sample(frac=1, random_state=7)

stream.to_csv("dashboard/demo_stream.csv", index=False)
print(f"Saved {len(stream)} transactions ({len(fraud)} fraud) to dashboard/demo_stream.csv")