import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/creditcard.csv")

# 1. Data quality
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print(f"Fraud rate: {df['Class'].mean() * 100:.3f}%\n")

# 2. Amount by class
print("Amount stats by class:")
print(df.groupby("Class")["Amount"].describe(), "\n")

plt.figure(figsize=(7, 4))
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False)
plt.title("Transaction amount by class (outliers hidden)")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("reports/figures/amount_by_class.png")
plt.close()

# 3. Time of day (Time is seconds since first transaction, data spans 2 days)
df["Hour"] = (df["Time"] // 3600) % 24
hourly = df.groupby("Hour")["Class"].mean() * 100

plt.figure(figsize=(9, 4))
hourly.plot(kind="bar")
plt.title("Fraud rate (%) by hour of day")
plt.ylabel("Fraud rate (%)")
plt.tight_layout()
plt.savefig("reports/figures/fraud_by_hour.png")
plt.close()

# 4. Features most correlated with fraud
corr = df.drop(columns=["Hour"]).corr()["Class"].drop("Class")
top = corr.abs().sort_values(ascending=False).head(10)
print("Top 10 features by |correlation| with fraud:")
print(corr[top.index].round(3), "\n")

plt.figure(figsize=(8, 4))
corr[top.index].plot(kind="bar")
plt.title("Top 10 features correlated with fraud")
plt.tight_layout()
plt.savefig("reports/figures/top_correlations.png")
plt.close()

# 5. Distributions of the 4 strongest features
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, feat in zip(axes.flat, top.index[:4]):
    sns.kdeplot(df[df["Class"] == 0][feat], ax=ax, label="Normal", fill=True)
    sns.kdeplot(df[df["Class"] == 1][feat], ax=ax, label="Fraud", fill=True)
    ax.set_title(feat)
    ax.legend()
plt.tight_layout()
plt.savefig("reports/figures/top_feature_distributions.png")
plt.close()

print("Charts saved to reports/figures/")