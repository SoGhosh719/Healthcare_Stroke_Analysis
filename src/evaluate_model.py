import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/feature_importance.csv")
df = df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(df["Feature"], df["Importance"], color='steelblue', edgecolor='black')
plt.title("Feature Importance - GBTClassifier")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("data/feature_importance_plot.png")
plt.show()
