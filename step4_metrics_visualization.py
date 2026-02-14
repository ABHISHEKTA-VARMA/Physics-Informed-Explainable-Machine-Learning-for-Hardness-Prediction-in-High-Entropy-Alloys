import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter

sns.set_style("white")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

# ------------------------------------------------------------
# Locate dataset automatically
# ------------------------------------------------------------
search_dirs = [
    "/content",
    "/content/drive/MyDrive",
    ".",
    "/mnt/data"
]

csv_candidates = []
for d in search_dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_candidates.append(os.path.join(root, f))

if len(csv_candidates) == 0:
    raise FileNotFoundError("No CSV files found.")

preferred = [
    p for p in csv_candidates
    if ("mpea" in p.lower()) or ("hea" in p.lower())
]

csv_path = preferred[0] if preferred else csv_candidates[0]
print(f"Using dataset file:\n{csv_path}")

df = pd.read_csv(csv_path)

# ------------------------------------------------------------
# Automatic column inference
# ------------------------------------------------------------
hardness_col = None
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        v = df[col].dropna()
        if len(v) > 50 and v.min() > 10 and v.max() < 2000:
            hardness_col = col
            break

if hardness_col is None:
    raise ValueError("Hardness column could not be inferred.")

composition_col = None
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().astype(str).head(20)
        if sample.str.contains(r"[A-Z][a-z]?", regex=True).all():
            composition_col = col
            break

if composition_col is None:
    raise ValueError("Composition column could not be inferred.")

# ------------------------------------------------------------
# Hardness distribution plot
# ------------------------------------------------------------
hv = df[hardness_col].dropna()

plt.figure(figsize=(7.2, 5.2))
sns.histplot(
    hv,
    bins=30,
    kde=True,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.6
)
plt.axvline(
    hv.median(),
    linestyle="--",
    color="black",
    linewidth=1.2,
    label="Median"
)
plt.xlabel("Vickers Hardness (HV)")
plt.ylabel("Frequency")
plt.title("Distribution of Experimental Vickers Hardness Values")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Element frequency (top 10)
# ------------------------------------------------------------
elements = []
for comp in df[composition_col].dropna():
    elements.extend(re.findall(r"[A-Z][a-z]?", str(comp)))

labels, counts = zip(*Counter(elements).most_common(10))

plt.figure(figsize=(7.2, 5.2))
plt.barh(
    labels[::-1],
    counts[::-1],
    color="#55A868",
    edgecolor="black",
    linewidth=0.6
)
plt.xlabel("Number of Alloy Entries")
plt.ylabel("Element")
plt.title("Most Frequently Occurring Elements in the Dataset")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Model performance plots (loaded from saved results)
# ------------------------------------------------------------
perf_path = "output/plots_step4_tuned/descriptor_model_comparison_tuned.csv"

if os.path.exists(perf_path):
    perf = pd.read_csv(perf_path)

    top_models = perf.sort_values("R2", ascending=False).head(2)

    plt.figure(figsize=(6.5, 4.8))
    plt.bar(
        top_models["Model"],
        top_models["R2"],
        color=["#4C72B0", "#DD8452"],
        edgecolor="black",
        linewidth=0.6
    )
    plt.ylim(0, 1.0)
    plt.ylabel("R²")
    plt.title("Comparison of Predictive Accuracy")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Error metric comparison
    # ------------------------------------------------------------
    RMSE = top_models["RMSE"].values
    MAE = top_models["MAE"].values
    models = top_models["Model"].values

    x = np.arange(len(models))
    width = 0.32

    plt.figure(figsize=(6.8, 4.8))
    plt.bar(x - width/2, RMSE, width,
            label="RMSE",
            color="#DD8452",
            edgecolor="black",
            linewidth=0.6)

    plt.bar(x + width/2, MAE, width,
            label="MAE",
            color="#55A868",
            edgecolor="black",
            linewidth=0.6)

    plt.xticks(x, models)
    plt.ylabel("Error (HV)")
    plt.title("Comparison of Prediction Error Metrics")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

else:
    print("Performance CSV not found — skipping model comparison plots.")
