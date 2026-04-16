import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


RAW_DATA_PATH = Path("output/MASTER_HV_DATASET_STEP1_LOCKED.csv")
DESC_DATA_PATH = Path("output/HEA_descriptor_dataset.csv")
PERF_PATH = Path("output/plots_step5/model_results.csv")

OUTPUT_DIR = Path("output/analysis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {RAW_DATA_PATH}")

if not DESC_DATA_PATH.exists():
    raise FileNotFoundError(f"Missing descriptor dataset: {DESC_DATA_PATH}")


df = pd.read_csv(RAW_DATA_PATH)
ddf = pd.read_csv(DESC_DATA_PATH)


if PERF_PATH.exists():
    perf = pd.read_csv(PERF_PATH)
    perf_available = True
else:
    perf_available = False


sns.set_style("whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 1.2
})


hardness_col = "PROPERTY: HV"

df = df[df[hardness_col].notnull() & (df[hardness_col] > 0)].copy()


elem_cols = [c for c in df.columns if c.startswith("ELEM_")]


hv = df[hardness_col]

plt.figure(figsize=(7, 5))
sns.histplot(hv, bins=30, kde=True)

plt.axvline(hv.mean(), linestyle="--", linewidth=1.5)
plt.axvline(hv.median(), linestyle=":", linewidth=1.5)

plt.xlabel("Vickers hardness (HV)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hardness_distribution.png", dpi=600)
plt.close()


counts = (df[elem_cols] > 0).sum().sort_values(ascending=False).head(12)

plt.figure(figsize=(7, 5))
plt.barh(
    [c.replace("ELEM_", "") for c in counts.index][::-1],
    counts.values[::-1]
)

plt.xlabel("Count")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "element_frequency.png", dpi=600)
plt.close()


num_elements = (df[elem_cols] > 0).sum(axis=1)

plt.figure(figsize=(7, 5))
plt.hist(num_elements, bins=10)

plt.xlabel("Number of elements")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "composition_complexity.png", dpi=600)
plt.close()


if perf_available:

    r2_col = "R2_mean" if "R2_mean" in perf.columns else "R2"
    perf = perf.sort_values(r2_col, ascending=False)

    plt.figure(figsize=(7, 5))
    plt.bar(perf["Model"], perf[r2_col])

    plt.xticks(rotation=20)
    plt.ylabel("R2")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance.png", dpi=600)
    plt.close()


if perf_available:

    plt.figure(figsize=(7, 5))

    x = np.arange(len(perf))

    plt.bar(x - 0.2, perf["RMSE"], 0.4)
    plt.bar(x + 0.2, perf["MAE"], 0.4)

    plt.xticks(x, perf["Model"], rotation=20)
    plt.ylabel("Error")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_errors.png", dpi=600)
    plt.close()


numeric = ddf.select_dtypes(include=np.number)

if hardness_col not in numeric.columns:
    raise ValueError(f"{hardness_col} not found in descriptor dataset")

corr = numeric.corr()[hardness_col].drop(hardness_col)

top = corr.abs().sort_values(ascending=False).head(15).index

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric[top.tolist() + [hardness_col]].corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=600)
plt.close()
