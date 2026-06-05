import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COL = "PROPERTY: HV"
CALCULATE_INTERACTIONS = False

inp_data = Path("output/step3_descriptor_engine/HEA_DESCRIPTOR_DATASET.csv")
inp_model = Path("output/step4_descriptor_ml/best_descriptor_model.pkl")
inp_features = Path("output/step4_descriptor_ml/descriptor_features.csv")

out_dir = Path("output/step7_shap_interpretability")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
})
sns.set_style("whitegrid")
np.random.seed(RANDOM_STATE)

df = pd.read_csv(inp_data)
df = df[df[TARGET_COL] > 0].copy()

model_pipeline = joblib.load(inp_model)
feature_df = pd.read_csv(inp_features)
feature_names = feature_df.iloc[:, 0].tolist()

X = df[feature_names].copy()
y = df[TARGET_COL].values

z_scores = np.abs((y - y.mean()) / (y.std() + 1e-9))
mask = z_scores < 3

X = X.iloc[mask].reset_index(drop=True)
y = y[mask]

X_proc = model_pipeline[:-1].transform(X)
X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
final_model = model_pipeline.named_steps["model"]

try:
    explainer = shap.TreeExplainer(
        final_model,
        feature_perturbation="tree_path_dependent",
    )
except Exception:
    explainer = shap.Explainer(final_model, X_proc)

try:
    shap_exp = explainer(X_proc_df, check_additivity=False)
except TypeError:
    shap_exp = explainer(X_proc_df)

shap_values = shap_exp.values

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_proc_df, show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_beeswarm.png", dpi=700, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_proc_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_bar.png", dpi=700, bbox_inches="tight")
plt.close()

mean_importance = np.abs(shap_values).mean(axis=0)
std_importance = np.abs(shap_values).std(axis=0)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance_mean": mean_importance,
    "Importance_std": std_importance,
})

shap_df["Type"] = shap_df["Feature"].apply(
    lambda x: "Elemental" if str(x).startswith("ELEM_") else "Descriptor"
)

shap_df = shap_df.sort_values(
    "Importance_mean",
    ascending=False,
).reset_index(drop=True)

shap_df.to_csv(out_dir / "shap_importance_final.csv", index=False)

plt.figure(figsize=(8, 10))
plt.errorbar(
    shap_df["Importance_mean"].head(20),
    shap_df["Feature"].head(20),
    xerr=shap_df["Importance_std"].head(20),
    fmt="o",
    color="#1f77b4",
    ecolor="gray",
    capsize=4,
)
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP Value|")
plt.title("Top 20 Features: Interpretability Stability")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "shap_importance_stability.png", dpi=700)
plt.close()

if shap_df["Type"].nunique() > 1:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=shap_df, x="Type", y="Importance_mean", palette="Set2")
    plt.ylabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(out_dir / "descriptor_vs_element.png", dpi=700)
    plt.close()

top_feats = shap_df["Feature"].head(5).tolist()

for feat in top_feats:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feat,
        shap_values,
        X_proc_df,
        interaction_index="auto",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"dependence_{feat}.png", dpi=700, bbox_inches="tight")
    plt.close()

mid_idx = len(X_proc_df) // 2

plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_exp[mid_idx], max_display=12, show=False)
plt.tight_layout()
plt.savefig(out_dir / "local_waterfall_median_sample.png", dpi=700, bbox_inches="tight")
plt.close()

if CALCULATE_INTERACTIONS:
    idx_sample = np.random.choice(
        len(X_proc_df),
        size=min(100, len(X_proc_df)),
        replace=False,
    )

    try:
        interaction_values = explainer.shap_interaction_values(X_proc[idx_sample])
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            interaction_values,
            X_proc_df.iloc[idx_sample],
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "shap_interaction_summary.png", dpi=700, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

top15 = shap_df["Feature"].head(15).tolist()
X_top = X_proc_df[top15]
X_scaled = (X_top - X_top.mean()) / (X_top.std() + 1e-9)
corr = X_scaled.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(out_dir / "top15_feature_correlation.png", dpi=700)
plt.close()

vif_data = []

for i, feature in enumerate(top15):
    try:
        vif_val = variance_inflation_factor(X_scaled.values, i)
    except Exception:
        vif_val = np.nan

    vif_data.append({
        "Feature": feature,
        "VIF": vif_val,
    })

pd.DataFrame(vif_data).to_csv(out_dir / "top15_vif_analysis.csv", index=False)

design_check = X_proc_df.copy()
design_check["Predicted_HV"] = model_pipeline.predict(X)

for feat in top_feats[:3]:
    plt.figure(figsize=(6, 5))
    plt.scatter(
        design_check[feat],
        design_check["Predicted_HV"],
        alpha=0.6,
        color="#1f77b4",
        edgecolor="black",
        s=20,
    )
    plt.xlabel(feat)
    plt.ylabel("Predicted HV")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"design_link_verification_{feat}.png", dpi=700)
    plt.close()

summary = pd.DataFrame({
    "Item": [
        "Samples used",
        "Features evaluated",
        "Top feature",
        "Top feature importance",
        "Output directory",
    ],
    "Value": [
        len(X_proc_df),
        len(feature_names),
        shap_df.iloc[0]["Feature"],
        f"{shap_df.iloc[0]['Importance_mean']:.4f}",
        str(out_dir),
    ],
})

print(summary.to_string(index=False))
