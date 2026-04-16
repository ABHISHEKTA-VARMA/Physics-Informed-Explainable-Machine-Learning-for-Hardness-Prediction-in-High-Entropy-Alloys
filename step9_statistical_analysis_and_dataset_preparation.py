import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor


DESCRIPTOR_PATH = Path("output/HEA_descriptor_dataset.csv")
SHAP_PATH = Path("output/plots_step8_shap/shap_importance.csv")

OUTPUT_DIR = Path("output/STEP9_STATISTICAL")
OUTPUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DESCRIPTOR_PATH)
shap_df = pd.read_csv(SHAP_PATH)


target = "PROPERTY: HV"

df = df[df[target].notnull() & (df[target] > 0)].reset_index(drop=True)


if "Mean_Abs_SHAP" in shap_df.columns:
    shap_col = "Mean_Abs_SHAP"
elif "Importance" in shap_df.columns:
    shap_col = "Importance"
else:
    raise ValueError("SHAP importance column not found")


TOP_N = 15

top_feats = (
    shap_df
    .sort_values(shap_col, ascending=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

feature_col = "Descriptor" if "Descriptor" in shap_df.columns else "Feature"
features = top_feats[feature_col].tolist()


raw_df = df[features + [target]].copy()


rename_map = {
    c: c.replace(" ", "_")
          .replace("(", "")
          .replace(")", "")
          .replace("/", "_")
    for c in raw_df.columns
}

raw_df.rename(columns=rename_map, inplace=True)
target_clean = rename_map[target]


raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna()


core_df = raw_df.copy()

for col in core_df.columns:
    if col != target_clean:
        mu = core_df[col].mean()
        sd = core_df[col].std(ddof=0)
        core_df[col] = (core_df[col] - mu) / sd if sd > 0 else 0.0


core_df.to_csv(
    OUTPUT_DIR / "MINITAB_SHAP_TOP15_STANDARDIZED.csv",
    index=False
)


shap_table = top_feats.copy()
shap_table.index += 1

shap_table.rename(columns={shap_col: "Mean(|SHAP|)"}, inplace=True)

shap_table.to_csv(
    OUTPUT_DIR / "SUPP_TABLE_S1_SHAP_RANKING.csv",
    index_label="Rank"
)


stats = []

for f in features:
    col = rename_map[f]

    stats.append([
        col,
        raw_df[col].mean(),
        raw_df[col].std(ddof=0),
        raw_df[col].min(),
        raw_df[col].max()
    ])

stats_df = pd.DataFrame(
    stats,
    columns=["Descriptor", "Mean", "Std", "Min", "Max"]
)

stats_df.to_csv(
    OUTPUT_DIR / "SUPP_TABLE_S2_DESCRIPTOR_STATS.csv",
    index=False
)


corr_df = raw_df.corr()
corr_df.to_csv(OUTPUT_DIR / "SUPP_TABLE_S3_CORRELATION_MATRIX.csv")

corr_spearman = raw_df.corr(method="spearman")
corr_spearman.to_csv(OUTPUT_DIR / "SUPP_TABLE_S5_SPEARMAN.csv")


X = core_df.drop(columns=[target_clean]).values

vif_vals = [
    variance_inflation_factor(X, i)
    for i in range(X.shape[1])
]

vif_df = pd.DataFrame({
    "Descriptor": core_df.drop(columns=[target_clean]).columns,
    "VIF": vif_vals
}).sort_values("VIF", ascending=False)

vif_df.to_csv(
    OUTPUT_DIR / "SUPP_TABLE_S4_VIF.csv",
    index=False
)


with open(OUTPUT_DIR / "README_STEP9.txt", "w") as f:

    f.write("Step 9: Statistical dataset preparation\n\n")
    f.write(f"Target: {target_clean}\n\n")

    f.write("Top descriptors:\n")
    for i, feat in enumerate(features, 1):
        f.write(f"{i:02d}. {feat}\n")

    f.write("\nProcessing:\n")
    f.write("- Features standardized (z-score)\n")
    f.write("- VIF computed on scaled data\n")

