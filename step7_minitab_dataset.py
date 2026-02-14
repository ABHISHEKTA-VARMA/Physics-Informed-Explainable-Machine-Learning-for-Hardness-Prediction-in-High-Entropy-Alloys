import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# File paths (relative)
# ------------------------------------------------------------
descriptor_file = "output/HEA_descriptor_43_dataset.csv"
shap_table_file = "output/plots_step5_shap/shap_global_importance.csv"

output_dir = "output/MINITAB_INPUTS"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------
df = pd.read_csv(descriptor_file)
shap_df = pd.read_csv(shap_table_file)

print("Descriptor dataset shape :", df.shape)
print("SHAP importance table   :", shap_df.shape)

# ------------------------------------------------------------
# Target filtering (hardness > 0)
# ------------------------------------------------------------
hardness_cols = [
    c for c in df.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]
if not hardness_cols:
    raise ValueError("Hardness column not found in descriptor dataset")

target = hardness_cols[0]

df = df[df[target].notnull() & (df[target] > 0)].reset_index(drop=True)
print("Filtered dataset (HV valid):", df.shape)

# ------------------------------------------------------------
# Select top-N descriptors based on SHAP ranking
# ------------------------------------------------------------
TOP_N = 15

top_features = (
    shap_df
    .sort_values("Mean_Abs_SHAP", ascending=False)
    .head(TOP_N)["Descriptor"]
    .tolist()
)

print("Top-15 descriptors from SHAP ranking:")
for i, f in enumerate(top_features, 1):
    print(f"{i:02d}. {f}")

# Check availability in dataset
missing = [f for f in top_features if f not in df.columns]
if missing:
    raise ValueError(f"Missing descriptors in dataset: {missing}")

# ------------------------------------------------------------
# Prepare dataset for export
# ------------------------------------------------------------
minitab_df = df[top_features + [target]].copy()

# Simplify column names (remove spaces and brackets)
rename_map = {
    c: c.replace(" ", "_").replace("(", "").replace(")", "")
    for c in minitab_df.columns
}
minitab_df.rename(columns=rename_map, inplace=True)

target_renamed = rename_map[target]

# ------------------------------------------------------------
# Standardize predictors only (z-score)
# ------------------------------------------------------------
for col in minitab_df.columns:
    if col != target_renamed:
        std = minitab_df[col].std(ddof=0)
        if std > 0:
            minitab_df[col] = (minitab_df[col] - minitab_df[col].mean()) / std
        else:
            minitab_df[col] = 0.0

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
output_csv = os.path.join(
    output_dir, "MINITAB_SHAP_TOP15_MASTER.csv"
)
minitab_df.to_csv(output_csv, index=False)

# ------------------------------------------------------------
# Save accompanying notes
# ------------------------------------------------------------
meta_path = os.path.join(
    output_dir, "MINITAB_SHAP_TOP15_META.txt"
)

with open(meta_path, "w") as f:
    f.write("Descriptor set prepared from SHAP ranking\n")
    f.write("=" * 55 + "\n\n")

    f.write("Target variable:\n")
    f.write(f" - {target} (experimental hardness values)\n\n")

    f.write("Selected descriptors (ranked by mean |SHAP|):\n")
    for i, feat in enumerate(top_features, 1):
        f.write(f"{i:02d}. {feat}\n")

    f.write("\nPreprocessing:\n")
    f.write("- Dataset filtered to rows with valid hardness values\n")
    f.write("- Predictors standardized using z-score normalization\n")
    f.write("- Target variable kept in original units\n")
    f.write("- Feature selection based on SHAP global importance\n")

print("Top-15 descriptor dataset written successfully.")
print("CSV file :", output_csv)
print("Notes file:", meta_path)
