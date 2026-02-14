import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------
# Global settings (reproducibility)
# ------------------------------------------------------------
np.random.seed(42)

plot_dir = "output/plots_step5_shap"
os.makedirs(plot_dir, exist_ok=True)

# ------------------------------------------------------------
# Load descriptor dataset
# ------------------------------------------------------------
descriptor_file = "output/HEA_descriptor_43_dataset.csv"
df = pd.read_csv(descriptor_file)
print("Descriptor dataset loaded:", df.shape)

# ------------------------------------------------------------
# Detect hardness column dynamically
# ------------------------------------------------------------
hardness_cols = [
    c for c in df.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]

if not hardness_cols:
    raise ValueError("Hardness column not found in descriptor dataset")

target_col = hardness_cols[0]

df = df[df[target_col].notnull() & (df[target_col] > 0)].copy()
y = df[target_col]

# ------------------------------------------------------------
# Descriptor feature matrix (consistent with Step-4 filtering)
# ------------------------------------------------------------
exclude_cols = [
    "FORMULA",
    "parsed",
    "PARSED_COMPOSITION",
    target_col,
    "PROPERTY: YS (MPa)",
    "PROPERTY: UTS (MPa)",
    "PROPERTY: grain size ($\\mu$m)",
    "PROPERTY: Test temperature ($^\\circ$C)",
    "REFERENCE: year"
]

X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
X = X.select_dtypes(exclude=["object"])
X = X.loc[:, X.notnull().any()]
X = X.loc[:, (X != 0).any(axis=0)]

print("Final descriptor matrix:", X.shape)

# ------------------------------------------------------------
# Train–test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# ------------------------------------------------------------
# Median imputation (fit on training data only)
# ------------------------------------------------------------
imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_train.columns,
    index=X_test.index
)

# ------------------------------------------------------------
# Train model for SHAP analysis
# ------------------------------------------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

model.fit(X_train_imp, y_train)
print("XGBoost model trained")

# ------------------------------------------------------------
# SHAP analysis (TreeExplainer for tree-based model)
# ------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_imp)

# ------------------------------------------------------------
# SHAP summary plot (beeswarm)
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values,
    X_test_imp,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(plot_dir, "shap_summary_beeswarm.png"),
    dpi=600
)
plt.close()

# ------------------------------------------------------------
# Global SHAP importance table
# ------------------------------------------------------------
mean_shap = np.abs(shap_values).mean(axis=0)

shap_importance = (
    pd.DataFrame({
        "Descriptor": X.columns,
        "Mean_Abs_SHAP": mean_shap
    })
    .sort_values("Mean_Abs_SHAP", ascending=False)
    .reset_index(drop=True)
)

# Save importance values for later use
shap_importance.to_csv(
    os.path.join(plot_dir, "shap_global_importance.csv"),
    index=False
)

# ------------------------------------------------------------
# SHAP importance bar plot (top features)
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
shap_importance.head(15).iloc[::-1].plot(
    kind="barh",
    x="Descriptor",
    y="Mean_Abs_SHAP",
    legend=False,
    edgecolor="black"
)
plt.xlabel("Mean |SHAP value|")
plt.title("Top Descriptor Contributions")
plt.tight_layout()
plt.savefig(
    os.path.join(plot_dir, "shap_global_bar.png"),
    dpi=600
)
plt.close()

# ------------------------------------------------------------
# SHAP dependence plots (top 3 features)
# ------------------------------------------------------------
top_features = shap_importance["Descriptor"].head(3).tolist()

for feat in top_features:
    shap.dependence_plot(
        feat,
        shap_values,
        X_test_imp,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"shap_dependence_{feat}.png"),
        dpi=600
    )
    plt.close()

# ------------------------------------------------------------
# Local SHAP explanation (median hardness sample)
# ------------------------------------------------------------
median_idx = np.argsort(y_test.values)[len(y_test) // 2]

plt.figure(figsize=(8, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[median_idx],
        base_values=explainer.expected_value,
        data=X_test_imp.iloc[median_idx],
        feature_names=X.columns
    ),
    max_display=10,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(plot_dir, "shap_local_waterfall.png"),
    dpi=600
)
plt.close()

# ------------------------------------------------------------
# Redundancy analysis for top descriptors
# ------------------------------------------------------------
topN = 15
top_feats = shap_importance["Descriptor"].head(topN).tolist()

# Correlation matrix
corr = X_train_imp[top_feats].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix of Top Descriptors")
plt.tight_layout()
plt.savefig(
    os.path.join(plot_dir, "redundancy_correlation_heatmap.png"),
    dpi=600
)
plt.close()

# VIF table
vif_df = pd.DataFrame({
    "Descriptor": top_feats,
    "VIF": [
        variance_inflation_factor(
            X_train_imp[top_feats].values, i
        )
        for i in range(len(top_feats))
    ]
})

vif_df.to_csv(
    os.path.join(plot_dir, "redundancy_vif_table.csv"),
    index=False
)

print("Step-5 SHAP analysis completed.")
