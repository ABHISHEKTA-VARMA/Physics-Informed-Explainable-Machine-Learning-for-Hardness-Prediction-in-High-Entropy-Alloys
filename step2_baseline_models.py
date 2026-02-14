import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor

# ------------------------------------------------------------
# Step 2: Hardness prediction using composition-derived features
# ------------------------------------------------------------

input_file = "/content/output/HEA_raw_experimental_dataset_STEP1_LOCKED.csv"
out_dir = "/content/output/plots_step2"
os.makedirs(out_dir, exist_ok=True)

MODEL_COLORS = {
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728",
    "SVR": "#9467bd",
    "Ridge": "#8c564b",
    "Lasso": "#e377c2",
    "ElasticNet": "#7f7f7f"
}

# ------------------------------------------------------------
# Load and filter dataset
# ------------------------------------------------------------
df = pd.read_csv(input_file)

# Detect hardness column
hardness_cols = [
    c for c in df.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]

if not hardness_cols:
    raise ValueError("No hardness column detected in dataset")

hv_col = hardness_cols[0]
print("Using hardness column:", hv_col)

df = df[
    (df["COMPOSITION_VALID"] == True) &
    (df["HARDNESS_AVAILABLE"] == True) &
    (df[hv_col].notnull()) &
    (df[hv_col] > 0)
].copy()

elem_cols = [c for c in df.columns if c.startswith("ELEM_")]

X = df[elem_cols]
y = df[hv_col]

print("Samples used for modeling:", len(df))
print("Elemental features:", len(elem_cols))

# Basic size check
if len(df) < 20:
    print("Warning: small dataset — results may vary")

# ------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------
models = {
    "Extra Trees": ExtraTreesRegressor(n_estimators=800, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0
    ),
    "SVR": SVR(C=10, gamma="scale"),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=30000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=30000)
}

# ------------------------------------------------------------
# Cross-validation setup
# ------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
cv_preds = {}

# ------------------------------------------------------------
# Cross-validated model evaluation
# ------------------------------------------------------------
for name, model in models.items():

    if name in ["SVR", "Ridge", "Lasso", "ElasticNet"]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        y_cv = cross_val_predict(pipe, X, y, cv=kf)
    else:
        y_cv = cross_val_predict(model, X, y, cv=kf)

    cv_preds[name] = y_cv

    results.append([
        name,
        r2_score(y, y_cv),
        np.sqrt(mean_squared_error(y, y_cv)),
        mean_absolute_error(y, y_cv)
    ])

results_df = pd.DataFrame(
    results, columns=["Model", "R2", "RMSE", "MAE"]
).sort_values("R2", ascending=False)

print(results_df)

# ------------------------------------------------------------
# Model comparison plots
# ------------------------------------------------------------
for metric in ["R2", "RMSE", "MAE"]:
    vals = results_df.set_index("Model")[metric]
    vals = vals.sort_values(ascending=(metric != "R2"))

    plt.figure(figsize=(7, 4))
    plt.bar(
        vals.index,
        vals.values,
        color=[MODEL_COLORS[m] for m in vals.index]
    )
    plt.ylabel(metric)
    plt.title(f"Model Comparison ({metric})")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison_{metric}.png", dpi=600)
    plt.close()

# ------------------------------------------------------------
# Actual vs predicted (cross-validated)
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
for name, y_cv in cv_preds.items():
    plt.scatter(
        y,
        y_cv,
        s=18,
        alpha=0.5,
        color=MODEL_COLORS[name],
        label=name
    )

lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--")
plt.xlabel("Actual Hardness (HV)")
plt.ylabel("Predicted Hardness (HV)")
plt.title("Actual vs Predicted Hardness")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/actual_vs_predicted_all_models.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# Residual distribution
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
for name, y_cv in cv_preds.items():
    try:
        sns.kdeplot(y - y_cv, label=name, linewidth=2)
    except:
        pass

plt.xlabel("Residual (Actual − Predicted)")
plt.title("Residual Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/residual_distribution.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# Feature importance (tree-based models)
# ------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]

if best_model_name in ["Extra Trees", "Random Forest"]:
    best_model = models[best_model_name]
    best_model.fit(X, y)

    importances = pd.Series(
        best_model.feature_importances_,
        index=elem_cols
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(6, 4))
    importances[::-1].plot(
        kind="barh",
        color=MODEL_COLORS[best_model_name],
        edgecolor="black"
    )
    plt.xlabel("Relative Feature Importance")
    plt.title(f"Top Elemental Contributors ({best_model_name})")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_tree_model.png", dpi=600)
    plt.close()

# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------
corr_df = X.copy()
corr_df["HV"] = y

top_elems = (
    corr_df.corr()["HV"]
    .abs()
    .sort_values(ascending=False)
    .iloc[1:16]
    .index
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df[top_elems.tolist() + ["HV"]].corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Correlation: Top Elements vs Hardness")
plt.tight_layout()
plt.savefig(f"{out_dir}/correlation_heatmap.png", dpi=600)
plt.close()

print("Step-2 completed.")
