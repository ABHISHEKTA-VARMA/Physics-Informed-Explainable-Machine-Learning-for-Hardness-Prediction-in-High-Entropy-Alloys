import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor

# ------------------------------------------------------------
# Step 4: Hardness prediction using descriptor features
# ------------------------------------------------------------

input_file = "output/HEA_descriptor_43_dataset.csv"
out_dir = "output/plots_step4_tuned"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------
# Color palette
# ------------------------------------------------------------
MODEL_COLORS = {
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost (Tuned)": "#d62728",
    "SVR": "#9467bd",
    "Ridge": "#8c564b",
    "Lasso": "#e377c2",
    "ElasticNet": "#7f7f7f"
}

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(input_file)

# Detect hardness column
hardness_cols = [
    c for c in df.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]

if not hardness_cols:
    raise ValueError("No hardness column found in descriptor dataset")

target_col = hardness_cols[0]
print("Using hardness column:", target_col)

# Keep rows with valid hardness values
df = df[df[target_col].notnull() & (df[target_col] > 0)].copy()

# ------------------------------------------------------------
# Descriptor feature selection
# Remove composition text and metadata columns
# ------------------------------------------------------------
exclude_cols = [
    "FORMULA", "parsed", "PARSED_COMPOSITION",
    target_col,
    "PROPERTY: Elongation (%)",
    "PROPERTY: Calculated Density (g/cm3)",
    "PROPERTY: YS (MPa)",
    "PROPERTY: UTS (MPa)",
    "PROPERTY: grain size ($\\mu$m)",
    "PROPERTY: Test temperature ($^\\circ$C)",
    "REFERENCE: year"
]

X = df.drop(columns=[c for c in exclude_cols if c in df.columns])

# Keep numeric columns only
X = X.select_dtypes(exclude=["object"])

# Remove empty or constant columns
X = X.loc[:, X.notnull().any()]
X = X.loc[:, (X != 0).any(axis=0)]

y = df[target_col]

print("Samples used for modeling:", len(df))
print("Number of descriptors:", X.shape[1])

if len(df) < 25:
    print("Warning: limited sample size")

# ------------------------------------------------------------
# Cross-validation setup
# ------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------------------------------------
# Hyperparameter search for XGBoost
# ------------------------------------------------------------
xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ))
])

param_dist = {
    "model__n_estimators": [300, 500, 700, 900],
    "model__max_depth": [3, 4, 5, 6, 7, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__reg_lambda": [0, 0.5, 1, 2, 5],
    "model__gamma": [0, 0.1, 0.2, 0.3]
}

print("\nRunning parameter search for XGBoost...")

search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_dist,
    n_iter=40,
    scoring="r2",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)
best_xgb_pipe = search.best_estimator_

print("\nSelected XGBoost parameters:")
print(search.best_params_)
print("Best internal CV score:", search.best_score_)

# ------------------------------------------------------------
# Model set
# ------------------------------------------------------------
models = {
    "Extra Trees": ExtraTreesRegressor(n_estimators=600, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost (Tuned)": best_xgb_pipe,
    "SVR": SVR(C=10, gamma="scale"),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=30000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=30000)
}

# ------------------------------------------------------------
# Cross-validated evaluation
# ------------------------------------------------------------
results = []
cv_preds = {}

for name, model in models.items():

    if name == "XGBoost (Tuned)":
        y_cv = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    else:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        y_cv = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)

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

print("\nCross-validated performance:")
print(results_df)

results_df.to_csv(f"{out_dir}/descriptor_model_comparison_tuned.csv", index=False)

# ------------------------------------------------------------
# Model comparison plots
# ------------------------------------------------------------
for metric in ["R2", "RMSE", "MAE"]:
    vals = results_df.set_index("Model")[metric]
    vals = vals.sort_values(ascending=(metric != "R2"))

    plt.figure(figsize=(7, 4))
    plt.bar(vals.index, vals.values,
            color=[MODEL_COLORS[m] for m in vals.index])
    plt.ylabel(metric)
    plt.title(f"Descriptor Model Comparison – {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison_{metric}.png", dpi=600)
    plt.close()

# ------------------------------------------------------------
# Actual vs predicted
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
for name, y_cv in cv_preds.items():
    plt.scatter(y, y_cv, s=18, alpha=0.5,
                color=MODEL_COLORS[name], label=name)

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
# Feature importance (if XGBoost selected)
# ------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]

if best_model_name == "XGBoost (Tuned)":
    best_xgb_pipe.fit(X, y)
    importances = best_xgb_pipe.named_steps["model"].feature_importances_

    top_feats = pd.Series(importances, index=X.columns)\
        .sort_values(ascending=False).head(15)

    plt.figure(figsize=(6, 4))
    top_feats[::-1].plot(kind="barh", color="#d62728", edgecolor="black")
    plt.xlabel("Relative Feature Importance")
    plt.title("Top Descriptor Contributors (XGBoost)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_xgb_tuned.png", dpi=600)
    plt.close()

# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------
corr_df = X.copy()
corr_df["HV"] = y

top_desc = (
    corr_df.corr()["HV"]
    .abs()
    .sort_values(ascending=False)
    .iloc[1:16]
    .index
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df[top_desc.tolist() + ["HV"]].corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True
)
plt.title("Correlation: Top Descriptors vs Hardness")
plt.tight_layout()
plt.savefig(f"{out_dir}/correlation_heatmap_descriptors.png", dpi=600)
plt.close()

print("Step-4 completed.")
