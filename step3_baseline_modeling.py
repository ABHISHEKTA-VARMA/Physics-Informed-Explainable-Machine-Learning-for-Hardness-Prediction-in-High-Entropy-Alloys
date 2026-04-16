import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


def bootstrap_r2(y_true, y_pred, n=500, random_state=42):
    rng = np.random.default_rng(random_state)
    scores = []
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.choice(n_samples, n_samples, replace=True)
        scores.append(r2_score(y_true[idx], y_pred[idx]))

    return np.percentile(scores, [2.5, 97.5])


RANDOM_STATE = 42

INPUT_PATH = Path("output/MASTER_HV_DATASET_STEP1_LOCKED.csv")
OUTPUT_DIR = Path("output/plots_step3")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13
})

MODEL_COLORS = {
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728"
}


df = pd.read_csv(INPUT_PATH)

target_col = "PROPERTY: HV"
elem_cols = [c for c in df.columns if c.startswith("ELEM_")]

X = df[elem_cols].values
y = df[target_col].values


models = {

    "Extra Trees": Pipeline([
        ("model", ExtraTreesRegressor(
            n_estimators=900,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ))
    ]),

    "XGBoost": Pipeline([
        ("model", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ))
    ])
}


strat_labels = df["Source"]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
cv_preds = {}
feature_importance_store = {}

for name, pipeline in models.items():

    r2_scores, rmse_scores, mae_scores = [], [], []
    y_pred_full = np.zeros_like(y, dtype=float)
    importances = []

    for train_idx, test_idx in kf.split(X, strat_labels):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_pred_full[test_idx] = y_pred

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    cv_preds[name] = y_pred_full

    ci_low, ci_high = bootstrap_r2(y, y_pred_full)

    results.append([
        name,
        np.mean(r2_scores),
        np.std(r2_scores),
        np.mean(rmse_scores),
        np.mean(mae_scores),
        ci_low,
        ci_high
    ])

    if importances:
        feature_importance_store[name] = np.mean(importances, axis=0)


results_df = pd.DataFrame(
    results,
    columns=["Model", "R2_mean", "R2_std", "RMSE", "MAE", "CI_low", "CI_high"]
).sort_values("R2_mean", ascending=False)

results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)


for metric in ["R2_mean", "RMSE", "MAE"]:

    vals = results_df.set_index("Model")[metric]
    vals = vals.sort_values(ascending=(metric != "R2_mean"))

    plt.figure(figsize=(7, 4))
    plt.bar(vals.index, vals.values,
            color=[MODEL_COLORS[m] for m in vals.index])

    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"model_{metric}.png", dpi=600)
    plt.close()


plt.figure(figsize=(7, 7))

for name, y_pred in cv_preds.items():
    plt.scatter(y, y_pred,
                s=18, alpha=0.5,
                color=MODEL_COLORS[name],
                label=name)

lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--")

plt.xlabel("Experimental HV")
plt.ylabel("Predicted HV")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=600)
plt.close()


plt.figure(figsize=(7, 4))

for name, y_pred in cv_preds.items():
    sns.kdeplot(y - y_pred, label=name, linewidth=2)

plt.xlabel("Residual (HV)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residuals.png", dpi=600)
plt.close()


best_model = results_df.iloc[0]["Model"]

if best_model in feature_importance_store:

    imp = pd.Series(
        feature_importance_store[best_model],
        index=elem_cols
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(6, 4))
    imp[::-1].plot(kind="barh",
                   color=MODEL_COLORS[best_model],
                   edgecolor="black")

    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=600)
    plt.close()
