import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.covariance import EmpiricalCovariance
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATASET_VERSION = "step2_baseline_composition_v1"

np.random.seed(RANDOM_STATE)

inp = Path("output/MASTER_MPEA_DATASET.csv")
out_dir = Path("output/step2_baseline_ml")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
})
sns.set_style("whitegrid")

df = pd.read_csv(inp)

target = "PROPERTY: HV"

elem_cols = [
    column for column in df.columns
    if column.startswith("ELEM_")
]

required_cols = [
    "SAMPLE_ID",
    "PHASE_CLASS",
    "PROCESS_CLASS",
    "SOURCE",
    "COMPOSITION_SIGNATURE",
]

for column in required_cols:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

if len(elem_cols) == 0:
    raise ValueError("No ELEM_* columns detected.")

X_df = df[elem_cols].copy()
X_df = X_df.fillna(0)

y = df[target].values
X = X_df.values
groups = df["COMPOSITION_SIGNATURE"].values

y_bins = pd.qcut(
    y,
    q=5,
    labels=False,
    duplicates="drop",
)

oof_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

metric_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE + 10,
)

models = {
    "Dummy": Pipeline([
        ("m", DummyRegressor(strategy="mean")),
    ]),

    "Ridge": Pipeline([
        ("s", StandardScaler()),
        ("m", Ridge(
            alpha=1.0,
            random_state=RANDOM_STATE,
        )),
    ]),

    "ElasticNet": Pipeline([
        ("s", StandardScaler()),
        ("m", ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            random_state=RANDOM_STATE,
            max_iter=10000,
        )),
    ]),

    "Extra Trees": Pipeline([
        ("m", ExtraTreesRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=False,
            random_state=RANDOM_STATE,
            n_jobs=2,
        )),
    ]),

    "Random Forest": Pipeline([
        ("m", RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=2,
        )),
    ]),

    "Gradient Boosting": Pipeline([
        ("m", GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            random_state=RANDOM_STATE,
        )),
    ]),

    "XGBoost": Pipeline([
        ("m", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.2,
            min_child_weight=2,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=0,
        )),
    ]),
}

colors = {
    "Dummy": "gray",
    "Ridge": "#9467bd",
    "ElasticNet": "#8c564b",
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost": "#d62728",
}


def bootstrap_r2(y_true, y_pred, n=1000):
    rng = np.random.default_rng(RANDOM_STATE)
    vals = []

    for _ in range(n):
        idx = rng.choice(
            len(y_true),
            len(y_true),
            replace=True,
        )

        vals.append(
            r2_score(
                y_true[idx],
                y_pred[idx],
            )
        )

    return np.percentile(vals, [2.5, 97.5])


rows = []
cv_preds = {}
cv_uncertainty = {}
feature_importances = {}
feature_importance_std = {}
fold_tracking = []

for name, pipe in models.items():
    print(f"Training {name}")

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for tr, te in metric_cv.split(X, y_bins, groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        model = clone(pipe)
        model.fit(X_tr, y_tr)

        yp = model.predict(X_te)

        r2_scores.append(r2_score(y_te, yp))
        rmse_scores.append(
            np.sqrt(mean_squared_error(y_te, yp))
        )
        mae_scores.append(mean_absolute_error(y_te, yp))

    y_oof = np.zeros_like(y, dtype=float)
    y_unc = np.zeros_like(y, dtype=float)
    fold_importances = []

    for fold, (tr, te) in enumerate(
        oof_cv.split(X, y_bins, groups),
        start=1,
    ):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        local_preds = []

        for seed in range(10):
            model = clone(pipe)

            if hasattr(model.named_steps["m"], "random_state"):
                model.named_steps["m"].set_params(
                    random_state=RANDOM_STATE + seed,
                )

            model.fit(X_tr, y_tr)
            yp_local = model.predict(X_te)

            local_preds.append(yp_local)

        local_preds = np.vstack(local_preds)

        y_oof[te] = local_preds.mean(axis=0)
        y_unc[te] = local_preds.std(axis=0)

        for idx in te:
            fold_tracking.append({
                "SAMPLE_ID": df.iloc[idx]["SAMPLE_ID"],
                "Model": name,
                "Fold": fold,
            })

        try:
            perm_model = clone(pipe)
            perm_model.fit(X_tr, y_tr)

            perm = permutation_importance(
                perm_model,
                X_te,
                y_te,
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=2,
            )

            fold_importances.append(perm.importances_mean)

        except Exception:
            pass

    cv_preds[name] = y_oof
    cv_uncertainty[name] = y_unc

    ci_l, ci_h = bootstrap_r2(y, y_oof)

    abs_error = np.abs(y - y_oof)

    pear_corr = pearsonr(y_unc, abs_error)[0]
    spear_corr = spearmanr(y_unc, abs_error)[0]

    rows.append([
        name,
        np.mean(r2_scores),
        np.std(r2_scores),
        np.mean(rmse_scores),
        np.std(rmse_scores),
        np.mean(mae_scores),
        np.std(mae_scores),
        ci_l,
        ci_h,
        pear_corr,
        spear_corr,
    ])

    if len(fold_importances) > 0:
        feature_importances[name] = np.mean(
            fold_importances,
            axis=0,
        )

        feature_importance_std[name] = np.std(
            fold_importances,
            axis=0,
        )

pd.DataFrame(fold_tracking).to_csv(
    out_dir / "cv_fold_assignments.csv",
    index=False,
)

results = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "R2",
        "R2_STD",
        "RMSE",
        "RMSE_STD",
        "MAE",
        "MAE_STD",
        "CI_LOW",
        "CI_HIGH",
        "UNC_PEARSON",
        "UNC_SPEARMAN",
    ],
)

results = results.sort_values(
    "R2",
    ascending=False,
)

results.to_csv(
    out_dir / "baseline_results.csv",
    index=False,
)

best_model_name = results.iloc[0]["Model"]

best_model = clone(
    models[best_model_name]
)

final_model = best_model.fit(X, y)

joblib.dump(
    final_model,
    out_dir / "best_baseline_model.pkl",
)

joblib.dump(
    elem_cols,
    out_dir / "baseline_feature_names.pkl",
)

pd.DataFrame({
    "Composition_Features": elem_cols,
}).to_csv(
    out_dir / "composition_features.csv",
    index=False,
)

best_preds = cv_preds[best_model_name]
best_residuals = y - best_preds

X_bp = sm.add_constant(best_preds)

bp_test = het_breuschpagan(
    best_residuals,
    X_bp,
)

pd.DataFrame({
    "Metric": [
        "LM Statistic",
        "LM p-value",
        "F Statistic",
        "F p-value",
    ],
    "Value": bp_test,
}).to_csv(
    out_dir / "heteroscedasticity_test.csv",
    index=False,
)

for metric in ["R2", "RMSE", "MAE"]:
    vals = results.set_index("Model")[metric]

    if metric == "R2":
        vals = vals.sort_values(ascending=False)
    else:
        vals = vals.sort_values(ascending=True)

    plt.figure(figsize=(7, 4))

    plt.bar(
        vals.index,
        vals.values,
        color=[colors[model] for model in vals.index],
        edgecolor="black",
    )

    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{metric}.png",
        dpi=700,
    )

    plt.close()

plt.figure(figsize=(7, 7))

plt.scatter(
    y,
    best_preds,
    s=22,
    alpha=0.6,
    color=colors[best_model_name],
)

lims = [
    y.min(),
    y.max(),
]

plt.plot(
    lims,
    lims,
    "k--",
    linewidth=1.5,
)

plt.xlabel("Experimental HV")
plt.ylabel("Predicted HV")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    out_dir / "best_model_parity.png",
    dpi=700,
)

plt.close()

plt.figure(figsize=(7, 4))

sns.kdeplot(
    best_residuals,
    linewidth=2,
    color=colors[best_model_name],
)

plt.xlabel("Residual (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    out_dir / "residual_distribution.png",
    dpi=700,
)

plt.close()

plt.figure(figsize=(6, 5))

plt.scatter(
    best_preds,
    best_residuals,
    s=20,
    alpha=0.6,
    color=colors[best_model_name],
)

plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5,
)

plt.xlabel("Predicted HV")
plt.ylabel("Residual (HV)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    out_dir / "residual_vs_predicted.png",
    dpi=700,
)

plt.close()

residual_df = pd.DataFrame({
    "Residual": best_residuals,
    "PHASE_CLASS": df["PHASE_CLASS"],
    "PROCESS_CLASS": df["PROCESS_CLASS"],
    "SOURCE": df["SOURCE"],
})

for meta_col in [
    "PHASE_CLASS",
    "PROCESS_CLASS",
    "SOURCE",
]:
    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=residual_df,
        x=meta_col,
        y="Residual",
    )

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        out_dir / f"residuals_by_{meta_col}.png",
        dpi=700,
    )

    plt.close()

if best_model_name in feature_importances:
    importance_table = pd.DataFrame({
        "Feature": elem_cols,
        "Mean_Importance": feature_importances[best_model_name],
        "Importance_STD": feature_importance_std[best_model_name],
    })

    importance_table = importance_table.sort_values(
        "Mean_Importance",
        ascending=False,
    )

    importance_table.to_csv(
        out_dir / "feature_importance_stability.csv",
        index=False,
    )

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

cov = EmpiricalCovariance()
cov.fit(X_scaled)

mahal = cov.mahalanobis(X_scaled)

plt.figure(figsize=(6, 4))

plt.hist(
    mahal,
    bins=30,
    edgecolor="black",
)

plt.xlabel("Mahalanobis Distance")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    out_dir / "mahalanobis_domain.png",
    dpi=700,
)

plt.close()

uncertainty = cv_uncertainty[best_model_name]

abs_error = np.abs(y - best_preds)

plt.figure(figsize=(6, 5))

plt.scatter(
    uncertainty,
    abs_error,
    s=20,
    alpha=0.6,
    color=colors[best_model_name],
)

plt.xlabel("Prediction Uncertainty")
plt.ylabel("Absolute Error")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    out_dir / "uncertainty_calibration.png",
    dpi=700,
)

plt.close()

failure_df = pd.DataFrame({
    "SAMPLE_ID": df["SAMPLE_ID"],
    "Experimental_HV": y,
    "Predicted_HV": best_preds,
    "Absolute_Error": abs_error,
    "PHASE_CLASS": df["PHASE_CLASS"],
    "SOURCE": df["SOURCE"],
})

failure_df = failure_df.sort_values(
    "Absolute_Error",
    ascending=False,
)

failure_df.head(50).to_csv(
    out_dir / "worst_predictions.csv",
    index=False,
)

pred_df = pd.DataFrame({
    "SAMPLE_ID": df["SAMPLE_ID"],
    "Experimental_HV": y,
})

for name, yp in cv_preds.items():
    pred_df[f"{name}_Pred"] = yp

pred_df.to_csv(
    out_dir / "crossval_predictions.csv",
    index=False,
)

unc_df = pd.DataFrame({
    "SAMPLE_ID": df["SAMPLE_ID"],
    "Experimental_HV": y,
})

for name, unc in cv_uncertainty.items():
    unc_df[f"{name}_UNC"] = unc

unc_df.to_csv(
    out_dir / "prediction_uncertainty.csv",
    index=False,
)

print(results)
print(f"Composition features: {len(elem_cols)}")
print(f"Dataset size: {len(df)}")
print(f"Best baseline model: {best_model_name}")
print(f"Dataset version: {DATASET_VERSION}")
print(f"Output folder: {out_dir}")
