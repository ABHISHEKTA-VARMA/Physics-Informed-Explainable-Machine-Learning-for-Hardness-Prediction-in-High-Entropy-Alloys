import os
import hashlib
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.992
VARIANCE_THRESHOLD = 1e-6
N_SPLITS = 5
N_REPEATS = 3
ENSEMBLE_SEEDS = 3

DATA_PATH = Path("output/step3_descriptor_engine/HEA_DESCRIPTOR_DATASET.csv")
OUTPUT_DIR = Path("output/step4_descriptor_ml")
TARGET = "PROPERTY: HV"

PRIORITY_FEATURES = [
    "Smix",
    "delta",
    "Tm_avg",
    "Tm_std",
    "VEC_avg",
    "VEC_std",
    "chi_avg",
    "chi_std",
    "G_avg",
    "K_avg",
    "E_avg",
    "rho_avg",
    "Pugh_ratio_avg",
    "interaction_mismatch_index",
    "stability_competition_index",
    "modulus_mismatch",
    "volume_mismatch",
    "entropy_stabilization",
    "electronic_distortion",
    "phase_competition_index",
    "refractory_fraction",
    "thermoelastic_stability",
    "local_strain_energy",
    "shear_entropy_strength",
]

COLORS = {
    "Dummy": "gray",
    "Ridge": "#9467bd",
    "ElasticNet": "#8c564b",
    "Extra Trees": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "Hist Gradient Boosting": "#17becf",
    "XGBoost": "#d62728",
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
    })
    sns.set_style("whitegrid")


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_descriptors(df):
    descriptor_cols = [
        column for column in df.columns
        if column not in [TARGET, "SAMPLE_ID"]
    ]

    X_df = df[descriptor_cols].copy()

    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_vt = vt.fit_transform(X_df)
    kept_cols = X_df.columns[vt.get_support()]

    X_df = pd.DataFrame(
        X_vt,
        columns=kept_cols,
        index=X_df.index,
    )

    corr = X_df.corr().abs()
    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)
    )

    drop_cols = []

    for column in upper.columns:
        high_corr = upper[column][upper[column] > CORRELATION_THRESHOLD]

        for index_column in high_corr.index:
            if index_column in PRIORITY_FEATURES:
                continue

            if index_column not in drop_cols:
                drop_cols.append(index_column)

    X_df = X_df.drop(
        columns=drop_cols,
        errors="ignore",
    )

    return X_df


def build_models():
    return {
        "Dummy": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),

        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(
                alpha=1.5,
                random_state=RANDOM_STATE,
            )),
        ]),

        "ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.0008,
                l1_ratio=0.35,
                max_iter=20000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=850,
                max_features=0.82,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=False,
                random_state=RANDOM_STATE,
                n_jobs=2,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=850,
                max_features=0.72,
                min_samples_leaf=1,
                min_samples_split=2,
                bootstrap=False,
                random_state=RANDOM_STATE,
                n_jobs=2,
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=850,
                learning_rate=0.022,
                max_depth=5,
                subsample=0.92,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=6,
                max_iter=700,
                min_samples_leaf=4,
                l2_regularization=0.001,
                random_state=RANDOM_STATE,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_estimators=850,
                max_depth=6,
                learning_rate=0.022,
                subsample=0.92,
                colsample_bytree=0.90,
                gamma=0,
                min_child_weight=1,
                reg_alpha=0.0001,
                reg_lambda=1.2,
                random_state=RANDOM_STATE,
                n_jobs=2,
                verbosity=0,
            )),
        ]),
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


def evaluate_models(X_df, y, y_bins, models):
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    cv_preds = {}
    cv_uncertainty = {}
    feature_importances = {}
    feature_importance_std = {}

    for name, pipe in models.items():
        print(f"  {name}", flush=True)

        r2_scores = []
        rmse_scores = []
        mae_scores = []
        y_oof = np.zeros_like(y, dtype=float)
        y_unc = np.zeros_like(y, dtype=float)
        fold_importances = []

        for tr, te in cv.split(X_df, y_bins):
            X_tr = X_df.iloc[tr]
            X_te = X_df.iloc[te]
            y_tr = y[tr]
            y_te = y[te]
            local_preds = []

            for seed in range(ENSEMBLE_SEEDS):
                model = clone(pipe)

                if hasattr(model.named_steps["model"], "random_state"):
                    model.named_steps["model"].set_params(
                        random_state=RANDOM_STATE + seed,
                    )

                model.fit(X_tr, y_tr)
                local_preds.append(model.predict(X_te))

            local_preds = np.vstack(local_preds)
            yp = local_preds.mean(axis=0)
            unc = local_preds.std(axis=0)

            y_oof[te] = yp
            y_unc[te] = unc

            r2_scores.append(r2_score(y_te, yp))
            rmse_scores.append(
                np.sqrt(
                    mean_squared_error(
                        y_te,
                        yp,
                    )
                )
            )
            mae_scores.append(
                mean_absolute_error(
                    y_te,
                    yp,
                )
            )

            try:
                fitted_model = clone(pipe)
                fitted_model.fit(X_tr, y_tr)

                perm = permutation_importance(
                    fitted_model,
                    X_te,
                    y_te,
                    n_repeats=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
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

    return results, cv_preds, cv_uncertainty, feature_importances, feature_importance_std


def save_core_outputs(
    output_dir,
    X_df,
    y,
    sample_ids,
    models,
    results,
    cv_preds,
    cv_uncertainty,
    feature_importances,
    feature_importance_std,
):
    results.to_csv(
        output_dir / "descriptor_results.csv",
        index=False,
    )

    best_model_name = results.iloc[0]["Model"]
    best_model = clone(
        models[best_model_name]
    )

    best_model.fit(
        X_df,
        y,
    )

    joblib.dump(
        best_model,
        output_dir / "best_descriptor_model.pkl",
    )

    pd.DataFrame({
        "Descriptor_Features": X_df.columns,
    }).to_csv(
        output_dir / "descriptor_features.csv",
        index=False,
    )

    best_preds = cv_preds[best_model_name]
    best_unc = cv_uncertainty[best_model_name]
    best_residuals = y - best_preds

    if best_model_name in feature_importances:
        importance_df = pd.DataFrame({
            "Feature": X_df.columns,
            "Mean_Importance": feature_importances[best_model_name],
            "Importance_STD": feature_importance_std[best_model_name],
        })

        importance_df = importance_df.sort_values(
            "Mean_Importance",
            ascending=False,
        )

        importance_df.to_csv(
            output_dir / "feature_importance.csv",
            index=False,
        )

    plt.figure(figsize=(7, 7))
    plt.scatter(
        y,
        best_preds,
        s=24,
        alpha=0.65,
        color=COLORS[best_model_name],
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
        output_dir / "parity_plot.png",
        dpi=700,
    )
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.kdeplot(
        best_residuals,
        linewidth=2,
        color=COLORS[best_model_name],
    )
    plt.xlabel("Residual (HV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "residual_distribution.png",
        dpi=700,
    )
    plt.close()

    abs_error = np.abs(y - best_preds)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        best_unc,
        abs_error,
        s=22,
        alpha=0.65,
        color=COLORS[best_model_name],
    )
    plt.xlabel("Prediction Uncertainty")
    plt.ylabel("Absolute Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "uncertainty_calibration.png",
        dpi=700,
    )
    plt.close()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    cov = EmpiricalCovariance()
    cov.fit(X_scaled)
    mahal = cov.mahalanobis(X_scaled)

    ad_df = pd.DataFrame({
        "SAMPLE_ID": sample_ids,
        "Mahalanobis_Distance": mahal,
    })

    ad_df.to_csv(
        output_dir / "applicability_domain.csv",
        index=False,
    )

    pred_df = pd.DataFrame({
        "SAMPLE_ID": sample_ids,
        "Experimental_HV": y,
    })

    for name, yp in cv_preds.items():
        pred_df[f"{name}_Pred"] = yp

    pred_df.to_csv(
        output_dir / "crossval_predictions.csv",
        index=False,
    )

    unc_df = pd.DataFrame({
        "SAMPLE_ID": sample_ids,
        "Experimental_HV": y,
    })

    for name, unc in cv_uncertainty.items():
        unc_df[f"{name}_UNC"] = unc

    unc_df.to_csv(
        output_dir / "prediction_uncertainty.csv",
        index=False,
    )


def save_diagnostic_outputs(output_dir, data_path):
    results = pd.read_csv(output_dir / "descriptor_results.csv")
    preds_df = pd.read_csv(output_dir / "crossval_predictions.csv")
    ad_df = pd.read_csv(output_dir / "applicability_domain.csv")
    df = pd.read_csv(data_path)

    best_model_name = results.iloc[0]["Model"]
    y = preds_df["Experimental_HV"].values
    best_preds = preds_df[f"{best_model_name}_Pred"].values
    sample_ids = preds_df["SAMPLE_ID"].values
    mahal = ad_df["Mahalanobis_Distance"].values

    best_residuals = y - best_preds
    abs_error = np.abs(best_residuals)

    for metric in ["R2", "RMSE", "MAE"]:
        vals = results.set_index("Model")[metric]
        vals = vals.sort_values(ascending=(metric != "R2"))

        plt.figure(figsize=(7, 4))
        plt.bar(
            vals.index,
            vals.values,
            color=[COLORS.get(model, "gray") for model in vals.index],
            edgecolor="black",
        )
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=700)
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(
        best_preds,
        best_residuals,
        s=20,
        alpha=0.6,
        color=COLORS.get(best_model_name, "gray"),
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted HV")
    plt.ylabel("Residual (HV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_predicted.png", dpi=700)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(mahal, bins=30, edgecolor="black")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mahalanobis_domain.png", dpi=700)
    plt.close()

    X_bp = sm.add_constant(best_preds)
    bp_test = het_breuschpagan(best_residuals, X_bp)

    pd.DataFrame({
        "Metric": ["LM Statistic", "LM p-value", "F Statistic", "F p-value"],
        "Value": bp_test,
    }).to_csv(output_dir / "heteroscedasticity_test.csv", index=False)

    failure_df = pd.DataFrame({
        "SAMPLE_ID": sample_ids,
        "Experimental_HV": y,
        "Predicted_HV": best_preds,
        "Absolute_Error": abs_error,
    })

    meta_cols = ["PHASE_CLASS", "PROCESS_CLASS", "SOURCE"]
    available_meta = [c for c in meta_cols if c in df.columns]

    if available_meta:
        residual_df = pd.DataFrame({"Residual": best_residuals})

        for c in available_meta:
            residual_df[c] = df[c]
            failure_df[c] = df[c]

        for meta_col in available_meta:
            plt.figure(figsize=(8, 4))
            sns.boxplot(data=residual_df, x=meta_col, y="Residual")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(output_dir / f"residuals_by_{meta_col}.png", dpi=700)
            plt.close()

    failure_df.sort_values("Absolute_Error", ascending=False).head(50).to_csv(
        output_dir / "worst_predictions.csv",
        index=False,
    )


def save_hash(output_dir):
    results_file = output_dir / "descriptor_results.csv"

    with open(results_file, "rb") as file:
        sha256 = hashlib.sha256(file.read()).hexdigest()

    pd.DataFrame({
        "File": [results_file.name],
        "SHA256": [sha256],
    }).to_csv(output_dir / "descriptor_results_sha256.csv", index=False)


def show_summary(output_dir, X_df, df):
    results = pd.read_csv(output_dir / "descriptor_results.csv")
    best_row = results.iloc[0]

    summary = pd.DataFrame({
        "Item": [
            "Best model",
            "R2",
            "RMSE",
            "MAE",
            "Retained descriptors",
            "Samples",
            "Output directory",
        ],
        "Value": [
            best_row["Model"],
            f"{best_row['R2']:.4f}",
            f"{best_row['RMSE']:.4f}",
            f"{best_row['MAE']:.4f}",
            len(X_df.columns),
            len(df),
            str(output_dir),
        ],
    })

    print(summary.to_string(index=False))


def main():
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    print("Loading dataset...", flush=True)
    df = load_data(DATA_PATH)
    print(f"Samples: {len(df)}", flush=True)
    print(f"Input file: {DATA_PATH}", flush=True)

    X_df = prepare_descriptors(df)
    print(f"\nRetained descriptors: {X_df.shape[1]}", flush=True)

    y = df[TARGET].values
    sample_ids = df["SAMPLE_ID"].values

    y_bins = pd.qcut(
        y,
        q=5,
        labels=False,
        duplicates="drop",
    )

    models = build_models()

    print("\nEvaluating models...", flush=True)
    results, cv_preds, cv_uncertainty, feature_importances, feature_importance_std = evaluate_models(
        X_df,
        y,
        y_bins,
        models,
    )

    print("\nSaving outputs...", flush=True)
    save_core_outputs(
        OUTPUT_DIR,
        X_df,
        y,
        sample_ids,
        models,
        results,
        cv_preds,
        cv_uncertainty,
        feature_importances,
        feature_importance_std,
    )

    save_diagnostic_outputs(OUTPUT_DIR, DATA_PATH)
    save_hash(OUTPUT_DIR)

    print()
    show_summary(OUTPUT_DIR, X_df, df)


if __name__ == "__main__":
    main()
