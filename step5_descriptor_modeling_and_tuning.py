import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


RANDOM_STATE = 42

INPUT_PATH = Path("output/HEA_descriptor_dataset.csv")
OUTPUT_DIR = Path("output/plots_step5")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15
})

MODEL_COLORS = {
    "Extra Trees (tuned)": "#ff7f0e",
    "Random Forest": "#1f77b4",
    "Gradient Boosting": "#2ca02c",
    "XGBoost (tuned)": "#d62728",
    "Stacking ensemble": "#9467bd"
}


df = pd.read_csv(INPUT_PATH)

target_col = "PROPERTY: HV"

df = df[df[target_col].notnull() & (df[target_col] > 0)].copy()

X = df.drop(columns=[target_col, "Source"], errors="ignore")
X = X.select_dtypes(include=np.number)

X = X.loc[:, X.notnull().any()]
X = X.loc[:, (X != 0).any(axis=0)]

y = df[target_col].values


strat_labels = df["Source"]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


rf = RandomForestRegressor(
    n_estimators=800,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    random_state=RANDOM_STATE
)


param_dist = {
    "model__n_estimators": [500, 700, 900],
    "model__max_depth": [4, 5, 6],
    "model__learning_rate": [0.03, 0.05],
    "model__subsample": [0.7, 0.85],
    "model__colsample_bytree": [0.7, 0.85]
}


et_param_dist = {
    "model__n_estimators": [500, 700, 900, 1200],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}


results = []
cv_preds = {}
feature_store = {}

for name, base_model in {
    "Extra Trees (tuned)": "et",
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "XGBoost (tuned)": "xgb"
}.items():

    r2_list, rmse_list, mae_list = [], [], []
    preds_all = np.zeros(len(y))
    importances = []

    for train_idx, test_idx in kf.split(X, strat_labels):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        z = np.abs((y_train - y_train.mean()) / y_train.std())
        mask = z < 3

        X_train = X_train.iloc[mask]
        y_train = y_train[mask]

        if name in ["XGBoost (tuned)", "Extra Trees (tuned)"]:

            if name == "XGBoost (tuned)":
                model_base = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0
                )
                param_grid_use = param_dist

            else:
                model_base = ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
                param_grid_use = et_param_dist

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model_base)
            ])

            inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid_use,
                n_iter=20,
                scoring="r2",
                cv=inner_cv,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            model = search.best_estimator_

        else:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", base_model)
            ])
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        preds_all[test_idx] = y_pred

        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))

        if hasattr(model.named_steps["model"], "feature_importances_"):
            importances.append(model.named_steps["model"].feature_importances_)

    cv_preds[name] = preds_all

    results.append([
        name,
        np.mean(r2_list),
        np.std(r2_list),
        np.mean(rmse_list),
        np.mean(mae_list)
    ])

    if importances:
        feature_store[name] = np.mean(importances, axis=0)


results_df = pd.DataFrame(
    results,
    columns=["Model", "R2_mean", "R2_std", "RMSE", "MAE"]
).sort_values("R2_mean", ascending=False)

results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)


for metric in ["R2_mean", "RMSE", "MAE"]:

    vals = results_df.set_index("Model")[metric]
    vals = vals.sort_values(ascending=(metric != "R2_mean"))

    plt.figure(figsize=(7, 4))
    plt.bar(
        vals.index,
        vals.values,
        color=[MODEL_COLORS[m] for m in vals.index]
    )

    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"model_comparison_{metric}.png", dpi=600)
    plt.close()


plt.figure(figsize=(7, 7))

for name, preds in cv_preds.items():
    plt.scatter(
        y, preds,
        s=18,
        alpha=0.5,
        color=MODEL_COLORS[name],
        label=name
    )

lims = [y.min(), y.max()]
plt.plot(lims, lims, "k--")

plt.xlabel("Experimental hardness (HV)")
plt.ylabel("Predicted hardness (HV)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted_all_models.png", dpi=600)
plt.close()


plt.figure(figsize=(7, 4))

for name, preds in cv_preds.items():
    sns.kdeplot(y - preds, label=name, linewidth=2)

plt.xlabel("Residual (experimental − predicted)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=600)
plt.close()


best_model = results_df.iloc[0]["Model"]

if best_model in feature_store:

    imp = pd.Series(
        feature_store[best_model],
        index=X.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(6, 4))
    imp[::-1].plot(
        kind="barh",
        color=MODEL_COLORS[best_model],
        edgecolor="black"
    )

    plt.xlabel("Relative importance")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=600)
    plt.close()


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

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=600)
plt.close()


best_model = results_df.iloc[0]["Model"]

ml_preds = cv_preds[best_model]

ml_df = pd.DataFrame({
    "Alloy_ID": np.arange(len(ml_preds)),
    "ML_Hardness_HV": ml_preds
})

ml_df.to_csv(OUTPUT_DIR / "ml_predictions.csv", index=False)
