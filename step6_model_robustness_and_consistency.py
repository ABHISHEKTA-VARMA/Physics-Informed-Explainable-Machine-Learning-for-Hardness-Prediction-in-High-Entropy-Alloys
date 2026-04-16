import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor


np.random.seed(42)


INPUT_PATH = Path("output/HEA_descriptor_dataset.csv")
OUTPUT_DIR = Path("output/algorithm_consistency")
OUTPUT_DIR.mkdir(exist_ok=True)


sns.set_style("whitegrid")


df = pd.read_csv(INPUT_PATH)

target_col = "PROPERTY: HV"

df = df[df[target_col].notnull() & (df[target_col] > 0)].copy()

X = df.drop(columns=[target_col, "Source"], errors="ignore")
X = X.select_dtypes(exclude="object")

X = X.loc[:, X.notnull().any()]
X = X.loc[:, (X != 0).any(axis=0)]

y = df[target_col].values


seeds = [0, 21, 42, 77, 100]
results = []

strat_labels = df["Source"]

for seed in seeds:

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    r2_list, rmse_list, mae_list = [], [], []
    dummy_r2_list = []

    for train_idx, test_idx in kf.split(X, strat_labels):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        z = np.abs((y_train - y_train.mean()) / y_train.std())
        mask = z < 3

        X_train = X_train.iloc[mask]
        y_train = y_train[mask]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))

        dummy_r2_list.append(r2_score(y_test, y_dummy))

    results.append([
        seed,
        np.mean(r2_list),
        np.std(r2_list),
        np.mean(rmse_list),
        np.mean(mae_list),
        np.mean(dummy_r2_list)
    ])


consistency_df = pd.DataFrame(
    results,
    columns=["Seed", "R2_mean", "R2_std", "RMSE", "MAE", "Dummy_R2"]
)

consistency_df.to_csv(
    OUTPUT_DIR / "algorithm_consistency_results.csv",
    index=False
)


summary_df = pd.DataFrame({
    "Metric": ["R2", "RMSE", "MAE"],
    "Mean": [
        consistency_df["R2_mean"].mean(),
        consistency_df["RMSE"].mean(),
        consistency_df["MAE"].mean()
    ],
    "Std": [
        consistency_df["R2_mean"].std(),
        consistency_df["RMSE"].std(),
        consistency_df["MAE"].std()
    ]
})


plt.figure(figsize=(7, 4))
plt.plot(consistency_df["Seed"], consistency_df["R2_mean"], marker="o")

plt.xlabel("Random seed")
plt.ylabel("R2")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "r2_stability.png", dpi=600)
plt.close()


perm_scores = []
rng = np.random.default_rng(42)

for i in range(50):

    y_perm = rng.permutation(y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

    fold_scores = []

    for train_idx, test_idx in kf.split(X, strat_labels):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_perm[train_idx], y_perm[test_idx]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                max_features="sqrt",
                random_state=i,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_scores.append(r2_score(y_test, y_pred))

    perm_scores.append(np.mean(fold_scores))


plt.figure(figsize=(6, 4))
sns.histplot(perm_scores, bins=10)

plt.xlabel("Permutation R2")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "permutation_test.png", dpi=600)
plt.close()
