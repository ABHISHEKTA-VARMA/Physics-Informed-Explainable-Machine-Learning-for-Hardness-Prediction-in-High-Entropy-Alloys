import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

# Configuration and Paths
RANDOM_STATE = 42
TARGET_COL = "PROPERTY: HV"
SEEDS = [0, 21, 42, 77, 100]
PERMUTATIONS = 50

out_dir = Path("output/step4b_algorithm_consistency")
out_dir.mkdir(parents=True, exist_ok=True)

data_path = Path("output/step3_descriptor_engine/HEA_DESCRIPTOR_DATASET.csv")
features_path = Path("output/step4_descriptor_ml/descriptor_features.csv")
model_path = Path("output/step4_descriptor_ml/best_descriptor_model.pkl")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
})
sns.set_style("whitegrid")

# Load Data and Model
df = pd.read_csv(data_path)
df = df[df[TARGET_COL].notnull() & (df[TARGET_COL] > 0)].copy()

features = pd.read_csv(features_path).iloc[:, 0].tolist()
X = df[features]
y = df[TARGET_COL].values

if len(X) < 50:
    raise ValueError("Dataset too small for stable cross-validation.")

best_model_pipeline = joblib.load(model_path)

def clone_and_seed(pipeline, seed):
    cloned = clone(pipeline)
    estimator = cloned.named_steps.get("model")
    # Explicitly check "is not None" to avoid truthiness evaluation on unfitted ensemble
    if estimator is not None and hasattr(estimator, "random_state"):
        estimator.set_params(random_state=seed)
    return cloned

# 1. Seed Stability Analysis
rows = []
y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

for s in SEEDS:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    r2_s, rmse_s, mae_s, dummy_s = [], [], [], []

    for tr, te in kf.split(X, y_bins):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        # Outlier isolation per fold
        z = np.abs((y_tr - y_tr.mean()) / (y_tr.std() + 1e-9))
        mask = z < 3
        X_tr, y_tr = X_tr.iloc[mask], y_tr[mask]

        model = clone_and_seed(best_model_pipeline, s)
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_tr, y_tr)
        yd = dummy.predict(X_te)

        r2_s.append(r2_score(y_te, yp))
        rmse_s.append(np.sqrt(mean_squared_error(y_te, yp)))
        mae_s.append(mean_absolute_error(y_te, yp))
        dummy_s.append(r2_score(y_te, yd))

    rows.append([
        s, np.mean(r2_s), np.std(r2_s),
        np.mean(rmse_s), np.mean(mae_s), np.mean(dummy_s)
    ])

cons = pd.DataFrame(rows, columns=["Seed", "R2_mean", "R2_std", "RMSE", "MAE", "Dummy_R2"])
cons["R2_gain"] = cons["R2_mean"] - cons["Dummy_R2"]
cons.to_csv(out_dir / "consistency_results.csv", index=False)

pd.DataFrame({
    "Metric": ["R2", "RMSE", "MAE"],
    "Mean": [cons["R2_mean"].mean(), cons["RMSE"].mean(), cons["MAE"].mean()],
    "Std": [cons["R2_mean"].std(), cons["RMSE"].std(), cons["MAE"].std()]
}).to_csv(out_dir / "summary.csv", index=False)

plt.figure(figsize=(6, 4))
plt.plot(cons["Seed"], cons["R2_mean"], marker="o", color="#1f77b4", linewidth=2)
plt.axhline(cons["R2_mean"].mean(), color="black", linestyle="--", alpha=0.5)
plt.xlabel("Random Seed")
plt.ylabel(r"Cross-Validated $R^2$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "r2_stability.png", dpi=700)
plt.close()

# 2. Y-Randomization (Permutation) Test
perm_scores = []
rng = np.random.default_rng(RANDOM_STATE)

for i in range(PERMUTATIONS):
    y_perm = rng.permutation(y)
    y_perm_bins = pd.qcut(y_perm, q=5, labels=False, duplicates="drop")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    fold_scores = []

    for tr, te in kf.split(X, y_perm_bins):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_perm[tr], y_perm[te]

        model = clone_and_seed(best_model_pipeline, i)
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        fold_scores.append(r2_score(y_te, yp))

    perm_scores.append(np.mean(fold_scores))

plt.figure(figsize=(6, 4))
sns.histplot(perm_scores, bins=12, color="#d62728", edgecolor="black")
plt.axvline(cons["R2_mean"].mean(), color="black", linestyle="--", linewidth=2, label="True Model")
plt.xlabel(r"Permutation $R^2$")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "permutation_test.png", dpi=700)
plt.close()

print("Algorithm consistency analysis complete.")
