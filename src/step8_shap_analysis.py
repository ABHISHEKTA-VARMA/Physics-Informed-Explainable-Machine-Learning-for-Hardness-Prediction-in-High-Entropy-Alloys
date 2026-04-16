import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor


np.random.seed(42)

INPUT_PATH = Path("output/HEA_descriptor_dataset.csv")
OUTPUT_DIR = Path("output/plots_step8_shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

target_col = "PROPERTY: HV"

df = df[df[target_col] > 0].copy()

X = df.select_dtypes(include=np.number).drop(columns=[target_col])
y = df[target_col].values

strat_labels = df["Source"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

shap_values_all = []
X_all = []

for train_idx, test_idx in kf.split(X, strat_labels):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y[train_idx]

    z = np.abs((y_train - y_train.mean()) / y_train.std())
    mask = z < 3

    X_train = X_train.iloc[mask]
    y_train = y_train[mask]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(
            n_estimators=900,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    X_test_imp = pipe.named_steps["imputer"].transform(X_test)

    shap_vals = explainer(X_test_imp, check_additivity=False)

    shap_values_all.append(shap_vals.values)
    X_all.append(X_test_imp)


shap_values = np.vstack(shap_values_all)
X_all = np.vstack(X_all)

X_all_df = pd.DataFrame(X_all, columns=X.columns)

# keep only descriptor features (remove ELEM_)
feature_names = X.columns
valid_idx = [i for i, col in enumerate(feature_names) if not col.startswith("ELEM_")]

shap_values = shap_values[:, valid_idx]
X_all_df = X_all_df.iloc[:, valid_idx]
X_all_df.columns = [feature_names[i] for i in valid_idx]


plt.figure()
shap.summary_plot(shap_values, X_all_df, show=False)
plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=600, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_all_df, plot_type="bar", show=False)
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=600, bbox_inches="tight")
plt.close()


mean_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X_all_df.columns,
    "Importance": mean_shap
}).sort_values("Importance", ascending=False)

shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)


top_feats = shap_df["Feature"].head(3)

for feat in top_feats:
    shap.dependence_plot(
        feat,
        shap_values,
        X_all_df,
        show=False
    )
    plt.savefig(OUTPUT_DIR / f"dependence_{feat}.png", dpi=600, bbox_inches="tight")
    plt.close()


mid = len(X_all_df) // 2

plt.figure(figsize=(8, 6))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[mid],
        base_values=np.mean(shap_values),
        data=X_all_df.iloc[mid],
        feature_names=X_all_df.columns
    ),
    max_display=12,
    show=False
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "local_waterfall.png", dpi=600, bbox_inches="tight")
plt.close()


top15 = shap_df["Feature"].head(15)

X_scaled = (X_all_df[top15] - X_all_df[top15].mean()) / X_all_df[top15].std()

corr = X_scaled.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation.png", dpi=600, bbox_inches="tight")
plt.close()


vif_df = pd.DataFrame({
    "Feature": top15,
    "VIF": [
        variance_inflation_factor(X_scaled.values, i)
        for i in range(len(top15))
    ]
})

vif_df.to_csv(OUTPUT_DIR / "vif.csv", index=False)

print("Step 8 complete")
