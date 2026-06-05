import hashlib
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_ITERATIONS = 8
TOP_K_PER_ITER = 1500
LOCAL_PERTURBATIONS = 12
EPS = 1e-12

save_dir = Path("output/step5b_adaptive_optimization")
save_dir.mkdir(parents=True, exist_ok=True)

candidate_path = Path("output/step5a_global_exploration/global_candidate_design_space.csv")
feature_path = Path("output/step4_descriptor_ml/descriptor_features.csv")
descriptor_dataset_path = Path("output/step3_descriptor_engine/HEA_DESCRIPTOR_DATASET.csv")
prop_db_path = Path("output/UNIVERSAL_PROPERTY_DB.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
})
sns.set_style("whitegrid")

candidate_df = pd.read_csv(candidate_path)
feature_df = pd.read_csv(feature_path)
descriptor_dataset = pd.read_csv(descriptor_dataset_path)

prop_db = pd.read_csv(prop_db_path)
prop_db = prop_db.set_index("Element")

feature_names = feature_df.iloc[:, 0].tolist()

elements = [
    column for column in candidate_df.columns
    if column in prop_db.index
]

train_descriptor_space = descriptor_dataset[feature_names].copy()

X_train = descriptor_dataset[feature_names].copy()
y_train = descriptor_dataset["PROPERTY: HV"].values

ensemble_models = {
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=700,
        max_features=0.70,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=700,
        max_features=0.60,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.025,
        max_depth=4,
        subsample=0.90,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.025,
        max_depth=6,
        max_iter=700,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_estimators=700,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.90,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    ),
}

trained_models = {}

print("Training ensemble models")

for name, model in ensemble_models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained")


def weighted_avg(frame, prop):
    return sum(
        frame[element] * prop_db.loc[element, prop]
        for element in elements
    )


def weighted_var(frame, prop):
    avg = weighted_avg(frame, prop)

    return sum(
        frame[element] * (prop_db.loc[element, prop] - avg) ** 2
        for element in elements
    )


def build_descriptors(design_df):
    d = pd.DataFrame(index=design_df.index)
    gas_constant = 8.314

    d["Smix"] = -gas_constant * np.sum(
        design_df[elements] * np.log(design_df[elements] + EPS),
        axis=1,
    )

    d["r_avg"] = weighted_avg(design_df, "r")

    d["delta"] = 100 * np.sqrt(
        sum(
            design_df[element]
            * (
                1
                - prop_db.loc[element, "r"] / (d["r_avg"] + EPS)
            ) ** 2
            for element in elements
        )
    )

    d["VEC_avg"] = weighted_avg(design_df, "VEC")
    d["chi_avg"] = weighted_avg(design_df, "chi")
    d["E_avg"] = weighted_avg(design_df, "E")
    d["G_avg"] = weighted_avg(design_df, "G")
    d["K_avg"] = weighted_avg(design_df, "K")
    d["rho_avg"] = weighted_avg(design_df, "rho")
    d["Tm_avg"] = weighted_avg(design_df, "Tm")

    d["Tm_std"] = np.sqrt(weighted_var(design_df, "Tm"))
    d["chi_std"] = np.sqrt(weighted_var(design_df, "chi"))
    d["VEC_std"] = np.sqrt(weighted_var(design_df, "VEC"))

    d["modulus_mismatch"] = (
        np.sqrt(weighted_var(design_df, "G"))
        / (d["G_avg"] + EPS)
    )

    d["volume_mismatch"] = np.sqrt(
        weighted_var(design_df, "atomic_volume")
    )

    d["electronic_distortion"] = (
        d["VEC_std"]
        * d["chi_std"]
    )

    d["entropy_stabilization"] = (
        d["Smix"]
        * d["Tm_avg"]
    )

    d["local_strain_energy"] = (
        (d["delta"] / 100) ** 2
        * d["G_avg"]
    )

    d["shear_entropy_strength"] = (
        d["G_avg"]
        * d["Smix"]
    )

    d["thermoelastic_stability"] = (
        d["Tm_avg"]
        * (d["E_avg"] / (d["rho_avg"] + EPS))
    )

    d["fcc_score"] = (
        1
        / (1 + np.exp(-2 * (d["VEC_avg"] - 8)))
    )

    d["bcc_score"] = (
        1
        / (1 + np.exp(2 * (d["VEC_avg"] - 6.87)))
    )

    d["phase_competition_index"] = (
        d["fcc_score"]
        * d["bcc_score"]
    )

    refractory_elements = [
        element for element in elements
        if prop_db.loc[element, "is_refractory"] == 1
    ]

    d["refractory_fraction"] = sum(
        design_df[element]
        for element in refractory_elements
    )

    d["dominant_fraction"] = design_df[elements].max(axis=1)

    d["active_elements"] = (
        (design_df[elements] > 0.03)
        .sum(axis=1)
    )

    for feature in feature_names:
        if feature not in d.columns:
            d[feature] = 0.0

    d = d[feature_names]

    return d


design_space = candidate_df[elements].copy()

scaler_ad = StandardScaler()

X_train_scaled = scaler_ad.fit_transform(
    train_descriptor_space
)

nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(X_train_scaled)

all_iteration_results = []

print("Starting adaptive optimization")

for iteration in range(1, N_ITERATIONS + 1):
    print(f"Iteration {iteration}")

    descriptor_df = build_descriptors(design_space)
    X_pred = descriptor_df[feature_names].copy()

    ensemble_predictions = []

    for model in trained_models.values():
        ensemble_predictions.append(model.predict(X_pred))

    ensemble_predictions = np.vstack(ensemble_predictions)

    descriptor_df["Predicted_HV"] = ensemble_predictions.mean(axis=0)
    descriptor_df["Prediction_Uncertainty"] = ensemble_predictions.std(axis=0)

    X_pred_scaled = scaler_ad.transform(X_pred)

    distances, _ = nbrs.kneighbors(X_pred_scaled)

    descriptor_df["Descriptor_Distance"] = distances.mean(axis=1)

    iso = IsolationForest(
        contamination=0.05,
        random_state=RANDOM_STATE,
    )

    descriptor_df["Outlier_Flag"] = iso.fit_predict(X_pred_scaled)

    def normalize(values):
        return (
            (values - values.min())
            / (values.max() - values.min() + EPS)
        )

    descriptor_df["Acquisition_Score"] = (
        0.40 * normalize(descriptor_df["Predicted_HV"])
        + 0.15 * normalize(descriptor_df["thermoelastic_stability"])
        + 0.10 * normalize(descriptor_df["entropy_stabilization"])
        + 0.10 * normalize(descriptor_df["shear_entropy_strength"])
        + 0.10 * (1 - normalize(descriptor_df["Prediction_Uncertainty"]))
        + 0.10 * (1 - normalize(descriptor_df["Descriptor_Distance"]))
        + 0.05 * (1 - normalize(descriptor_df["dominant_fraction"]))
    )

    valid_idx = (
        (descriptor_df["delta"] <= 9.0)
        & (descriptor_df["dominant_fraction"] <= 0.40)
        & (descriptor_df["active_elements"] >= 4)
        & (descriptor_df["active_elements"] <= 8)
        & (
            descriptor_df["Descriptor_Distance"]
            <= descriptor_df["Descriptor_Distance"].quantile(0.95)
        )
        & (
            descriptor_df["Prediction_Uncertainty"]
            <= descriptor_df["Prediction_Uncertainty"].quantile(0.95)
        )
        & (descriptor_df["Outlier_Flag"] == 1)
    )

    descriptor_df = descriptor_df.loc[
        valid_idx
    ].reset_index(drop=True)

    design_space = design_space.loc[
        valid_idx
    ].reset_index(drop=True)

    sorted_idx = np.argsort(
        -descriptor_df["Acquisition_Score"].values
    )

    descriptor_df = descriptor_df.iloc[
        sorted_idx
    ].reset_index(drop=True)

    design_space = design_space.iloc[
        sorted_idx
    ].reset_index(drop=True)

    current_df = pd.concat(
        [design_space, descriptor_df],
        axis=1,
    )

    current_df["Iteration"] = iteration

    all_iteration_results.append(current_df)

    print(f"Remaining alloys: {len(current_df)}")
    print(f"Best predicted HV: {current_df['Predicted_HV'].max():.2f}")
    print(f"Lowest uncertainty: {current_df['Prediction_Uncertainty'].min():.2f}")

    elite_df = current_df.head(TOP_K_PER_ITER).copy()

    new_designs = []

    for _, row in elite_df.iterrows():
        base_comp = row[elements].values.copy()

        active_mask = base_comp >= 0.03
        active_indices = np.where(active_mask)[0]

        if len(active_indices) < 4:
            continue

        if len(active_indices) > 9:
            continue

        for _ in range(LOCAL_PERTURBATIONS):
            new_comp = np.zeros(len(elements))

            perturbed = (
                base_comp[active_indices]
                + rng.normal(
                    loc=0,
                    scale=0.020,
                    size=len(active_indices),
                )
            )

            perturbed = np.clip(perturbed, 0, None)

            dropout_mask = (
                rng.random(len(active_indices))
                > 0.15
            )

            perturbed = perturbed * dropout_mask

            if perturbed.sum() <= 0:
                continue

            perturbed = perturbed / perturbed.sum()

            new_comp[active_indices] = perturbed
            new_comp[new_comp < 0.015] = 0

            if new_comp.sum() <= 0:
                continue

            new_comp = new_comp / new_comp.sum()

            active_count = np.sum(new_comp >= 0.03)
            dominant_fraction = new_comp.max()
            retained_fraction = new_comp[new_comp >= 0.03].sum()

            if active_count < 4:
                continue

            if active_count > 8:
                continue

            if dominant_fraction > 0.40:
                continue

            if retained_fraction < 0.92:
                continue

            sorted_comp = np.sort(new_comp[new_comp > 0])[::-1]

            if len(sorted_comp) >= 6:
                top6_fraction = sorted_comp[:6].sum()

                if top6_fraction < 0.80:
                    continue

            new_designs.append(new_comp)

    if len(new_designs) == 0:
        raise ValueError("Adaptive exploration produced no feasible alloys.")

    design_space = pd.DataFrame(
        new_designs,
        columns=elements,
    )

    signature = (
        (design_space * 100)
        .round()
        .astype(int)
        .astype(str)
        .agg("-".join, axis=1)
    )

    design_space = (
        design_space.loc[~signature.duplicated()]
        .reset_index(drop=True)
    )

    print(f"Generated feasible alloys: {len(design_space)}")

final_results = pd.concat(
    all_iteration_results,
    axis=0,
)

final_results = (
    final_results
    .sort_values("Acquisition_Score", ascending=False)
    .reset_index(drop=True)
)

final_results["signature"] = (
    (final_results[elements] * 10)
    .round()
    .astype(int)
    .astype(str)
    .agg("-".join, axis=1)
)

final_results = (
    final_results
    .drop_duplicates("signature")
    .drop(columns="signature")
    .reset_index(drop=True)
)

top_alloys = final_results.head(25).copy()

top_alloys["Alloy_ID"] = [
    f"OPT_ALLOY_{index + 1}"
    for index in range(len(top_alloys))
]


def composition_string(row):
    return ", ".join([
        f"{element}:{row[element]:.2f}"
        for element in elements
        if row[element] >= 0.03
    ])


top_alloys["Composition"] = top_alloys.apply(
    composition_string,
    axis=1,
)

final_results.to_csv(
    save_dir / "adaptive_optimization_results.csv",
    index=False,
)

top_alloys.to_csv(
    save_dir / "top_optimized_alloys.csv",
    index=False,
)

result_file = save_dir / "top_optimized_alloys.csv"

with open(result_file, "rb") as file:
    sha256 = hashlib.sha256(file.read()).hexdigest()

print("Top optimized alloys")
print(
    top_alloys[
        [
            "Alloy_ID",
            "Predicted_HV",
            "Prediction_Uncertainty",
            "Descriptor_Distance",
            "Acquisition_Score",
            "Composition",
        ]
    ]
)

print(f"Output folder: {save_dir}")
print(f"SHA256: {sha256}")
