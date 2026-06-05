import hashlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EPS = 1e-12

N_FINAL_ALLOYS = 10
N_CLUSTERS = 5
MIN_COMPOSITION_DISTANCE = 0.08

inp = Path("output/step5b_adaptive_optimization/adaptive_optimization_results.csv")

save_dir = Path("output/step5c_final_fem_alloys")
save_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
})
sns.set_style("whitegrid")

if not inp.exists():
    raise FileNotFoundError(f"Missing optimization results: {inp}")

df = pd.read_csv(inp)

known_elements = [
    "Al",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Hf",
    "Mn",
    "Mo",
    "Nb",
    "Ni",
    "Ru",
    "Si",
    "Ta",
    "Ti",
    "V",
    "W",
    "Zr",
]

elements = [
    element for element in known_elements
    if element in df.columns
]

df["active_elements"] = (
    df[elements] >= 0.05
).sum(axis=1)

df["dominant_fraction"] = df[elements].max(axis=1)

retained_fraction = []

for _, row in df.iterrows():
    retained_fraction.append(
        row[elements][row[elements] >= 0.03].sum()
    )

df["retained_fraction"] = retained_fraction

min_active = max(
    4,
    int(df["active_elements"].quantile(0.10)),
)

max_active = min(
    8,
    int(df["active_elements"].quantile(0.90)),
)

dominance_limit = min(
    0.45,
    df["dominant_fraction"].quantile(0.90),
)

retained_limit = max(
    0.80,
    df["retained_fraction"].quantile(0.10),
)

df = df.loc[
    (df["active_elements"] >= min_active)
    & (df["active_elements"] <= max_active)
    & (df["dominant_fraction"] <= dominance_limit)
    & (df["retained_fraction"] >= retained_limit)
].reset_index(drop=True)

if len(df) < 500:
    df = pd.read_csv(inp)

    df["active_elements"] = (
        df[elements] >= 0.04
    ).sum(axis=1)

    df["dominant_fraction"] = df[elements].max(axis=1)

    retained_fraction = []

    for _, row in df.iterrows():
        retained_fraction.append(
            row[elements][row[elements] >= 0.02].sum()
        )

    df["retained_fraction"] = retained_fraction

    df = df.loc[
        (df["active_elements"] >= 4)
        & (df["active_elements"] <= 9)
        & (df["dominant_fraction"] <= 0.50)
        & (df["retained_fraction"] >= 0.75)
    ].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("No alloys survived governance.")

uncertainty_limit = np.percentile(
    df["Prediction_Uncertainty"],
    90,
)

distance_limit = np.percentile(
    df["Descriptor_Distance"],
    90,
)

df = df.loc[
    (df["Prediction_Uncertainty"] <= uncertainty_limit)
    & (df["Descriptor_Distance"] <= distance_limit)
].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("No alloys survived reliability filtering.")


def normalize(values):
    return (
        (values - values.min())
        / (values.max() - values.min() + EPS)
    )


df["manufacturability_index"] = (
    1 - normalize(df["dominant_fraction"])
)

df["FEM_Score"] = (
    0.25 * normalize(df["Predicted_HV"])
    + 0.15 * normalize(df["thermoelastic_stability"])
    + 0.10 * normalize(df["entropy_stabilization"])
    + 0.10 * normalize(df["shear_entropy_strength"])
    + 0.10 * normalize(df["refractory_fraction"])
    + 0.10 * (1 - normalize(df["Prediction_Uncertainty"]))
    + 0.10 * (1 - normalize(df["Descriptor_Distance"]))
    + 0.10 * normalize(df["manufacturability_index"])
)

family_labels = []

for _, row in df.iterrows():
    if row["refractory_fraction"] >= df["refractory_fraction"].quantile(0.75):
        family = "Refractory_Strengthened"
    elif row["delta"] >= df["delta"].quantile(0.75):
        family = "Distortion_Strengthened"
    elif row["entropy_stabilization"] >= df["entropy_stabilization"].quantile(0.75):
        family = "Entropy_Stabilized"
    else:
        family = "Balanced_MPEA"

    family_labels.append(family)

df["Physics_Family"] = family_labels

cluster_features = [
    "Predicted_HV",
    "delta",
    "VEC_avg",
    "G_avg",
    "Tm_avg",
    "entropy_stabilization",
    "thermoelastic_stability",
    "refractory_fraction",
]

scaled = StandardScaler().fit_transform(
    df[cluster_features]
)

cluster_model = KMeans(
    n_clusters=min(N_CLUSTERS, len(df)),
    random_state=RANDOM_STATE,
    n_init=50,
)

df["Cluster"] = cluster_model.fit_predict(scaled)

selected_rows = []

family_targets = {
    "Refractory_Strengthened": 2,
    "Distortion_Strengthened": 2,
    "Entropy_Stabilized": 2,
    "Balanced_MPEA": 2,
}

for family, target_count in family_targets.items():
    family_df = (
        df.loc[df["Physics_Family"] == family]
        .sort_values("FEM_Score", ascending=False)
        .reset_index(drop=True)
    )

    count = 0

    for _, row in family_df.iterrows():
        current_comp = row[elements].values
        accept = True

        for previous in selected_rows:
            previous_comp = previous[elements].values
            distance = np.linalg.norm(current_comp - previous_comp)

            if distance < MIN_COMPOSITION_DISTANCE:
                accept = False
                break

        if accept:
            selected_rows.append(row)
            count += 1

        if count >= target_count:
            break

remaining_df = (
    df.sort_values("FEM_Score", ascending=False)
    .reset_index(drop=True)
)

for _, row in remaining_df.iterrows():
    if len(selected_rows) >= N_FINAL_ALLOYS:
        break

    current_comp = row[elements].values
    accept = True

    for previous in selected_rows:
        previous_comp = previous[elements].values
        distance = np.linalg.norm(current_comp - previous_comp)

        if distance < MIN_COMPOSITION_DISTANCE:
            accept = False
            break

    if accept:
        selected_rows.append(row)

final_alloys = pd.DataFrame(selected_rows).reset_index(drop=True)

if len(final_alloys) == 0:
    raise ValueError("No final alloys survived selection.")

final_alloys["FEM_Alloy_ID"] = [
    f"FEM_ALLOY_{index + 1}"
    for index in range(len(final_alloys))
]


def composition_string(row):
    return ", ".join([
        f"{element}:{row[element]:.2f}"
        for element in elements
        if row[element] >= 0.03
    ])


final_alloys["Composition"] = final_alloys.apply(
    composition_string,
    axis=1,
)

final_alloys.to_csv(
    save_dir / "final_fem_candidate_alloys.csv",
    index=False,
)

summary_cols = [
    "FEM_Alloy_ID",
    "Physics_Family",
    "Predicted_HV",
    "Prediction_Uncertainty",
    "Descriptor_Distance",
    "FEM_Score",
    "Composition",
]

final_alloys[summary_cols].to_csv(
    save_dir / "final_fem_summary.csv",
    index=False,
)

pca_features = [
    "Predicted_HV",
    "delta",
    "VEC_avg",
    "G_avg",
    "Tm_avg",
    "refractory_fraction",
    "entropy_stabilization",
    "thermoelastic_stability",
]

scaled_pca = StandardScaler().fit_transform(
    final_alloys[pca_features]
)

pca = PCA(n_components=2)
proj = pca.fit_transform(scaled_pca)

plt.figure(figsize=(7, 6))

scatter = plt.scatter(
    proj[:, 0],
    proj[:, 1],
    c=final_alloys["Predicted_HV"],
    s=120,
    edgecolor="black",
    alpha=0.9,
)

for index, alloy_id in enumerate(final_alloys["FEM_Alloy_ID"]):
    plt.annotate(
        alloy_id,
        (proj[index, 0], proj[index, 1]),
        fontsize=10,
    )

cbar = plt.colorbar(scatter)
cbar.set_label("Predicted HV")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()

plt.savefig(
    save_dir / "final_fem_alloy_space.png",
    dpi=700,
)

plt.close()

plot_df = final_alloys.sort_values(
    "Predicted_HV",
    ascending=True,
)

plt.figure(figsize=(8, 6))

bars = plt.barh(
    plot_df["FEM_Alloy_ID"],
    plot_df["Predicted_HV"],
    edgecolor="black",
)

for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 2,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}",
        va="center",
    )

plt.xlabel("Predicted Hardness (HV)")
plt.ylabel("Final FEM Alloy")
plt.tight_layout()

plt.savefig(
    save_dir / "final_fem_hardness.png",
    dpi=700,
)

plt.close()

result_file = save_dir / "final_fem_candidate_alloys.csv"

with open(result_file, "rb") as file:
    sha256 = hashlib.sha256(file.read()).hexdigest()

print("Final FEM candidates")
print(
    final_alloys[
        [
            "FEM_Alloy_ID",
            "Physics_Family",
            "Predicted_HV",
            "Prediction_Uncertainty",
            "Descriptor_Distance",
            "FEM_Score",
            "Composition",
        ]
    ]
)

print(f"Selected alloys: {len(final_alloys)}")
print(f"Output folder: {save_dir}")
print(f"SHA256: {sha256}")
