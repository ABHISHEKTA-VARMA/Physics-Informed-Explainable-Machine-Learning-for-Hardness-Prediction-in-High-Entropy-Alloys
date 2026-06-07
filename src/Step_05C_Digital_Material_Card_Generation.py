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

print("Loading candidate data...")
df = pd.read_csv(inp)
print(f"Candidates loaded: {len(df)}")

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
    raise ValueError("No alloys satisfied selection criteria.")

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
    raise ValueError(
        "No alloys satisfied uncertainty and descriptor-distance criteria."
    )

print(f"Candidates after filtering: {len(df)}")


def normalize(values):
    return (
        (values - values.min())
        / (values.max() - values.min() + EPS)
    )


df["manufacturability_index"] = (
    1 - normalize(df["dominant_fraction"])
)

df["Candidate_Score"] = (
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
        family = "Refractory_Dominant"
    elif row["delta"] >= df["delta"].quantile(0.75):
        family = "Distortion_Dominant"
    elif row["entropy_stabilization"] >= df["entropy_stabilization"].quantile(0.75):
        family = "Entropy_Stabilized"
    else:
        family = "Balanced_MPEA"

    family_labels.append(family)

df["Candidate_Family"] = family_labels

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
    "Refractory_Dominant": 2,
    "Distortion_Dominant": 2,
    "Entropy_Stabilized": 2,
    "Balanced_MPEA": 2,
}

print("Selecting final candidate alloys...")

for family, target_count in family_targets.items():
    family_df = (
        df.loc[df["Candidate_Family"] == family]
        .sort_values("Candidate_Score", ascending=False)
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
    df.sort_values("Candidate_Score", ascending=False)
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
    raise ValueError("No final candidate alloys satisfied final selection criteria.")

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
    "Candidate_Family",
    "Predicted_HV",
    "Prediction_Uncertainty",
    "Descriptor_Distance",
    "Candidate_Score",
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
plt.ylabel("Final Candidate Alloy")
plt.tight_layout()

plt.savefig(
    save_dir / "final_fem_hardness.png",
    dpi=700,
)

plt.close()

result_file = save_dir / "final_fem_candidate_alloys.csv"

with open(result_file, "rb") as file:
    sha256 = hashlib.sha256(file.read()).hexdigest()

print("Final candidate alloys selected for finite element analysis")

print(
    final_alloys[
        [
            "FEM_Alloy_ID",
            "Candidate_Family",
            "Predicted_HV",
            "Prediction_Uncertainty",
            "Descriptor_Distance",
            "Candidate_Score",
            "Composition",
        ]
    ]
)

print(f"Selected alloys: {len(final_alloys)}")
print(f"Output folder: {save_dir}")
print(f"SHA256: {sha256}")

print("Generating publication figures...")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "axes.linewidth": 1.5,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
})

sns.set_theme(style="ticks", context="paper")

# Figure 3.4.1: Candidate screening distributions
try:
    global_df = pd.read_csv(
        "output/step5a_global_exploration/global_candidate_design_space.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    sns.kdeplot(
        data=global_df,
        x="Smix_estimate",
        fill=True,
        color="#8c92ac",
        alpha=0.6,
        ax=axes[0],
        cut=0,
    )

    axes[0].set_title(
        "(a) Initial Candidate Space",
        fontweight="bold",
    )
    axes[0].set_xlabel(
        "Mixing Entropy (J/K.mol)",
        fontweight="bold",
    )
    axes[0].set_ylabel(
        "Probability Density",
        fontweight="bold",
    )

    sns.kdeplot(
        data=df,
        x="Predicted_HV",
        fill=True,
        color="#d62728",
        alpha=0.8,
        ax=axes[1],
        cut=0,
    )

    axes[1].set_title(
        "(b) Selected Candidate Population",
        fontweight="bold",
    )
    axes[1].set_xlabel(
        "Predicted Hardness (HV)",
        fontweight="bold",
    )
    axes[1].set_ylabel("")

    plt.tight_layout()

    plt.savefig(
        save_dir / "Fig3_4_1_Candidate_Screening.png",
        dpi=600,
    )

    plt.close()

    print("Figure 3.4.1 saved.")

except Exception as e:
    print(f"Skipping Figure 3.4.1: {e}")

# Figure 3.4.2: Elemental distribution heatmap
try:
    heatmap_data = final_alloys[elements].copy()
    heatmap_data.index = final_alloys["FEM_Alloy_ID"]

    heatmap_data = heatmap_data.loc[
        :,
        (heatmap_data != 0).any(axis=0),
    ]

    heatmap_data = heatmap_data * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    mask = heatmap_data == 0

    sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap="rocket_r",
        annot=True,
        fmt=".1f",
        annot_kws={
            "size": 12,
            "weight": "bold",
        },
        cbar_kws={
            "label": "Atomic Percentage (at%)",
        },
        linewidths=2,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel(
        "Active Alloying Elements",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Selected MPEA Candidates",
        fontweight="bold",
    )
    ax.set_title(
        "Elemental Distribution Across Selected Candidate Alloys",
        pad=15,
        fontweight="bold",
    )

    plt.xticks(rotation=0, fontweight="bold")
    plt.yticks(rotation=0, fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        save_dir / "Fig3_4_2_Chemical_Heatmap.png",
        dpi=600,
    )

    plt.close()

    print("Figure 3.4.2 saved.")

except Exception as e:
    print(f"Skipping Figure 3.4.2: {e}")

# Figure 3.4.3: Adaptive optimization convergence
try:
    from matplotlib.ticker import MaxNLocator

    opt_history = pd.read_csv(
        "output/step5b_adaptive_optimization/adaptive_optimization_results.csv"
    )

    convergence_data = (
        opt_history.groupby("Iteration")["Predicted_HV"]
        .max()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        convergence_data["Iteration"],
        convergence_data["Predicted_HV"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color="#1f77b4",
    )

    max_iter = convergence_data["Iteration"].max()
    inflection = int(max_iter * 0.3) if max_iter > 10 else 3

    ax.axvspan(
        0,
        inflection,
        color="lightgray",
        alpha=0.3,
        label="Initial Search Stage",
    )

    ax.axvspan(
        inflection,
        max_iter,
        color="#d62728",
        alpha=0.1,
        label="Refinement Stage",
    )

    ax.set_xlabel(
        "Optimization Iteration",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Maximum Predicted Hardness (HV)",
        fontweight="bold",
    )
    ax.set_title(
        "Adaptive Optimization Convergence",
        pad=15,
        fontweight="bold",
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        save_dir / "Fig3_4_3_Optimization_Convergence.png",
        dpi=600,
    )

    plt.close()

    print("Figure 3.4.3 saved.")

except Exception as e:
    print(f"Skipping Figure 3.4.3: {e}")
