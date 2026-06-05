import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold


warnings.filterwarnings("ignore")

EPS = 1e-12
ACTIVE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.97
VARIANCE_THRESHOLD = 1e-5
R = 8.314
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

inp_dataset = Path("output/MASTER_MPEA_DATASET.csv")
inp_db = Path("output/UNIVERSAL_PROPERTY_DB.csv")

out_dir = Path("output/step3_descriptor_engine")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp_dataset)
prop_db = pd.read_csv(inp_db).set_index("Element")

target = "PROPERTY: HV"

elem_cols = sorted([
    column for column in df.columns
    if column.startswith("ELEM_")
])

elements = [
    column.replace("ELEM_", "")
    for column in elem_cols
]

supported_elements = [
    element for element in elements
    if element in prop_db.index
]

supported_cols = [
    f"ELEM_{element}"
    for element in supported_elements
]

coverage = df[supported_cols].sum(axis=1)

df["descriptor_coverage"] = coverage

df = df.loc[
    coverage >= 0.95
].reset_index(drop=True)

comp = df[supported_cols].copy()

comp = comp.div(
    comp.sum(axis=1),
    axis=0,
).fillna(0)

active_mask = comp >= ACTIVE_THRESHOLD


def weighted_avg(prop):
    values = np.zeros(len(comp))

    for element in supported_elements:
        values += (
            comp[f"ELEM_{element}"].values
            * prop_db.loc[element, prop]
        )

    return values


def weighted_var(prop):
    avg = weighted_avg(prop)
    values = np.zeros(len(comp))

    for element in supported_elements:
        values += (
            comp[f"ELEM_{element}"].values
            * (prop_db.loc[element, prop] - avg) ** 2
        )

    return values


def weighted_std(prop):
    return np.sqrt(weighted_var(prop))


d = pd.DataFrame(index=df.index)

entropy_comp = comp.where(active_mask, 0)

d["Smix"] = -R * np.sum(
    entropy_comp * np.log(entropy_comp + EPS),
    axis=1,
)

d["r_avg"] = weighted_avg("r")

d["delta"] = 100 * np.sqrt(
    sum(
        comp[f"ELEM_{element}"]
        * (
            1
            - prop_db.loc[element, "r"] / (d["r_avg"] + EPS)
        ) ** 2
        for element in supported_elements
    )
)

d["atomic_volume_avg"] = weighted_avg("atomic_volume")
d["volume_mismatch"] = weighted_std("atomic_volume")

d["packing_dispersion"] = (
    weighted_std("atomic_volume")
    / (d["atomic_volume_avg"] + EPS)
)

d["packing_frustration"] = (
    d["volume_mismatch"]
    * d["delta"]
)

d["Tm_avg"] = weighted_avg("Tm")
d["Tm_std"] = weighted_std("Tm")
d["cohesive_energy_avg"] = weighted_avg("cohesive_energy")

d["entropy_stabilization"] = (
    d["Smix"]
    * d["Tm_avg"]
)

d["entropy_distortion_balance"] = (
    d["Smix"]
    / (d["delta"] + EPS)
)

d["VEC_avg"] = weighted_avg("VEC")
d["VEC_std"] = weighted_std("VEC")
d["chi_avg"] = weighted_avg("chi")
d["chi_std"] = weighted_std("chi")
d["work_function_avg"] = weighted_avg("work_function")

d["electronic_distortion"] = (
    d["VEC_std"]
    * d["chi_std"]
)

d["vec_entropy_coupling"] = (
    d["VEC_avg"]
    * d["Smix"]
)

d["electronic_stability_index"] = (
    d["work_function_avg"]
    * d["cohesive_energy_avg"]
)

d["electron_density_mismatch"] = (
    weighted_std("VEC")
    * weighted_std("work_function")
)

d["bonding_incompatibility"] = (
    d["chi_std"]
    * weighted_std("cohesive_energy")
)

d["E_avg"] = weighted_avg("E")
d["G_avg"] = weighted_avg("G")
d["K_avg"] = weighted_avg("K")
d["poisson_avg"] = weighted_avg("poisson_ratio")
d["Pugh_ratio_avg"] = weighted_avg("Pugh_ratio")
d["specific_stiffness_avg"] = weighted_avg("specific_stiffness")
d["reduced_modulus_avg"] = weighted_avg("reduced_modulus_proxy")
d["elastic_anisotropy_avg"] = weighted_avg("elastic_anisotropy_proxy")

d["modulus_mismatch"] = (
    weighted_std("G")
    / (d["G_avg"] + EPS)
)

d["elastic_incompatibility"] = (
    weighted_std("E")
    * weighted_std("G")
)

d["shear_modulus_fluctuation"] = (
    weighted_std("G") ** 2
    / (d["G_avg"] + EPS)
)

d["local_strain_energy"] = (
    (d["delta"] / 100) ** 2
    * d["G_avg"]
)

d["modulus_distortion_strength"] = (
    d["G_avg"]
    * d["delta"]
)

d["shear_entropy_strength"] = (
    d["G_avg"]
    * d["Smix"]
)

d["density_strength_index"] = (
    d["G_avg"]
    / (weighted_avg("rho") + EPS)
)

d["thermoelastic_stability"] = (
    d["Tm_avg"]
    * d["specific_stiffness_avg"]
)

d["rho_avg"] = weighted_avg("rho")
d["rho_std"] = weighted_std("rho")

d["segregation_index"] = (
    d["rho_std"]
    * d["Tm_std"]
)

d["oxidation_sensitivity_avg"] = weighted_avg("oxidation_sensitivity")
d["cost_index_avg"] = weighted_avg("cost_index")

refractory_elements = prop_db[
    prop_db["is_refractory"] == 1
].index.tolist()

ref_cols = [
    f"ELEM_{element}"
    for element in refractory_elements
    if f"ELEM_{element}" in comp.columns
]

if len(ref_cols) > 0:
    d["refractory_fraction"] = comp[ref_cols].sum(axis=1)
else:
    d["refractory_fraction"] = 0.0

d["refractory_entropy_synergy"] = (
    d["refractory_fraction"]
    * d["Smix"]
)

d["dominant_fraction"] = comp.max(axis=1)

d["composition_complexity"] = (
    (comp > 0.05)
    .sum(axis=1)
)

d["active_elements"] = (
    (comp > ACTIVE_THRESHOLD)
    .sum(axis=1)
)

d["complexity_density"] = (
    d["composition_complexity"]
    / (d["rho_avg"] + EPS)
)

interaction_index = np.zeros(len(comp))

for i, element_i in enumerate(supported_elements):
    for j, element_j in enumerate(supported_elements):
        if j <= i:
            continue

        ci = comp[f"ELEM_{element_i}"]
        cj = comp[f"ELEM_{element_j}"]

        chi_i = prop_db.loc[element_i, "chi"]
        chi_j = prop_db.loc[element_j, "chi"]

        vec_i = prop_db.loc[element_i, "VEC"]
        vec_j = prop_db.loc[element_j, "VEC"]

        radius_i = prop_db.loc[element_i, "r"]
        radius_j = prop_db.loc[element_j, "r"]

        modulus_i = prop_db.loc[element_i, "G"]
        modulus_j = prop_db.loc[element_j, "G"]

        pair_term = (
            abs(chi_i - chi_j)
            + 0.15 * abs(vec_i - vec_j)
            + 0.05 * abs(radius_i - radius_j)
            + 0.002 * abs(modulus_i - modulus_j)
        )

        interaction_index += pair_term * ci * cj

d["interaction_mismatch_index"] = interaction_index

d["stability_competition_index"] = (
    d["Tm_avg"]
    * d["Smix"]
) / (
    d["interaction_mismatch_index"]
    + EPS
)

d["solid_solution_tendency"] = (
    d["stability_competition_index"]
    / (d["delta"] + EPS)
)

d["intermetallic_risk"] = (
    d["interaction_mismatch_index"]
    * d["chi_std"]
)

d["fcc_score"] = (
    1
    / (
        1
        + np.exp(
            -2 * (d["VEC_avg"] - 8)
        )
    )
)

d["bcc_score"] = (
    1
    / (
        1
        + np.exp(
            2 * (d["VEC_avg"] - 6.87)
        )
    )
)

d["phase_competition_index"] = (
    d["fcc_score"]
    * d["bcc_score"]
)

d["phase_boundary_distance"] = np.minimum(
    np.abs(d["VEC_avg"] - 8),
    np.abs(d["VEC_avg"] - 6.87),
)

d["electronic_phase_competition"] = (
    d["VEC_std"]
    * d["phase_competition_index"]
)

d["entropy_modulus_interaction"] = (
    d["Smix"]
    * d["G_avg"]
)

d["distortion_strength_interaction"] = (
    d["delta"]
    * d["G_avg"]
)

d["thermal_elastic_interaction"] = (
    d["Tm_avg"]
    * d["G_avg"]
)

d["bonding_strength_interaction"] = (
    d["chi_std"]
    * d["cohesive_energy_avg"]
)

d["electronic_thermal_competition"] = (
    d["VEC_std"]
    * d["Tm_std"]
)

d["metallurgical_validity"] = (
    (d["delta"] <= 15).astype(int)
    + (d["dominant_fraction"] <= 0.50).astype(int)
    + (d["composition_complexity"] >= 3).astype(int)
    + (d["active_elements"] >= 3).astype(int)
) / 4

numeric_cols = d.select_dtypes(include=np.number).columns

for column in numeric_cols:
    values = d[column].values

    if np.isnan(values).any():
        raise ValueError(f"NaN detected in {column}")

    if np.isinf(values).any():
        raise ValueError(f"Infinite value detected in {column}")

selector = VarianceThreshold(
    threshold=VARIANCE_THRESHOLD,
)

selector.fit(d)

keep_cols = d.columns[
    selector.get_support()
]

d = d[keep_cols]

corr = d.corr().abs()

upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1,
    ).astype(bool)
)

drop_cols = []

for column in upper.columns:
    high_corr = upper[column][
        upper[column] > CORRELATION_THRESHOLD
    ]

    if len(high_corr) > 0:
        for index_column in high_corr.index:
            remove_col = (
                column
                if len(column) > len(index_column)
                else index_column
            )

            if remove_col not in drop_cols:
                drop_cols.append(remove_col)

d = d.drop(
    columns=drop_cols,
    errors="ignore",
)

descriptor_origin = []

for column in d.columns:
    if any(key in column for key in ["avg", "std", "mismatch"]):
        origin = "physics_informed"
    elif any(key in column for key in ["interaction", "competition", "strength", "stability"]):
        origin = "metallurgy_engineered"
    else:
        origin = "descriptor_engineered"

    descriptor_origin.append([
        column,
        origin,
    ])

descriptor_provenance = pd.DataFrame(
    descriptor_origin,
    columns=[
        "Descriptor",
        "Origin",
    ],
)

descriptor_dataset = pd.concat(
    [
        df[[
            "SAMPLE_ID",
            target,
        ]],
        d,
    ],
    axis=1,
)

descriptor_dataset.to_csv(
    out_dir / "HEA_DESCRIPTOR_DATASET.csv",
    index=False,
)

d.describe().T.to_csv(
    out_dir / "DESCRIPTOR_STATISTICS.csv",
)

pd.DataFrame({
    "Descriptor": d.columns,
}).to_csv(
    out_dir / "DESCRIPTOR_LIST.csv",
    index=False,
)

descriptor_provenance.to_csv(
    out_dir / "DESCRIPTOR_PROVENANCE.csv",
    index=False,
)

descriptor_path = out_dir / "HEA_DESCRIPTOR_DATASET.csv"

with open(descriptor_path, "rb") as file:
    descriptor_sha256 = hashlib.sha256(file.read()).hexdigest()

print(f"Input alloys retained: {len(d)}")
print(f"Generated descriptors: {len(d.columns)}")
print(f"Removed descriptors: {len(drop_cols)}")
print(f"SHA256: {descriptor_sha256}")
print(f"Output folder: {out_dir}")
