import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import dirichlet


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_GLOBAL_SAMPLES = 150000
MIN_ACTIVE_FRACTION = 0.02
MIN_ELEMENT_OCCURRENCE = 8
EPS = 1e-12
R = 8.314

dataset_path = Path("output/step3_descriptor_engine/HEA_DESCRIPTOR_DATASET.csv")
property_db_path = Path("output/UNIVERSAL_PROPERTY_DB.csv")
raw_dataset_path = Path("output/MASTER_MPEA_DATASET.csv")

out_dir = Path("output/step5a_global_exploration")
out_dir.mkdir(parents=True, exist_ok=True)

dataset = pd.read_csv(dataset_path)
raw_df = pd.read_csv(raw_dataset_path)

prop_db = pd.read_csv(property_db_path)
prop_db = prop_db.set_index("Element")

elem_cols = [
    column for column in raw_df.columns
    if column.startswith("ELEM_")
]

all_elements = [
    column.replace("ELEM_", "")
    for column in elem_cols
]

valid_elements = []
element_occurrence = {}

for element in all_elements:
    column = f"ELEM_{element}"

    if column not in raw_df.columns:
        continue

    occurrence = (raw_df[column] > 0.01).sum()
    element_occurrence[element] = occurrence

    if occurrence >= MIN_ELEMENT_OCCURRENCE and element in prop_db.index:
        valid_elements.append(element)

valid_elements = sorted(valid_elements)

required_properties = [
    "r",
    "VEC",
    "chi",
    "Tm",
    "E",
    "G",
    "K",
    "rho",
    "atomic_volume",
]

elements = []

for element in valid_elements:
    valid = True

    for prop in required_properties:
        if prop not in prop_db.columns:
            valid = False
            break

        if pd.isna(prop_db.loc[element, prop]):
            valid = False
            break

    if valid:
        elements.append(element)

refractory_elements = [
    element for element in elements
    if (
        "is_refractory" in prop_db.columns
        and prop_db.loc[element, "is_refractory"] == 1
    )
]

light_elements = [
    element for element in elements
    if element in ["Al", "Ti", "Mg", "Li", "Be"]
]

distortion_elements = [
    element for element in elements
    if element in ["Zr", "Hf", "Ta", "Nb", "W"]
]

frequency_weights = np.array([
    element_occurrence[element]
    for element in elements
])

frequency_weights = frequency_weights / frequency_weights.sum()

sampling_modes = []

sampling_modes.append({
    "name": "balanced_HEA",
    "alpha": 2.0 + 6.0 * frequency_weights,
    "fraction": 0.25,
})

alpha_ref = np.ones(len(elements)) * 1.2

for index, element in enumerate(elements):
    if element in refractory_elements:
        alpha_ref[index] += 3.5

sampling_modes.append({
    "name": "refractory_strengthened",
    "alpha": alpha_ref,
    "fraction": 0.25,
})

alpha_dist = np.ones(len(elements)) * 1.5

for index, element in enumerate(elements):
    if element in distortion_elements:
        alpha_dist[index] += 4.0

sampling_modes.append({
    "name": "distortion_engineered",
    "alpha": alpha_dist,
    "fraction": 0.20,
})

sampling_modes.append({
    "name": "entropy_maximized",
    "alpha": np.ones(len(elements)) * 3.5,
    "fraction": 0.20,
})

alpha_tm = np.ones(len(elements)) * 2.0

for index, element in enumerate(elements):
    if element in light_elements:
        alpha_tm[index] *= 0.7

sampling_modes.append({
    "name": "transition_balanced",
    "alpha": alpha_tm,
    "fraction": 0.10,
})

design_frames = []

for mode in sampling_modes:
    n_mode = int(N_GLOBAL_SAMPLES * mode["fraction"])

    samples = dirichlet.rvs(
        alpha=mode["alpha"],
        size=n_mode,
        random_state=rng,
    )

    mode_df = pd.DataFrame(
        samples,
        columns=elements,
    )

    mode_df["sampling_mode"] = mode["name"]
    design_frames.append(mode_df)

design_space = pd.concat(
    design_frames,
    axis=0,
).reset_index(drop=True)

design_space["active_elements"] = (
    design_space[elements] >= MIN_ACTIVE_FRACTION
).sum(axis=1)

if len(refractory_elements) > 0:
    design_space["refractory_fraction"] = sum(
        design_space[element]
        for element in refractory_elements
    )
else:
    design_space["refractory_fraction"] = 0.0

if len(distortion_elements) > 0:
    design_space["distortion_fraction"] = sum(
        design_space[element]
        for element in distortion_elements
    )
else:
    design_space["distortion_fraction"] = 0.0

if len(light_elements) > 0:
    design_space["light_fraction"] = sum(
        design_space[element]
        for element in light_elements
    )
else:
    design_space["light_fraction"] = 0.0

design_space["max_fraction"] = design_space[elements].max(axis=1)

design_space["Smix_estimate"] = -R * np.sum(
    design_space[elements] * np.log(design_space[elements] + EPS),
    axis=1,
)

valid_idx = (
    (design_space["active_elements"] >= 4)
    & (design_space["max_fraction"] <= 0.40)
    & (design_space["Smix_estimate"] >= 10.0)
    & (design_space["light_fraction"] <= 0.50)
)

if len(refractory_elements) > 0:
    valid_idx &= (
        (design_space["refractory_fraction"] >= 0.15)
        & (design_space["refractory_fraction"] <= 0.85)
    )

if len(distortion_elements) > 0:
    valid_idx &= design_space["distortion_fraction"] >= 0.08

design_space = design_space.loc[
    valid_idx
].reset_index(drop=True)

design_space["signature"] = (
    (design_space[elements] * 20)
    .round()
    .astype(int)
    .astype(str)
    .agg("-".join, axis=1)
)

design_space = (
    design_space
    .drop_duplicates("signature")
    .drop(columns="signature")
    .reset_index(drop=True)
)

design_space["Global_Alloy_ID"] = [
    f"GALLOY_{index + 1:07d}"
    for index in range(len(design_space))
]

ordered_cols = (
    ["Global_Alloy_ID"]
    + elements
    + [
        "sampling_mode",
        "active_elements",
        "refractory_fraction",
        "distortion_fraction",
        "light_fraction",
        "max_fraction",
        "Smix_estimate",
    ]
)

design_space = design_space[ordered_cols]

design_space.to_csv(
    out_dir / "global_candidate_design_space.csv",
    index=False,
)

element_summary = pd.DataFrame({
    "Element": elements,
    "Dataset_Occurrence": [
        element_occurrence[element]
        for element in elements
    ],
})

element_summary.to_csv(
    out_dir / "element_design_space_summary.csv",
    index=False,
)

print(f"Initial generated alloys: {N_GLOBAL_SAMPLES}")
print(f"Final valid alloys: {len(design_space)}")
print(f"Design elements: {len(elements)}")
print(f"Elements explored: {elements}")
print(f"Output folder: {out_dir}")
