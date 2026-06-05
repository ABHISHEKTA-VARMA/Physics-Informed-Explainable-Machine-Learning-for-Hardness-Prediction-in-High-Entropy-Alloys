import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


RANDOM_STATE = 42
DATASET_VERSION = "mpea_dataset_v1"
GPA_TO_HV = 102.0
R = 8.314

BORG_PATH = Path("MPEA_dataset.csv")
GORSSE_PATH = Path("Gorsse_HV_FINAL.csv")
GE_PATH = Path("GE_RefractoryAlloyScreeningDataset_FINAL.csv")

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

MASTER_OUT = OUT_DIR / "MASTER_MPEA_DATASET.csv"
META_OUT = OUT_DIR / "MASTER_METADATA.txt"
ELEMENT_FREQ_OUT = OUT_DIR / "ELEMENT_FREQUENCY.csv"
SOURCE_SUMMARY_OUT = OUT_DIR / "SOURCE_SUMMARY.csv"
SAMPLE_ID_OUT = OUT_DIR / "SAMPLE_IDS.csv"
SPLIT_OUT = OUT_DIR / "DATA_SPLITS.csv"

SUPPORTED_ELEMENTS = {
    "Al", "Co", "Cr", "Fe", "Ni",
    "Mn", "Ti", "V", "Cu", "Zn",
    "Nb", "Mo", "Ta", "W", "Hf",
    "Zr", "Ru", "Rh", "Pd", "Pt",
    "Si", "B", "C", "Y", "Sc",
    "Ga", "Ge", "Sn",
}

REFRACTORY_ELEMENTS = {"Nb", "Mo", "Ta", "W", "Hf", "Zr", "V"}


def parse_formula(formula):
    if pd.isna(formula):
        return {}

    formula = str(formula).strip().replace(" ", "")
    pattern = r"([A-Z][a-z]*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?"
    parts = re.findall(pattern, formula)

    composition = {}

    for element, amount in parts:
        if element not in SUPPORTED_ELEMENTS:
            continue

        try:
            amount = 1.0 if amount in ["", None] else float(amount)
        except ValueError:
            amount = 1.0

        if amount >= 0:
            composition[f"ELEM_{element}"] = amount

    total = sum(composition.values())

    if total <= 0:
        return {}

    return {key: value / total for key, value in composition.items()}


def normalize_processing(value):
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper()

    if any(key in text for key in ["AS-CAST", "CAST"]):
        return "AS_CAST"
    if any(key in text for key in ["ANNEAL", "HEAT", "SOLUTION", "HOMOGEN"]):
        return "HEAT_TREATED"
    if any(key in text for key in ["ROLL", "FORGE", "WROUGHT"]):
        return "DEFORMED"
    if any(key in text for key in ["SPS", "POWDER", "PM"]):
        return "POWDER_MET"
    if any(key in text for key in ["ARC", "VACUUM"]):
        return "ARC_MELTED"

    return "OTHER"


def normalize_phase(value):
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper()

    if "FCC+BCC" in text or ("FCC" in text and "BCC" in text):
        return "FCC_BCC"
    if "FCC" in text:
        return "FCC"
    if "BCC" in text:
        return "BCC"
    if "HCP" in text:
        return "HCP"
    if any(key in text for key in ["AMORPH", "GLASS"]):
        return "AMORPHOUS"
    if any(key in text for key in ["INTERMETALLIC", "LAVES"]):
        return "INTERMETALLIC"

    return "OTHER"


def load_borg_dataset(path):
    raw = pd.read_csv(path)
    data = raw.dropna(subset=["PROPERTY: HV"]).copy()

    elements = pd.DataFrame(
        data["FORMULA"].apply(parse_formula).tolist()
    ).fillna(0)

    elements["PROPERTY: HV"] = data["PROPERTY: HV"].values
    elements["FORMULA"] = data["FORMULA"].values
    elements["SOURCE_ROW_ID"] = "BORG_" + data.index.astype(str)

    for column in ["MICROSTRUCTURE", "PROCESSING", "PHASE", "TEST_TEMP"]:
        if column in data.columns:
            elements[column] = data[column].values
        else:
            elements[column] = np.nan

    elements["PHASE_CLASS"] = elements["PHASE"].apply(normalize_phase)
    elements["PROCESS_CLASS"] = elements["PROCESSING"].apply(normalize_processing)
    elements["SOURCE"] = "BORG"

    return elements


def load_gorsse_dataset(path):
    raw = pd.read_csv(path)

    elements = pd.DataFrame(
        raw["Composition"].apply(parse_formula).tolist()
    ).fillna(0)

    elements["PROPERTY: HV"] = raw["HV"].values
    elements["FORMULA"] = raw["Composition"].values
    elements["SOURCE_ROW_ID"] = "GORSSE_" + raw.index.astype(str)
    elements["PHASE_CLASS"] = "UNKNOWN"
    elements["PROCESS_CLASS"] = "UNKNOWN"
    elements["SOURCE"] = "GORSSE"

    return elements


def load_ge_dataset(path):
    raw = pd.read_csv(path)

    element_map = {
        "Hf(at%)": "ELEM_Hf",
        "Mo(at%)": "ELEM_Mo",
        "Nb(at%)": "ELEM_Nb",
        "Ru(at%)": "ELEM_Ru",
        "Ta(at%)": "ELEM_Ta",
        "Ti(at%)": "ELEM_Ti",
        "W(at%)": "ELEM_W",
        "Zr(at%)": "ELEM_Zr",
    }

    elements = pd.DataFrame()

    for old_column, new_column in element_map.items():
        if old_column in raw.columns:
            elements[new_column] = raw[old_column]

    elements = elements.fillna(0)

    row_sum = elements.sum(axis=1)
    row_sum[row_sum == 0] = 1

    elements = elements.div(row_sum, axis=0)

    if "Hardness (GPa)" in raw.columns:
        elements["PROPERTY: HV"] = raw["Hardness (GPa)"] * GPA_TO_HV
    else:
        elements["PROPERTY: HV"] = np.nan

    elements["FORMULA"] = "GE_REFRACTORY"
    elements["SOURCE_ROW_ID"] = "GE_" + raw.index.astype(str)
    elements["PHASE_CLASS"] = "REFRACTORY"
    elements["PROCESS_CLASS"] = "SCREENED"
    elements["SOURCE"] = "GE"

    return elements


def mode_or_unknown(values):
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else "UNKNOWN"


np.random.seed(RANDOM_STATE)

borg_elem = load_borg_dataset(BORG_PATH)
gorsse_elem = load_gorsse_dataset(GORSSE_PATH)
ge_elem = load_ge_dataset(GE_PATH)

master = pd.concat(
    [borg_elem, gorsse_elem, ge_elem],
    ignore_index=True,
).fillna(0)

master["DATASET_VERSION"] = DATASET_VERSION

elem_cols = [
    column for column in master.columns
    if column.startswith("ELEM_")
]

row_sum = master[elem_cols].sum(axis=1)
row_sum[row_sum == 0] = 1

master[elem_cols] = master[elem_cols].div(row_sum, axis=0)

master["CONFIG_ENTROPY"] = -R * (
    master[elem_cols]
    .replace(0, np.nan)
    .apply(lambda row: np.nansum(row * np.log(row)), axis=1)
)

master["NUM_ELEMENTS"] = (master[elem_cols] > 0.01).sum(axis=1)

master["MAX_ELEMENT"] = master[elem_cols].max(axis=1)

master = master.loc[
    (master["NUM_ELEMENTS"] >= 3)
    & (master["MAX_ELEMENT"] <= 0.80)
]

refractory_cols = [
    column for column in elem_cols
    if column.replace("ELEM_", "") in REFRACTORY_ELEMENTS
]

master["REFRACTORY_FRACTION"] = master[refractory_cols].sum(axis=1)

master = master.loc[
    (master["PROPERTY: HV"] >= 50)
    & (master["PROPERTY: HV"] <= 1200)
]

if "TEST_TEMP" in master.columns:
    master["TEST_TEMP"] = pd.to_numeric(
        master["TEST_TEMP"],
        errors="coerce",
    )

master["LOW_TEMP_MECHANICAL_REGIME"] = (
    master["TEST_TEMP"].isna()
    | (master["TEST_TEMP"] <= 200)
)

master = master.loc[
    master["LOW_TEMP_MECHANICAL_REGIME"]
]

comp_error = np.abs(
    master[elem_cols].sum(axis=1) - 1
)

master = master.loc[
    comp_error < 1e-4
]

pre_dup_count = len(master)

master["COMPOSITION_SIGNATURE"] = (
    master[elem_cols]
    .round(6)
    .astype(str)
    .agg("-".join, axis=1)
)

hv_stats = (
    master.groupby("COMPOSITION_SIGNATURE")["PROPERTY: HV"]
    .agg(["std", "count"])
    .reset_index()
)

hv_stats.columns = [
    "COMPOSITION_SIGNATURE",
    "PROPERTY_HV_STD",
    "PROPERTY_HV_COUNT",
]

aggregation_dict = {
    **{column: "mean" for column in elem_cols},
    "PROPERTY: HV": "median",
    "FORMULA": "first",
    "PHASE_CLASS": mode_or_unknown,
    "PROCESS_CLASS": mode_or_unknown,
    "SOURCE": lambda values: "|".join(sorted(set(values.astype(str)))),
    "SOURCE_ROW_ID": lambda values: "|".join(sorted(set(values.astype(str)))),
    "REFRACTORY_FRACTION": "mean",
    "NUM_ELEMENTS": "mean",
    "MAX_ELEMENT": "mean",
    "CONFIG_ENTROPY": "mean",
    "LOW_TEMP_MECHANICAL_REGIME": "max",
    "DATASET_VERSION": "first",
}

if "TEST_TEMP" in master.columns:
    aggregation_dict["TEST_TEMP"] = "median"

master = (
    master.groupby("COMPOSITION_SIGNATURE", as_index=False)
    .agg(aggregation_dict)
)

master = master.merge(
    hv_stats,
    on="COMPOSITION_SIGNATURE",
    how="left",
)

master["PROPERTY_HV_STD"] = master["PROPERTY_HV_STD"].fillna(0)

post_dup_count = len(master)

duplicates_removed = pre_dup_count - post_dup_count

master["HV_CLASS"] = pd.qcut(
    master["PROPERTY: HV"],
    q=4,
    labels=["LOW", "MEDIUM", "HIGH", "ULTRA_HIGH"],
)

master = master.reset_index(drop=True)

master["SAMPLE_ID"] = [
    f"MPEA_{index:05d}"
    for index in range(len(master))
]

master["ALLOY_HASH"] = [
    hashlib.sha256(signature.encode()).hexdigest()
    for signature in master["COMPOSITION_SIGNATURE"]
]

groups = master["COMPOSITION_SIGNATURE"]

split_col = np.array(
    ["TRAIN"] * len(master)
)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    gss.split(
        master,
        groups=groups,
    )
)

split_col[test_idx] = "TEST"

master["DATA_SPLIT"] = split_col

source_summary = pd.DataFrame({
    "Count": master["SOURCE"].value_counts()
})

source_summary.to_csv(
    SOURCE_SUMMARY_OUT
)

element_frequency = pd.DataFrame({
    "Element": elem_cols,
    "Frequency": [
        (master[column] > 0).sum()
        for column in elem_cols
    ],
})

element_frequency = element_frequency.sort_values(
    "Frequency",
    ascending=False,
)

element_frequency.to_csv(
    ELEMENT_FREQ_OUT,
    index=False,
)

master[[
    "SAMPLE_ID",
    "FORMULA",
    "PROPERTY: HV",
]].to_csv(
    SAMPLE_ID_OUT,
    index=False,
)

master[[
    "SAMPLE_ID",
    "COMPOSITION_SIGNATURE",
    "DATA_SPLIT",
]].to_csv(
    SPLIT_OUT,
    index=False,
)

tmp_path = OUT_DIR / "_tmp_dataset.csv"

master.to_csv(
    tmp_path,
    index=False,
)

with open(tmp_path, "rb") as file:
    sha = hashlib.sha256(file.read()).hexdigest()

tmp_path.unlink()

master.to_csv(
    MASTER_OUT,
    index=False,
)

metadata_lines = [
    "MPEA dataset curation report",
    "",
    f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
    f"Rows: {len(master)}",
    f"Columns: {len(master.columns)}",
    f"Elements: {len(elem_cols)}",
    f"SHA256: {sha}",
    f"Dataset version: {DATASET_VERSION}",
    f"GPa to HV factor: {GPA_TO_HV}",
    f"Duplicates removed: {duplicates_removed}",
    f"Supported chemistry elements: {len(SUPPORTED_ELEMENTS)}",
    "",
    "Source distribution",
    master["SOURCE"].value_counts().to_string(),
    "",
    "Phase distribution",
    master["PHASE_CLASS"].value_counts().to_string(),
    "",
    "Processing distribution",
    master["PROCESS_CLASS"].value_counts().to_string(),
    "",
    "Hardness range",
    f"{master['PROPERTY: HV'].min():.2f} to {master['PROPERTY: HV'].max():.2f} HV",
    "",
]

META_OUT.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

print(f"Dataset size: {master.shape}")
print(f"Detected elements: {len(elem_cols)}")
print(f"Train samples: {(master['DATA_SPLIT'] == 'TRAIN').sum()}")
print(f"Test samples: {(master['DATA_SPLIT'] == 'TEST').sum()}")
print(f"Saved to: {OUT_DIR}")
