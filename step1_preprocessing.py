import pandas as pd
import numpy as np
import os
import re
import hashlib

# ------------------------------------------------------------
# Step 1: Load raw experimental dataset and lock an immutable copy
# ------------------------------------------------------------

input_file = "/content/MPEA_dataset.csv"
output_dir = "/content/output"
os.makedirs(output_dir, exist_ok=True)

df_raw = pd.read_csv(input_file)
print("Dataset shape:", df_raw.shape)

# ------------------------------------------------------------
# Dataset fingerprint for reproducibility
# ------------------------------------------------------------

dataset_hash = hashlib.sha256(
    pd.util.hash_pandas_object(df_raw, index=True).values
).hexdigest()

print("Dataset SHA-256 fingerprint:", dataset_hash)
print("Total rows:", len(df_raw))
print("Total columns:", len(df_raw.columns))

# ------------------------------------------------------------
# Column inspection
# ------------------------------------------------------------

print("Available columns:")
for col in df_raw.columns:
    print(" -", col)

# Save immutable raw copy
locked_path = os.path.join(output_dir, "HEA_raw_experimental_dataset.csv")
df_raw.to_csv(locked_path, index=False)
print("Raw dataset copy saved at:", locked_path)

# ------------------------------------------------------------
# Missing value audit
# ------------------------------------------------------------

missing_counts = df_raw.isnull().sum()
missing_percent = (missing_counts / len(df_raw)) * 100

missing_summary = pd.DataFrame({
    "Missing_Count": missing_counts,
    "Missing_Percentage (%)": missing_percent
}).sort_values(by="Missing_Count", ascending=False)

print("Missing value audit (top 15 columns):")
print(missing_summary.head(15))

# ------------------------------------------------------------
# Identify key information columns
# ------------------------------------------------------------

composition_cols = [
    c for c in df_raw.columns
    if "formula" in c.lower() or "composition" in c.lower()
]

phase_cols = [c for c in df_raw.columns if "phase" in c.lower()]

strength_cols = [
    c for c in df_raw.columns
    if ("strength" in c.lower()) or ("yield" in c.lower())
]

# Detect hardness column dynamically
hardness_cols = [
    c for c in df_raw.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]

print("Composition columns:", composition_cols)
print("Phase columns:", phase_cols)
print("Strength columns:", strength_cols)
print("Hardness columns:", hardness_cols)

# ------------------------------------------------------------
# Dataset coverage summary
# ------------------------------------------------------------

coverage_summary = {
    "Total_alloys": len(df_raw),
    "Alloys_with_composition":
        df_raw[composition_cols].notnull().any(axis=1).sum()
        if composition_cols else 0,
    "Alloys_with_phase_info":
        df_raw[phase_cols].notnull().any(axis=1).sum()
        if phase_cols else 0,
    "Alloys_with_strength":
        df_raw[strength_cols].notnull().any(axis=1).sum()
        if strength_cols else 0,
    "Alloys_with_hardness":
        df_raw[hardness_cols].notnull().any(axis=1).sum()
        if hardness_cols else 0
}

coverage_df = pd.DataFrame.from_dict(
    coverage_summary,
    orient="index",
    columns=["Count"]
)

print("Dataset coverage summary:")
print(coverage_df)

# ------------------------------------------------------------
# Composition parsing (explicit stoichiometric formulas only)
# ------------------------------------------------------------

def parse_formula_safe(formula):
    if not isinstance(formula, str):
        return None

    # Exclude complex formulas (parentheses often imply phase grouping or uncertainty)
    if "(" in formula or ")" in formula:
        return None

    tokens = re.findall(
        r"([A-Z][a-z]*)([0-9]*\.?[0-9]*)", formula
    )

    if len(tokens) == 0:
        return None

    comp = {}
    for el, amt in tokens:
        comp[el] = float(amt) if amt else 1.0

    total = sum(comp.values())
    if total == 0:
        return None

    normalized = {el: (v / total) * 100.0 for el, v in comp.items()}

    # Precision sanity check (helps detect malformed formulas)
    if abs(sum(normalized.values()) - 100.0) > 1e-6:
        return None

    return normalized

# ------------------------------------------------------------
# Select composition column
# ------------------------------------------------------------

if len(composition_cols) == 0:
    raise ValueError("No composition/formula column detected in dataset")

# Use primary composition column (first detected).
# This avoids mixing representations from multiple fields.
formula_col = composition_cols[0]
print("Using composition column:", formula_col)

df_raw["PARSED_COMPOSITION"] = df_raw[formula_col].apply(parse_formula_safe)

df_raw["COMPOSITION_VALID"] = df_raw["PARSED_COMPOSITION"].notnull()

df_raw["COMPOSITION_PARSE_REASON"] = np.where(
    df_raw[formula_col].astype(str).str.contains(r"[()]"),
    "Excluded: complex formula",
    np.where(
        df_raw["PARSED_COMPOSITION"].isnull(),
        "Excluded: unparseable formula",
        "Parsed successfully"
    )
)

print(
    "Valid composition rows:",
    df_raw["COMPOSITION_VALID"].sum(),
    "/",
    len(df_raw)
)

# ------------------------------------------------------------
# Elemental space construction
# ------------------------------------------------------------

all_elements = sorted({
    el
    for comp in df_raw["PARSED_COMPOSITION"].dropna()
    for el in comp.keys()
})

print("Total unique elements discovered:", len(all_elements))
print(all_elements)

# Explicitly encode elemental absence as 0.0
for el in all_elements:
    df_raw[f"ELEM_{el}"] = 0.0

for idx, comp in df_raw["PARSED_COMPOSITION"].items():
    if isinstance(comp, dict):
        for el, val in comp.items():
            df_raw.at[idx, f"ELEM_{el}"] = val

# ------------------------------------------------------------
# Property availability flags
# ------------------------------------------------------------

if hardness_cols:
    df_raw["HARDNESS_AVAILABLE"] = (
        df_raw[hardness_cols].notnull().any(axis=1)
    )
else:
    df_raw["HARDNESS_AVAILABLE"] = False

if strength_cols:
    df_raw["STRENGTH_AVAILABLE"] = (
        df_raw[strength_cols].notnull().any(axis=1)
    )
else:
    df_raw["STRENGTH_AVAILABLE"] = False

if phase_cols:
    df_raw["PHASE_AVAILABLE"] = (
        df_raw[phase_cols].notnull().any(axis=1)
    )
else:
    df_raw["PHASE_AVAILABLE"] = False

print("Hardness-available rows:", df_raw["HARDNESS_AVAILABLE"].sum())
print("Strength-available rows:", df_raw["STRENGTH_AVAILABLE"].sum())
print("Phase-available rows:", df_raw["PHASE_AVAILABLE"].sum())

# ------------------------------------------------------------
# Lock STEP-1 dataset
# ------------------------------------------------------------

final_step1_path = os.path.join(
    output_dir,
    "HEA_raw_experimental_dataset_STEP1_LOCKED.csv"
)

df_raw.to_csv(final_step1_path, index=False)
print("STEP-1 dataset written to:", final_step1_path)
