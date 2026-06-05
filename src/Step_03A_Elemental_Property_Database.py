import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DB_VERSION = "step3a_property_db_v1"
REFERENCE_STATE = "Room-temperature equilibrium bulk elemental properties"
PROPERTY_ASSUMPTION = (
    "Scalar elemental properties are used as transferable composition-based "
    "descriptor inputs. They do not explicitly model temperature-dependent "
    "phase transitions or polymorphic state evolution."
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "UNIVERSAL_PROPERTY_DB.csv"
property_group_out = out_dir / "PROPERTY_GROUPS.txt"
units_out = out_dir / "PROPERTY_UNITS.json"
taxonomy_out = out_dir / "PROPERTY_TAXONOMY.json"
metadata_out = out_dir / "PROPERTY_DB_METADATA.txt"
descriptor_family_out = out_dir / "DESCRIPTOR_FAMILY_TABLE.csv"
property_source_out = out_dir / "PROPERTY_SOURCE_REFERENCE.json"

PROPERTY_UNITS = {
    "r": "Angstrom",
    "atomic_volume": "cm3/mol",
    "rho": "g/cm3",
    "Tm": "K",
    "E": "GPa",
    "G": "GPa",
    "K": "GPa",
    "thermal_cond": "W/mK",
    "cohesive_energy": "eV/atom",
    "work_function": "eV",
    "chi": "Pauling",
    "diffusivity_proxy": "relative",
    "cost_index": "relative",
    "poisson_ratio": "dimensionless",
    "Pugh_ratio": "dimensionless",
    "anisotropy_proxy": "dimensionless",
    "elastic_anisotropy_proxy": "dimensionless",
    "reduced_modulus_proxy": "GPa",
    "specific_stiffness": "GPa.cm3/g",
    "stiffness_density_index": "GPa.cm3/g",
    "Tm_over_rho": "K.cm3/g",
    "thermal_cond_specific": "W.cm3/gK",
    "modulus_to_melting_ratio": "GPa/K",
    "segregation_tendency": "relative",
    "oxidation_sensitivity": "relative",
    "oxide_affinity": "relative",
    "manufacturing_difficulty": "relative",
    "feasibility_weight": "dimensionless",
}

PROPERTY_TAXONOMY = {
    "canonical": [
        "r", "atomic_volume", "rho", "Tm", "E", "G", "K", "VEC",
        "thermal_cond", "cohesive_energy", "work_function", "chi",
    ],
    "derived_physical": [
        "poisson_ratio", "Pugh_ratio", "anisotropy_proxy",
        "elastic_anisotropy_proxy", "reduced_modulus_proxy",
        "specific_stiffness", "stiffness_density_index",
        "Tm_over_rho", "thermal_cond_specific",
        "modulus_to_melting_ratio",
    ],
    "heuristic": [
        "segregation_tendency", "oxidation_sensitivity", "oxide_affinity",
        "manufacturing_difficulty", "feasibility_weight",
    ],
    "categorical": ["crystal", "role"],
}

PROPERTY_SOURCE_REFERENCE = {
    "canonical": "ASM Handbook, CRC Handbook, and standard metallurgy references",
    "cohesive_energy": "Materials thermodynamics literature",
    "work_function": "Surface science literature",
    "chi": "Pauling electronegativity scale",
    "derived_physical": "Relations derived from canonical elemental properties",
    "heuristic": "Metallurgy-informed engineering priors",
}

DESCRIPTOR_FAMILY_MAP = {
    "r": "atomic_size",
    "atomic_volume": "atomic_size",
    "rho": "density",
    "Tm": "thermodynamic",
    "E": "elastic",
    "G": "elastic",
    "K": "elastic",
    "VEC": "electronic",
    "thermal_cond": "transport",
    "cohesive_energy": "bonding",
    "work_function": "electronic",
    "chi": "electronic",
    "poisson_ratio": "elastic_ratio",
    "Pugh_ratio": "elastic_ratio",
    "anisotropy_proxy": "anisotropy",
    "elastic_anisotropy_proxy": "anisotropy",
    "reduced_modulus_proxy": "elastic",
    "specific_stiffness": "density_normalized",
    "stiffness_density_index": "density_normalized",
    "Tm_over_rho": "density_normalized",
    "thermal_cond_specific": "transport",
    "modulus_to_melting_ratio": "thermoelastic",
    "segregation_tendency": "segregation",
    "oxidation_sensitivity": "oxidation",
    "oxide_affinity": "oxidation",
    "manufacturing_difficulty": "manufacturability",
    "feasibility_weight": "manufacturability",
}

data = [
    ["Li", 1.52, 13.1, 0.53, 454, 4.9, 4.2, 11, 1, 85, 1.63, 2.9, 0.98, 0.36, 1.0, 1, 2, "BCC", "lightweight"],
    ["Be", 1.12, 4.9, 1.85, 1560, 287, 132, 130, 2, 200, 3.32, 5.0, 1.57, 0.03, 2.0, 2, 2, "HCP", "lightweight"],
    ["Mg", 1.60, 14.0, 1.74, 923, 45, 17, 35, 2, 160, 1.51, 3.7, 1.31, 0.22, 1.0, 2, 3, "HCP", "lightweight"],
    ["Al", 1.43, 10.0, 2.70, 933, 70, 26, 76, 3, 237, 3.39, 4.2, 1.61, 0.12, 1.0, 13, 3, "FCC", "lightweight"],
    ["Ti", 1.47, 10.6, 4.50, 1941, 116, 44, 110, 4, 22, 4.85, 4.3, 1.54, 0.15, 2.0, 4, 4, "HCP", "transition"],
    ["V", 1.34, 8.4, 6.11, 2183, 128, 47, 160, 5, 31, 5.31, 4.3, 1.63, 0.18, 2.0, 5, 4, "BCC", "bcc_stabilizer"],
    ["Cr", 1.28, 7.2, 7.19, 2180, 279, 115, 160, 6, 94, 4.10, 4.5, 1.66, 0.21, 2.0, 6, 4, "BCC", "bcc_stabilizer"],
    ["Mn", 1.27, 7.4, 7.21, 1519, 198, 76, 120, 7, 7.8, 2.92, 4.1, 1.55, 0.26, 2.0, 7, 4, "COMPLEX", "transition"],
    ["Fe", 1.26, 7.1, 7.87, 1811, 211, 82, 170, 8, 80, 4.28, 4.5, 1.83, 0.17, 2.0, 8, 4, "BCC", "transition"],
    ["Co", 1.25, 6.7, 8.90, 1768, 211, 75, 180, 9, 100, 4.39, 5.0, 1.88, 0.16, 2.0, 9, 4, "HCP", "fcc_stabilizer"],
    ["Ni", 1.24, 6.6, 8.90, 1728, 200, 76, 180, 10, 91, 4.44, 5.2, 1.91, 0.14, 2.0, 10, 4, "FCC", "fcc_stabilizer"],
    ["Cu", 1.28, 7.1, 8.96, 1358, 130, 48, 140, 11, 401, 3.49, 4.7, 1.90, 0.13, 2.0, 11, 4, "FCC", "fcc_stabilizer"],
    ["Zn", 1.39, 9.2, 7.14, 693, 108, 43, 70, 12, 116, 1.35, 4.3, 1.65, 0.30, 1.0, 12, 4, "HCP", "transition"],
    ["Zr", 1.60, 14.0, 6.52, 2128, 88, 33, 92, 4, 23, 6.25, 4.1, 1.33, 0.18, 3.0, 4, 5, "HCP", "refractory"],
    ["Nb", 1.46, 10.8, 8.57, 2750, 105, 38, 170, 5, 54, 7.57, 4.3, 1.60, 0.17, 3.0, 5, 5, "BCC", "refractory"],
    ["Mo", 1.39, 9.4, 10.28, 2896, 329, 126, 230, 6, 138, 6.82, 4.6, 2.16, 0.16, 3.0, 6, 5, "BCC", "refractory"],
    ["Hf", 1.59, 13.6, 13.31, 2506, 78, 30, 110, 4, 23, 6.44, 3.9, 1.30, 0.20, 4.0, 4, 6, "HCP", "refractory"],
    ["Ta", 1.46, 10.9, 16.69, 3290, 186, 69, 200, 5, 57, 8.10, 4.2, 1.50, 0.14, 4.0, 5, 6, "BCC", "refractory"],
    ["W", 1.39, 9.5, 19.25, 3695, 411, 161, 310, 6, 173, 8.90, 4.5, 2.36, 0.12, 4.0, 6, 6, "BCC", "refractory"],
    ["Re", 1.37, 8.9, 21.02, 3459, 463, 178, 370, 7, 48, 8.03, 5.1, 1.90, 0.11, 5.0, 7, 6, "HCP", "refractory"],
    ["Ru", 1.34, 8.3, 12.37, 2607, 447, 173, 320, 8, 117, 6.74, 4.7, 2.20, 0.15, 5.0, 8, 5, "HCP", "transition"],
    ["Rh", 1.34, 8.3, 12.41, 2237, 380, 150, 270, 9, 150, 5.75, 4.9, 2.28, 0.14, 5.0, 9, 5, "FCC", "fcc_stabilizer"],
    ["Pd", 1.37, 8.9, 12.02, 1828, 121, 44, 180, 10, 72, 3.89, 5.1, 2.20, 0.13, 5.0, 10, 5, "FCC", "fcc_stabilizer"],
    ["Ag", 1.44, 10.3, 10.49, 1235, 83, 30, 100, 11, 429, 2.95, 4.3, 1.93, 0.12, 4.0, 11, 5, "FCC", "transition"],
    ["Ir", 1.36, 8.5, 22.56, 2739, 528, 210, 355, 9, 147, 8.71, 5.7, 2.20, 0.10, 5.0, 9, 6, "FCC", "transition"],
    ["Pt", 1.39, 9.1, 21.45, 2041, 168, 61, 230, 10, 72, 5.84, 5.6, 2.28, 0.11, 5.0, 10, 6, "FCC", "fcc_stabilizer"],
    ["Au", 1.44, 10.2, 19.30, 1337, 79, 27, 180, 11, 318, 3.81, 5.3, 2.54, 0.10, 4.0, 11, 6, "FCC", "transition"],
    ["Sc", 1.64, 15.0, 2.99, 1814, 74, 29, 57, 3, 16, 3.90, 3.5, 1.36, 0.20, 2.0, 3, 4, "HCP", "rare_earth"],
    ["Y", 1.80, 19.9, 4.47, 1799, 64, 26, 41, 3, 17, 4.37, 3.1, 1.22, 0.24, 2.0, 3, 5, "HCP", "rare_earth"],
    ["La", 1.87, 22.5, 6.15, 1193, 37, 14, 28, 3, 13, 4.47, 3.5, 1.10, 0.31, 1.0, 3, 6, "HCP", "rare_earth"],
    ["Ce", 1.82, 20.7, 6.77, 1071, 34, 13, 22, 4, 11, 4.16, 2.9, 1.12, 0.34, 1.0, 4, 6, "FCC", "rare_earth"],
    ["Nd", 1.82, 20.6, 7.01, 1297, 41, 16, 32, 3, 17, 4.00, 3.2, 1.14, 0.30, 1.0, 3, 6, "HCP", "rare_earth"],
    ["Si", 1.11, 12.1, 2.33, 1687, 130, 51, 98, 4, 149, 4.63, 4.8, 1.90, 0.28, 1.0, 14, 3, "DIAMOND", "metalloid"],
    ["Ge", 1.22, 13.6, 5.32, 1211, 103, 41, 77, 4, 60, 3.85, 4.7, 2.01, 0.26, 1.0, 14, 4, "DIAMOND", "metalloid"],
]

columns = [
    "Element", "r", "atomic_volume", "rho", "Tm", "E", "G", "K",
    "VEC", "thermal_cond", "cohesive_energy", "work_function", "chi",
    "diffusivity_proxy", "cost_index", "group", "period", "crystal", "role",
]

db = pd.DataFrame(data, columns=columns)

numeric_cols = [
    column for column in db.columns
    if column not in ["Element", "crystal", "role"]
]

db[numeric_cols] = db[numeric_cols].apply(pd.to_numeric, errors="coerce")

db["property_reference_state"] = REFERENCE_STATE
db["property_assumption"] = PROPERTY_ASSUMPTION

db["poisson_ratio"] = (
    (3 * db["K"] - 2 * db["G"])
    / (2 * (3 * db["K"] + db["G"]) + 1e-9)
)

db["Pugh_ratio"] = db["G"] / (db["K"] + 1e-9)
db["anisotropy_proxy"] = db["E"] / (3 * db["G"] + 1e-9)
db["elastic_anisotropy_proxy"] = 2 * db["G"] / (3 * db["K"] + 1e-9)

db["reduced_modulus_proxy"] = (
    db["E"] / (1 - db["poisson_ratio"] ** 2 + 1e-9)
)

db["specific_stiffness"] = db["E"] / (db["rho"] + 1e-9)
db["stiffness_density_index"] = db["G"] / (db["rho"] + 1e-9)
db["Tm_over_rho"] = db["Tm"] / (db["rho"] + 1e-9)
db["thermal_cond_specific"] = db["thermal_cond"] / (db["rho"] + 1e-9)
db["modulus_to_melting_ratio"] = db["E"] / (db["Tm"] + 1e-9)

db["segregation_tendency"] = db["diffusivity_proxy"] * db["rho"]
db["oxidation_sensitivity"] = 1 / (db["cohesive_energy"] + 1e-9)

oxide_affinity_map = {
    "Al": 5.0,
    "Ti": 4.8,
    "Zr": 4.8,
    "Hf": 4.7,
    "Y": 5.0,
    "La": 5.0,
    "Ce": 5.0,
    "Nd": 4.9,
    "Mg": 4.8,
}

db["oxide_affinity"] = db["Element"].map(oxide_affinity_map).fillna(2.5)

db["manufacturing_difficulty"] = (
    db["cost_index"] * db["rho"] * db["Tm"] / 1000
)

db["d_block"] = ((db["group"] >= 3) & (db["group"] <= 12)).astype(int)
db["f_block"] = (db["role"] == "rare_earth").astype(int)

db["fcc_tendency"] = (db["VEC"] >= 8).astype(int)
db["bcc_tendency"] = (db["VEC"] <= 6).astype(int)

db["polymorphic_element"] = db["Element"].isin(
    ["Fe", "Ti", "Co", "Zr", "Hf"]
).astype(int)

db["is_refractory"] = (db["role"] == "refractory").astype(int)
db["is_fcc_stabilizer"] = (db["role"] == "fcc_stabilizer").astype(int)
db["is_bcc_stabilizer"] = (db["role"] == "bcc_stabilizer").astype(int)
db["is_lightweight"] = (db["role"] == "lightweight").astype(int)
db["is_rare_earth"] = (db["role"] == "rare_earth").astype(int)
db["is_metalloid"] = (db["role"] == "metalloid").astype(int)

cost_n = db["cost_index"] / db["cost_index"].max()
mfg_n = db["manufacturing_difficulty"] / db["manufacturing_difficulty"].max()
seg_n = db["segregation_tendency"] / db["segregation_tendency"].max()

db["feasibility_weight"] = (
    1 / (1 + 0.35 * cost_n + 0.35 * mfg_n + 0.30 * seg_n)
)

db["DB_VERSION"] = DB_VERSION

if db.isnull().values.any():
    raise ValueError("NaN detected in property database.")

numeric_check = db.select_dtypes(include=np.number)

if np.isinf(numeric_check.values).any():
    raise ValueError("Infinite value detected in property database.")

if db.duplicated("Element").any():
    raise ValueError("Duplicate element detected in property database.")

if not ((db["poisson_ratio"] > -1).all() and (db["poisson_ratio"] < 0.5).all()):
    raise ValueError("Invalid Poisson ratio detected.")

if (db["E"] <= 0).any():
    raise ValueError("Invalid Young modulus detected.")

if (db["G"] <= 0).any():
    raise ValueError("Invalid shear modulus detected.")

if (db["K"] <= 0).any():
    raise ValueError("Invalid bulk modulus detected.")

db.to_csv(out_csv, index=False)

with open(units_out, "w", encoding="utf-8") as file:
    json.dump(PROPERTY_UNITS, file, indent=4)

with open(taxonomy_out, "w", encoding="utf-8") as file:
    json.dump(PROPERTY_TAXONOMY, file, indent=4)

with open(property_source_out, "w", encoding="utf-8") as file:
    json.dump(PROPERTY_SOURCE_REFERENCE, file, indent=4)

descriptor_family_df = pd.DataFrame({
    "Feature": list(DESCRIPTOR_FAMILY_MAP.keys()),
    "Descriptor_Family": list(DESCRIPTOR_FAMILY_MAP.values()),
})

descriptor_family_df.to_csv(descriptor_family_out, index=False)

property_groups = {
    "Thermodynamic": [
        "Tm", "Tm_over_rho", "modulus_to_melting_ratio",
    ],
    "Elastic": [
        "E", "G", "K", "poisson_ratio", "Pugh_ratio",
        "anisotropy_proxy", "elastic_anisotropy_proxy",
        "reduced_modulus_proxy", "specific_stiffness",
        "stiffness_density_index",
    ],
    "Electronic": [
        "VEC", "work_function", "chi",
    ],
    "Bonding": [
        "cohesive_energy",
    ],
    "Transport": [
        "thermal_cond", "thermal_cond_specific",
    ],
    "Manufacturability": [
        "manufacturing_difficulty", "cost_index", "feasibility_weight",
    ],
    "Oxidation": [
        "oxide_affinity", "oxidation_sensitivity",
    ],
    "Segregation": [
        "segregation_tendency", "diffusivity_proxy",
    ],
    "Periodic": [
        "group", "period", "d_block", "f_block",
    ],
}

with open(property_group_out, "w", encoding="utf-8") as file:
    for group_name, features in property_groups.items():
        file.write(f"{group_name}\n")
        for feature in features:
            file.write(f"{feature}\n")
        file.write("\n")

db_sha256 = hashlib.sha256(out_csv.read_bytes()).hexdigest()

metadata_lines = [
    "Elemental property database",
    "",
    f"Database version: {DB_VERSION}",
    f"Reference state: {REFERENCE_STATE}",
    f"Assumption: {PROPERTY_ASSUMPTION}",
    f"Total elements: {len(db)}",
    f"Total properties: {len(db.columns)}",
    f"SHA256: {db_sha256}",
    "",
]

metadata_out.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

print(f"Database version: {DB_VERSION}")
print(f"Total elements: {len(db)}")
print(f"Total properties: {len(db.columns)}")
print(f"Saved to: {out_csv}")
print(f"SHA256: {db_sha256}")
