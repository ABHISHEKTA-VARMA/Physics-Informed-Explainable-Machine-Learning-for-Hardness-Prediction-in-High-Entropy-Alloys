import pandas as pd
import re
from pathlib import Path


BORG_PATH = Path("MPEA_dataset.csv")
GORSSE_PATH = Path("Gorsse_HV_FINAL.csv")
GE_PATH = Path("GE_RefractoryAlloyScreeningDataset_FINAL.csv")
OUTPUT_PATH = Path("MASTER_HV_DATASET.csv")


def parse_formula(formula):
    """
    Convert a chemical formula into normalized elemental fractions.
    """
    parts = re.findall(r'([A-Z][a-z]*)([0-9\.]*)', str(formula))

    comp = {}
    for el, val in parts:
        comp[f"ELEM_{el}"] = float(val) if val else 1.0

    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total for k, v in comp.items()}

    return comp


borg = pd.read_csv(BORG_PATH)
if borg.empty:
    raise ValueError("BORG dataset is empty")

borg = borg.dropna(subset=["PROPERTY: HV"])

borg_elem = pd.DataFrame(borg["FORMULA"].apply(parse_formula).tolist()).fillna(0)
borg_elem["PROPERTY: HV"] = borg["PROPERTY: HV"].values
borg_elem["Source"] = "Borg"
borg_elem["HV_origin"] = "Experimental"


gorsse = pd.read_csv(GORSSE_PATH)
if gorsse.empty:
    raise ValueError("GORSSE dataset is empty")

gorsse_elem = pd.DataFrame(
    gorsse["Composition"].apply(parse_formula).tolist()
).fillna(0)

gorsse_elem["PROPERTY: HV"] = gorsse["HV"]
gorsse_elem["Source"] = "Gorsse"
gorsse_elem["HV_origin"] = "Experimental"


ge = pd.read_csv(GE_PATH)
if ge.empty:
    raise ValueError("GE dataset is empty")

element_map = {
    "Hf(at%)": "ELEM_Hf",
    "Mo(at%)": "ELEM_Mo",
    "Nb(at%)": "ELEM_Nb",
    "Re(at%)": "ELEM_Re",
    "Ru(at%)": "ELEM_Ru",
    "Ta(at%)": "ELEM_Ta",
    "Ti(at%)": "ELEM_Ti",
    "W(at%)": "ELEM_W",
    "Zr(at%)": "ELEM_Zr",
}

ge_elem = pd.DataFrame()

for old, new in element_map.items():
    if old in ge.columns:
        ge_elem[new] = ge[old]

ge_elem = ge_elem.fillna(0)

row_sums = ge_elem.sum(axis=1)
row_sums[row_sums == 0] = 1
ge_elem = ge_elem.div(row_sums, axis=0)

if "Hardness (GPa)" in ge.columns:
    ge_elem["PROPERTY: HV"] = ge["Hardness (GPa)"] * 100
    ge_elem["HV_origin"] = "Derived_from_GPa"
    ge_elem["HV_conversion_note"] = "Linear scaling from GPa to HV"

ge_elem["Source"] = "GE"


master = pd.concat([borg_elem, gorsse_elem, ge_elem], ignore_index=True)
master = master.fillna(0)
master = master.drop_duplicates()


elem_cols = [c for c in master.columns if c.startswith("ELEM_")]

row_sums = master[elem_cols].sum(axis=1)
row_sums[row_sums == 0] = 1
master[elem_cols] = master[elem_cols].div(row_sums, axis=0)


composition_sum = master[elem_cols].sum(axis=1)
invalid = master[abs(composition_sum - 1) > 1e-4]

if len(invalid) > 0:
    print(f"{len(invalid)} rows adjusted during normalization")


master["num_elements"] = (master[elem_cols] > 0).sum(axis=1)
master["HV_unit"] = "kgf/mm^2"

master["processing_note"] = (
    "Merged Borg, Gorsse, and GE datasets; compositions normalized; duplicates removed"
)


master.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset saved to: {OUTPUT_PATH}")
