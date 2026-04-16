import numpy as np
import pandas as pd
from pathlib import Path


INPUT_PATH = Path("output/MASTER_HV_DATASET_STEP1_LOCKED.csv")
OUTPUT_PATH = Path("output/HEA_descriptor_dataset.csv")


df = pd.read_csv(INPUT_PATH)
if df.empty:
    raise ValueError("Dataset is empty")


props = {
    "Al":{"r":1.43,"M":26.98,"chi":1.61,"Tm":933,"VEC":3,"E":70,"G":26,"K":76},
    "Co":{"r":1.25,"M":58.93,"chi":1.88,"Tm":1768,"VEC":9,"E":211,"G":75,"K":180},
    "Cr":{"r":1.28,"M":52.00,"chi":1.66,"Tm":2180,"VEC":6,"E":279,"G":115,"K":160},
    "Fe":{"r":1.26,"M":55.85,"chi":1.83,"Tm":1811,"VEC":8,"E":211,"G":82,"K":170},
    "Ni":{"r":1.24,"M":58.69,"chi":1.91,"Tm":1728,"VEC":10,"E":200,"G":76,"K":180},
    "Mn":{"r":1.27,"M":54.94,"chi":1.55,"Tm":1519,"VEC":7,"E":198,"G":74,"K":120},
    "Cu":{"r":1.28,"M":63.55,"chi":1.90,"Tm":1357,"VEC":11,"E":130,"G":48,"K":140},
    "Ti":{"r":1.47,"M":47.87,"chi":1.54,"Tm":1941,"VEC":4,"E":116,"G":44,"K":110},
    "V":{"r":1.34,"M":50.94,"chi":1.63,"Tm":2183,"VEC":5,"E":128,"G":47,"K":160},
    "Nb":{"r":1.46,"M":92.91,"chi":1.60,"Tm":2750,"VEC":5,"E":105,"G":38,"K":170},
    "Mo":{"r":1.39,"M":95.95,"chi":2.16,"Tm":2896,"VEC":6,"E":329,"G":126,"K":230},
    "Ta":{"r":1.46,"M":180.95,"chi":1.50,"Tm":3290,"VEC":5,"E":186,"G":69,"K":200},
    "Zr":{"r":1.60,"M":91.22,"chi":1.33,"Tm":2128,"VEC":4,"E":88,"G":33,"K":92},
    "Hf":{"r":1.59,"M":178.49,"chi":1.30,"Tm":2506,"VEC":4,"E":78,"G":30,"K":110},
    "W":{"r":1.39,"M":183.84,"chi":2.36,"Tm":3695,"VEC":6,"E":411,"G":161,"K":310},
}


elem_cols = [c for c in df.columns if c.startswith("ELEM_")]
elements_all = [c.replace("ELEM_", "") for c in elem_cols]

supported = [e for e in elements_all if e in props]
if not supported:
    raise ValueError("No supported elements found in dataset")

supported_cols = [f"ELEM_{e}" for e in supported]

comp_full = df[elem_cols].fillna(0)
coverage = comp_full[supported_cols].sum(axis=1)

df["descriptor_coverage"] = coverage

THRESHOLD = 0.9
mask = coverage >= THRESHOLD

df = df[mask].reset_index(drop=True)


comp = df[supported_cols].copy()
comp = comp.div(comp.sum(axis=1), axis=0).fillna(0)

elements = supported


def wavg(p):
    return sum(comp[f"ELEM_{e}"] * props[e][p] for e in elements)


def wvar(p):
    avg = wavg(p)
    return sum(comp[f"ELEM_{e}"] * (props[e][p] - avg) ** 2 for e in elements)


d = pd.DataFrame(index=df.index)

R = 8.314

d["Smix"] = -R * np.sum(comp * np.log(comp + 1e-12), axis=1)
d["r_avg"] = wavg("r")
d["r_var"] = wvar("r")

d["delta"] = 100 * np.sqrt(
    sum(comp[f"ELEM_{e}"] * (1 - props[e]["r"] / d["r_avg"]) ** 2 for e in elements)
)

d["M_avg"] = wavg("M")
d["chi_avg"] = wavg("chi")
d["Tm_avg"] = wavg("Tm")
d["VEC_avg"] = wavg("VEC")

d["E_avg"] = wavg("E")
d["G_avg"] = wavg("G")
d["K_avg"] = wavg("K")

d["E_var"] = wvar("E")
d["G_var"] = wvar("G")
d["K_var"] = wvar("K")

d["elastic_aniso"] = d["E_var"] / (d["E_avg"] + 1e-6)

d["fcc_proxy"] = 1 / (1 + np.exp(-(d["VEC_avg"] - 8)))
d["bcc_proxy"] = 1 / (1 + np.exp(d["VEC_avg"] - 6.87))

d["elastic_energy"] = d["G_avg"] * d["delta"] ** 2
d["bond_energy_proxy"] = d["chi_avg"] * d["Tm_avg"]

d["delta_sq"] = d["delta"] ** 2
d["G_delta"] = d["G_avg"] * d["delta"]
d["Smix_delta"] = d["Smix"] * d["delta"]


if d.isnull().values.any():
    raise ValueError("NaN values detected in descriptors")

if np.isinf(d.values).any():
    raise ValueError("Infinite values detected in descriptors")


df_out = pd.concat([df, d], axis=1)
df_out.to_csv(OUTPUT_PATH, index=False)
