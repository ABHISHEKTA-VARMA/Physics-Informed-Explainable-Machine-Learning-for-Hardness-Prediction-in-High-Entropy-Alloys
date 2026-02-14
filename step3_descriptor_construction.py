import pandas as pd
import numpy as np
import ast
import os

# ------------------------------------------------------------
# Step 3: Descriptor construction from composition
# ------------------------------------------------------------

input_path = "output/HEA_raw_experimental_dataset_STEP1_LOCKED.csv"
df = pd.read_csv(input_path)

print("Dataset loaded from Step-1 output")
print("Initial shape:", df.shape)

# ------------------------------------------------------------
# Restore parsed composition dictionaries
# ------------------------------------------------------------
def restore_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return None
    return None

df["parsed"] = df["PARSED_COMPOSITION"].apply(restore_dict)

# ------------------------------------------------------------
# Elemental property table
# ------------------------------------------------------------
props = {
    "Al": {"r":1.43,"M":26.98,"chi":1.61,"Tm":933,"VEC":3,"E":70,"G":26,"K":76},
    "Co": {"r":1.25,"M":58.93,"chi":1.88,"Tm":1768,"VEC":9,"E":211,"G":75,"K":180},
    "Cr": {"r":1.28,"M":52.00,"chi":1.66,"Tm":2180,"VEC":6,"E":279,"G":115,"K":160},
    "Fe": {"r":1.26,"M":55.85,"chi":1.83,"Tm":1811,"VEC":8,"E":211,"G":82,"K":170},
    "Ni": {"r":1.24,"M":58.69,"chi":1.91,"Tm":1728,"VEC":10,"E":200,"G":76,"K":180},
    "Mn": {"r":1.27,"M":54.94,"chi":1.55,"Tm":1519,"VEC":7,"E":198,"G":74,"K":120},
    "Cu": {"r":1.28,"M":63.55,"chi":1.90,"Tm":1357,"VEC":11,"E":130,"G":48,"K":140},
    "Ti": {"r":1.47,"M":47.87,"chi":1.54,"Tm":1941,"VEC":4,"E":116,"G":44,"K":110},
    "V":  {"r":1.34,"M":50.94,"chi":1.63,"Tm":2183,"VEC":5,"E":128,"G":47,"K":160},
    "Nb": {"r":1.46,"M":92.91,"chi":1.60,"Tm":2750,"VEC":5,"E":105,"G":38,"K":170},
    "Mo": {"r":1.39,"M":95.95,"chi":2.16,"Tm":2896,"VEC":6,"E":329,"G":126,"K":230},
    "Ta": {"r":1.46,"M":180.95,"chi":1.50,"Tm":3290,"VEC":5,"E":186,"G":69,"K":200},
    "Zr": {"r":1.60,"M":91.22,"chi":1.33,"Tm":2128,"VEC":4,"E":88,"G":33,"K":92},
    "Hf": {"r":1.59,"M":178.49,"chi":1.30,"Tm":2506,"VEC":4,"E":78,"G":30,"K":110},
    "W":  {"r":1.39,"M":183.84,"chi":2.36,"Tm":3695,"VEC":6,"E":411,"G":161,"K":310},
}

R = 8.314

# ------------------------------------------------------------
# Keep rows with supported elements
# ------------------------------------------------------------
df["SUPPORTED_ELEMENTS"] = df["parsed"].apply(
    lambda c: isinstance(c, dict) and len(c) > 0 and all(e in props for e in c)
)

df_desc = df[
    (df["SUPPORTED_ELEMENTS"]) &
    (df["HARDNESS_AVAILABLE"])
].reset_index(drop=True)

print("Rows used for descriptor construction:", df_desc.shape[0])

if df_desc.shape[0] == 0:
    raise ValueError("No valid rows available for descriptor construction")

# ------------------------------------------------------------
# Helper functions (convert composition to fraction scale)
# ------------------------------------------------------------
def frac(c):
    return {e: v/100 for e, v in c.items()}

def avg(c, p):
    f = frac(c)
    return sum(f[e] * props[e][p] for e in f)

def var(c, p):
    f = frac(c)
    m = avg(c, p)
    return sum(f[e] * (props[e][p] - m) ** 2 for e in f)

def delta(c):
    f = frac(c)
    r_avg = avg(c, "r")
    return 100 * np.sqrt(sum(f[e] * (1 - props[e]["r"] / r_avg) ** 2 for e in f))

def entropy(c):
    f = frac(c)
    return -R * sum(f[e] * np.log(f[e] + 1e-12) for e in f)

# ------------------------------------------------------------
# Descriptor calculation
# ------------------------------------------------------------
desc = pd.DataFrame()

desc["N_elements"] = df_desc["parsed"].apply(len)
desc["std_frac"] = df_desc["parsed"].apply(lambda c: np.std(list(frac(c).values())))
desc["comp_entropy"] = df_desc["parsed"].apply(entropy)
desc["uniformity"] = 1 / (1 + desc["std_frac"])

desc["r_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "r"))
desc["r_var"] = df_desc["parsed"].apply(lambda c: var(c, "r"))
desc["delta"] = df_desc["parsed"].apply(delta)

desc["Tm_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "Tm"))
desc["Smix"] = df_desc["parsed"].apply(entropy)
desc["Omega_proxy"] = desc["Tm_avg"] * desc["Smix"]

desc["chi_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "chi"))
desc["VEC_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "VEC"))

desc["E_proxy_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "E"))
desc["G_proxy_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "G"))
desc["K_proxy_avg"] = df_desc["parsed"].apply(lambda c: avg(c, "K"))

desc["nu_proxy_avg"] = (
    3 * desc["K_proxy_avg"] - 2 * desc["G_proxy_avg"]
) / (2 * (3 * desc["K_proxy_avg"] + desc["G_proxy_avg"] + 1e-9))

desc["elastic_proxy_aniso"] = (
    df_desc["parsed"].apply(lambda c: var(c, "E")) /
    (desc["E_proxy_avg"] + 1e-6)
)

# ------------------------------------------------------------
# Save descriptor dataset
# ------------------------------------------------------------
df_out = pd.concat([df_desc, desc], axis=1)
output_path = "output/HEA_descriptor_43_dataset.csv"
df_out.to_csv(output_path, index=False)

print("Descriptor dataset saved at:", output_path)
