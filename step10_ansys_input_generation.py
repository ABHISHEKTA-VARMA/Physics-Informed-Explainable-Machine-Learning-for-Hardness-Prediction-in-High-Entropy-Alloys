import numpy as np
import pandas as pd
from pathlib import Path


np.random.seed(42)


INPUT_PATH = Path("output/HEA_descriptor_dataset.csv")
OUTPUT_DIR = Path("output/ANSYS_INPUTS")
OUTPUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(INPUT_PATH)

target_col = "PROPERTY: HV"

df = df[df[target_col].notnull() & (df[target_col] > 0)].copy()


required_cols = ["E_avg", "G_avg", "K_avg"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing descriptor: {col}")


G = df["G_avg"].values * 1e9
K = df["K_avg"].values * 1e9


denominator = (2 * (3 * K + G))
denominator[denominator == 0] = 1e-9

nu = (3 * K - 2 * G) / denominator
nu = np.clip(nu, 0.18, 0.38)


E = (9 * K * G) / (3 * K + G + 1e-9)
E_reconstructed = 2 * G * (1 + nu)

consistency_error = np.abs(E - E_reconstructed) / (E + 1e-9)


E_ref = np.median(E)
nu_ref = np.median(nu)

E_low = np.percentile(E, 5)
E_high = np.percentile(E, 95)

N_CASES = 10


E_samples = np.random.normal(E_ref, 0.1 * E_ref, N_CASES)
E_samples = np.clip(E_samples, E_low, E_high)


indent_df = pd.DataFrame({
    "Case": [f"Indent_{i+1}" for i in range(N_CASES)],
    "Youngs_Modulus_Pa": E_samples,
    "Poissons_Ratio": nu_ref,
    "Indenter_Radius_m": np.linspace(1e-6, 50e-6, N_CASES),
    "Friction_Coeff": np.linspace(0.05, 0.25, N_CASES)
})

indent_df.to_csv(
    OUTPUT_DIR / "ANSYS_contact_indentation.csv",
    index=False
)


G_iso = E_ref / (2 * (1 + nu_ref))

delta_levels = np.linspace(0, 0.3, 6)

rows = []

for i, d in enumerate(delta_levels, 1):

    Ex = E_ref * (1 + d)
    Ey = E_ref
    Ez = E_ref * (1 - d)

    rows.append({
        "Case": f"Aniso_{i}",
        "Ex_Pa": Ex,
        "Ey_Pa": Ey,
        "Ez_Pa": Ez,
        "nu_xy": nu_ref,
        "nu_yz": nu_ref,
        "nu_xz": nu_ref,
        "Gxy_Pa": G_iso,
        "Gyz_Pa": G_iso,
        "Gxz_Pa": G_iso
    })

aniso_df = pd.DataFrame(rows)

aniso_df.to_csv(
    OUTPUT_DIR / "ANSYS_anisotropic_static.csv",
    index=False
)


ys = (df[target_col].values * 9.807 / 3.0) * 1e6


material_df = pd.DataFrame({
    "Youngs_Modulus_Pa": E,
    "Poissons_Ratio": nu,
    "Yield_Strength_Pa": ys
})


material_df.sample(
    min(12, len(material_df)),
    random_state=42
).to_csv(
    OUTPUT_DIR / "ANSYS_material_strength_cases.csv",
    index=False
)


with open(OUTPUT_DIR / "README_STEP10.txt", "w") as f:

    f.write("ANSYS material input generation\n\n")
    f.write(f"Samples used: {len(df)}\n")
    f.write(f"E_ref (GPa): {E_ref/1e9:.2f}\n")
    f.write(f"nu_ref: {nu_ref:.3f}\n\n")

    f.write("Details:\n")
    f.write("- Isotropic baseline with derived elastic constants\n")
    f.write("- Yield strength estimated from hardness\n")
    f.write("- Sampling constrained within dataset bounds\n")
