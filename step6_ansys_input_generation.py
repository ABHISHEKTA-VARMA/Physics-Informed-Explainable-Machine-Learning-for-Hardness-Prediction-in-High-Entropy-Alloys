import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Step 6: ANSYS input preparation
# SI unit system (N–m–Pa)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
input_path = "output/HEA_descriptor_43_dataset.csv"
output_dir = "output/ANSYS_INPUTS_Q1"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# Load descriptor dataset
# ------------------------------------------------------------
df = pd.read_csv(input_path)

# Detect hardness column dynamically
hardness_cols = [
    c for c in df.columns
    if ("hv" in c.lower()) or ("hardness" in c.lower())
]

if not hardness_cols:
    raise ValueError("Hardness column not found in descriptor dataset")

target_col = hardness_cols[0]

# Keep rows with valid hardness values
df = df[df[target_col].notnull() & (df[target_col] > 0)].reset_index(drop=True)

print("Dataset used for ANSYS inputs:", df.shape)

# ------------------------------------------------------------
# Required descriptors
# ------------------------------------------------------------
required_cols = [
    "E_proxy_avg",        # Elastic modulus proxy (MPa)
    "nu_proxy_avg",       # Poisson's ratio proxy
    "elastic_proxy_aniso" # Dimensionless descriptor
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required descriptor: {col}")

# ------------------------------------------------------------
# Representative elastic statistics
# ------------------------------------------------------------
N_CASES = 8

# Unit conversion: MPa → Pa
E_proxy_Pa = df["E_proxy_avg"] * 1.0e6

# Check for limited variation
if E_proxy_Pa.nunique() < 5:
    print("Warning: limited variation in elastic modulus proxy")

# Elastic modulus sweep (Pa)
E_vals = np.linspace(
    E_proxy_Pa.quantile(0.05),
    E_proxy_Pa.quantile(0.95),
    N_CASES
)

E_ref = E_proxy_Pa.median()
nu_ref = df["nu_proxy_avg"].median()

# Physical bounds check
if not (0.0 < nu_ref < 0.5):
    raise ValueError("Poisson ratio proxy outside physical bounds")

# ------------------------------------------------------------
# Contact indentation parameter set
# ------------------------------------------------------------
# Unit conversion: mm → m
indent_df = pd.DataFrame({
    "Case": [f"Indent_{i+1}" for i in range(N_CASES)],
    "Youngs_Modulus_Pa": E_vals,
    "Poissons_Ratio": nu_ref,
    "Indenter_Radius_m": np.linspace(0.5e-3, 5.0e-3, N_CASES)
})

indent_df.to_csv(
    f"{output_dir}/ANSYS_contact_indentation.csv",
    index=False
)

# ------------------------------------------------------------
# Elastic anisotropy parameter set
# ------------------------------------------------------------
# Isotropic shear modulus (Pa)
G_iso = E_ref / (2.0 * (1.0 + nu_ref))

# Controlled anisotropy amplitude
delta_max = 0.30
aniso_levels = np.linspace(0.0, 1.0, 5)

aniso_rows = []
for i, a in enumerate(aniso_levels, start=1):
    aniso_rows.append({
        "Case": f"Aniso_{i}",
        "Ex_Pa": E_ref * (1.0 + delta_max * a),
        "Ey_Pa": E_ref,
        "Ez_Pa": E_ref * (1.0 - delta_max * a),
        "nu_xy": nu_ref,
        "nu_yz": nu_ref,
        "nu_xz": nu_ref,
        "Gxy_Pa": G_iso,
        "Gyz_Pa": G_iso,
        "Gxz_Pa": G_iso
    })

aniso_df = pd.DataFrame(aniso_rows)

# ------------------------------------------------------------
# Basic physical checks
# ------------------------------------------------------------
assert (aniso_df[["Ex_Pa", "Ey_Pa", "Ez_Pa"]] > 0).all().all()
assert (aniso_df[["Gxy_Pa", "Gyz_Pa", "Gxz_Pa"]] > 0).all().all()
assert (aniso_df[["nu_xy", "nu_yz", "nu_xz"]] > 0).all().all()
assert (aniso_df[["nu_xy", "nu_yz", "nu_xz"]] < 0.5).all().all()

aniso_df.to_csv(
    f"{output_dir}/ANSYS_anisotropic_static.csv",
    index=False
)

# ------------------------------------------------------------
# Completion message
# ------------------------------------------------------------
print("ANSYS input generation completed.")
print("Generated files:")
print(" - ANSYS_contact_indentation.csv")
print(" - ANSYS_anisotropic_static.csv")
print("Unit system used: N – m – Pa")
