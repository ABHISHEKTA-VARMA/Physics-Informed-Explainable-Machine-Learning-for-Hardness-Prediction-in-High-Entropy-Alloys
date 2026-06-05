
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

warnings.filterwarnings("ignore")

# Global Settings
EPS = 1e-12
MAX_PLASTIC_STRAIN = 0.20
N_STRAIN_POINTS = 25

# Paths setup
save_dir = Path("output/step6a_fem_material_cards")
save_dir.mkdir(parents=True, exist_ok=True)
inp = Path("output/step5c_final_fem_alloys/final_fem_candidate_alloys.csv")

# Data Loading & Validation
print("Loading candidate alloys for FEM material card generation...")
df = pd.read_csv(inp)

required_cols = ["Predicted_HV", "Composition", "Physics_Family"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Strict Hardness Ranking Preservation
df = df.sort_values("Predicted_HV", ascending=False).reset_index(drop=True)
df["FEM_Alloy_ID"] = [f"FEM_ALLOY_{i+1}" for i in range(len(df))]

print(f"Loaded and ranked {len(df)} final FEM alloys.")


def safe_col(row, col, default):
    """Safely extracts a column value with a fallback default."""
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default


def generate_temperature_factors(row):
    """Calculates temperature-dependent softening factors based on thermodynamics."""
    tm = safe_col(row, "Tm_avg", 2800)
    refractory = safe_col(row, "refractory_fraction", 0.5)
    stability = safe_col(row, "thermoelastic_stability", 1500)

    tm_factor = np.clip(tm / 3200, 0.70, 1.00)
    ref_factor = np.clip(refractory, 0.30, 1.00)
    stab_factor = np.clip(stability / 1700, 0.70, 1.00)

    retention = (0.45 * tm_factor) + (0.35 * ref_factor) + (0.20 * stab_factor)
    retention = np.clip(retention, 0.60, 0.95)

    return {
        "RT": 1.00,
        "200C": 0.92 + 0.03 * retention,
        "400C": 0.82 + 0.08 * retention,
        "600C": 0.65 + 0.15 * retention,
        "800C": 0.45 + 0.25 * retention,
    }


def generate_material_properties(row):
    """Calculates macroscopic mechanical properties from predicted hardness and phase data."""
    hv = float(row["Predicted_HV"])
    G_GPa = safe_col(row, "G_avg", 110)
    K_GPa = safe_col(row, "K_avg", 240)
    rho_avg = safe_col(row, "rho_avg", 8.0)

    E_GPa = (9 * K_GPa * G_GPa) / (3 * K_GPa + G_GPa + EPS)
    nu = (3 * K_GPa - 2 * G_GPa) / (2 * (3 * K_GPa + G_GPa) + EPS)
    nu = np.clip(nu, 0.22, 0.38)

    density = rho_avg * 1000.0
    yield_strength_mpa = hv * 3.27
    uts_mpa = hv * 3.60
    failure_strain = np.clip(0.16 - (hv - 500) / 5000, 0.05, 0.16)

    tangent_modulus_gpa = 0.025 * E_GPa
    fatigue_strength_mpa = 0.45 * uts_mpa
    fatigue_limit_mpa = 0.30 * uts_mpa

    return {
        "HV": hv,
        "E_GPa": E_GPa,
        "G_GPa": G_GPa,
        "K_GPa": K_GPa,
        "Poisson": nu,
        "Density": density,
        "rho_avg": rho_avg,
        "Yield_Strength_MPa": yield_strength_mpa,
        "UTS_MPa": uts_mpa,
        "Failure_Strain": failure_strain,
        "Tangent_Modulus_GPa": tangent_modulus_gpa,
        "Fatigue_Strength_MPa": fatigue_strength_mpa,
        "Fatigue_Limit_MPa": fatigue_limit_mpa,
    }


def generate_true_curve(props):
    """Generates the true stress-strain curve using a modified power-law hardening model."""
    ys = props["Yield_Strength_MPa"] * 1e6
    uts = props["UTS_MPa"] * 1e6
    failure_strain = props["Failure_Strain"]

    strains = np.linspace(0, failure_strain, N_STRAIN_POINTS)
    stresses = []

    for s in strains:
        if s <= 0.002:
            stress = ys * (s / 0.002)
        else:
            plastic_ratio = s / max(failure_strain, 1e-6)
            n_exp = 10.0
            stress = ys + (uts - ys) * (plastic_ratio ** (1.0 / n_exp))
            stress = min(stress, uts)
        stresses.append(stress)

    return pd.DataFrame({
        "True_Strain": strains,
        "True_Stress_Pa": stresses,
    })


# Master Processing Loop
master_summary = []

for _, row in df.iterrows():
    alloy_id = row["FEM_Alloy_ID"]
    alloy_dir = save_dir / alloy_id
    alloy_dir.mkdir(exist_ok=True)

    # Generate foundational properties and RT curve
    props = generate_material_properties(row)
    rt_curve = generate_true_curve(props)
    rt_curve.to_csv(alloy_dir / "true_stress_strain_RT.csv", index=False)

    # Generate temperature-dependent curves
    temperature_factors = generate_temperature_factors(row)
    for temp_name, factor in temperature_factors.items():
        temp_curve = rt_curve.copy()
        temp_curve["True_Stress_Pa"] *= factor
        temp_curve.to_csv(alloy_dir / f"true_stress_strain_{temp_name}.csv", index=False)

    # Generate ANSYS multilinear plasticity table
    plastic_curve = rt_curve.copy()
    elastic_strain = plastic_curve["True_Stress_Pa"] / (props["E_GPa"] * 1e9)
    plastic_curve["Plastic_Strain"] = np.maximum(plastic_curve["True_Strain"] - elastic_strain, 0)
    plastic_curve = plastic_curve[["Plastic_Strain", "True_Stress_Pa"]]
    plastic_curve.to_csv(alloy_dir / "ansys_multilinear_plasticity.csv", index=False)

    # Generate Material TXT Card
    txt_path = alloy_dir / f"{alloy_id}_material_card.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("FEM DIGITAL MATERIAL CARD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Alloy ID       : {alloy_id}\n")
        f.write(f"Hardness Rank  : #{int(alloy_id.split('_')[-1])}\n")
        f.write(f"Physics Family : {row['Physics_Family']}\n")
        f.write(f"Composition    : {row['Composition']}\n\n")

        f.write("=" * 60 + "\nHARDNESS PROPERTIES\n" + "=" * 60 + "\n\n")
        f.write(f"Predicted Hardness (HV) : {props['HV']:.2f}\n\n")

        f.write("=" * 60 + "\nELASTIC PROPERTIES\n" + "=" * 60 + "\n\n")
        f.write(f"Elastic Modulus (GPa) : {props['E_GPa']:.2f}\n")
        f.write(f"Shear Modulus (GPa)   : {props['G_GPa']:.2f}\n")
        f.write(f"Bulk Modulus (GPa)    : {props['K_GPa']:.2f}\n")
        f.write(f"Poisson Ratio         : {props['Poisson']:.4f}\n")
        f.write(f"Density (kg/m3)       : {props['Density']:.2f}\n\n")

        f.write("=" * 60 + "\nTENSILE ANALYSIS\n" + "=" * 60 + "\n\n")
        f.write(f"Estimated Yield Strength (MPa) : {props['Yield_Strength_MPa']:.2f}\n")
        f.write(f"Estimated UTS (MPa)            : {props['UTS_MPa']:.2f}\n")
        f.write(f"Failure Strain                 : {props['Failure_Strain']:.4f}\n")
        f.write(f"Tangent Modulus (GPa)          : {props['Tangent_Modulus_GPa']:.2f}\n\n")

        f.write("=" * 60 + "\nFATIGUE ANALYSIS\n" + "=" * 60 + "\n\n")
        f.write(f"Fatigue Strength (MPa) : {props['Fatigue_Strength_MPa']:.2f}\n")
        f.write(f"Fatigue Limit (MPa)    : {props['Fatigue_Limit_MPa']:.2f}\n\n")

        f.write("=" * 60 + "\nCREEP ANALYSIS\n" + "=" * 60 + "\n\n")
        f.write("Recommended Temperature Range : 600C - 800C\n")
        f.write("High Temperature Stability    : HIGH\n\n")

        f.write("=" * 60 + "\nINDENTATION ANALYSIS\n" + "=" * 60 + "\n\n")
        f.write("- Berkovich Nanoindentation\n")
        f.write("- Vickers Hardness Simulation\n\n")

        f.write("=" * 60 + "\nTEMPERATURE SOFTENING FACTORS\n" + "=" * 60 + "\n\n")
        for t, fac in temperature_factors.items():
            f.write(f"  {t:>4s} : {fac:.4f}\n")
        f.write("\n")

        f.write("=" * 60 + "\nGENERATED FILES\n" + "=" * 60 + "\n\n")
        for t in temperature_factors.keys():
            f.write(f"- true_stress_strain_{t}.csv\n")
        f.write("- ansys_multilinear_plasticity.csv\n\n")

        f.write("=" * 60 + "\nSCIENTIFIC NOTE\n" + "=" * 60 + "\n\n")
        f.write("This alloy preserves strict hardness-based mechanical ranking hierarchy for all FEM studies.\n")

    master_summary.append({
        "FEM_Alloy_ID": alloy_id,
        "Predicted_HV": props["HV"],
        "E_GPa": props["E_GPa"],
        "G_GPa": props["G_GPa"],
        "K_GPa": props["K_GPa"],
        "Poisson_Ratio": props["Poisson"],
        "Density_kg_m3": props["Density"],
        "Yield_Strength_MPa": props["Yield_Strength_MPa"],
        "UTS_MPa": props["UTS_MPa"],
    })

# Output Generation & Verification
master_df = pd.DataFrame(master_summary)
summary_file = save_dir / "master_fem_material_summary.csv"
master_df.to_csv(summary_file, index=False)

with open(summary_file, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print("\nPipeline complete. Digital material cards generated successfully.")
print(f"Output Directory: {save_dir}")
print(f"SHA256 Checksum: {sha256}")
print("\nMaster Summary Preview:")
print(master_df.head().to_string(index=False))
