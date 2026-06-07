import hashlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

EPS = 1e-12
N_STRAIN_POINTS = 25

save_dir = Path("output/step6a_fem_material_cards")
save_dir.mkdir(parents=True, exist_ok=True)

inp = Path("output/step5c_final_fem_alloys/final_fem_candidate_alloys.csv")

if not inp.exists():
    raise FileNotFoundError(f"Missing candidate alloy file: {inp}")

print("Loading candidate alloys for material property generation...")
df = pd.read_csv(inp)

if "Alloy_Family" not in df.columns:
    if "Candidate_Family" in df.columns:
        df["Alloy_Family"] = df["Candidate_Family"]
    elif "Physics_Family" in df.columns:
        df["Alloy_Family"] = df["Physics_Family"]

required_cols = [
    "Predicted_HV",
    "Composition",
    "Alloy_Family",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.sort_values(
    "Predicted_HV",
    ascending=False,
).reset_index(drop=True)

df["FEM_Alloy_ID"] = [
    f"FEM_ALLOY_{i + 1}"
    for i in range(len(df))
]

print(f"Loaded {len(df)} candidate alloys for material property generation.")


def safe_col(row, col, default):
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default


def generate_temperature_factors(row):
    tm = safe_col(row, "Tm_avg", 2800)
    refractory = safe_col(row, "refractory_fraction", 0.5)
    stability = safe_col(row, "thermoelastic_stability", 1500)

    tm_factor = np.clip(tm / 3200, 0.70, 1.00)
    ref_factor = np.clip(refractory, 0.30, 1.00)
    stab_factor = np.clip(stability / 1700, 0.70, 1.00)

    retention = (
        0.45 * tm_factor
        + 0.35 * ref_factor
        + 0.20 * stab_factor
    )

    retention = np.clip(retention, 0.60, 0.95)

    return {
        "Room_Temperature": 1.00,
        "200C": 0.92 + 0.03 * retention,
        "400C": 0.82 + 0.08 * retention,
        "600C": 0.65 + 0.15 * retention,
        "800C": 0.45 + 0.25 * retention,
    }


def generate_material_properties(row):
    hv = float(row["Predicted_HV"])
    G_GPa = safe_col(row, "G_avg", 110)
    K_GPa = safe_col(row, "K_avg", 240)
    rho_avg = safe_col(row, "rho_avg", 8.0)

    E_GPa = (9 * K_GPa * G_GPa) / (3 * K_GPa + G_GPa + EPS)

    nu = (3 * K_GPa - 2 * G_GPa) / (
        2 * (3 * K_GPa + G_GPa) + EPS
    )
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
    ys = props["Yield_Strength_MPa"] * 1e6
    uts = props["UTS_MPa"] * 1e6
    failure_strain = props["Failure_Strain"]

    strains = np.linspace(0, failure_strain, N_STRAIN_POINTS)
    stresses = []

    for strain in strains:
        if strain <= 0.002:
            stress = ys * (strain / 0.002)
        else:
            plastic_ratio = strain / max(failure_strain, 1e-6)
            n_exp = 10.0
            stress = ys + (uts - ys) * (plastic_ratio ** (1.0 / n_exp))
            stress = min(stress, uts)

        stresses.append(stress)

    return pd.DataFrame({
        "True_Strain": strains,
        "True_Stress_Pa": stresses,
    })


master_summary = []
retention_summary = []

print("Generating material property summaries...")

for _, row in df.iterrows():
    alloy_id = row["FEM_Alloy_ID"]
    alloy_dir = save_dir / alloy_id
    alloy_dir.mkdir(parents=True, exist_ok=True)

    props = generate_material_properties(row)
    rt_curve = generate_true_curve(props)

    temperature_factors = generate_temperature_factors(row)

    for temp_name, factor in temperature_factors.items():
        temp_curve = rt_curve.copy()
        temp_curve["True_Stress_Pa"] *= factor

        temp_curve.to_csv(
            alloy_dir / f"true_stress_strain_{temp_name}.csv",
            index=False,
        )

        retention_summary.append({
            "FEM_Alloy_ID": alloy_id,
            "Temperature": temp_name,
            "Normalized_Strength_Retention": factor,
        })

    plastic_curve = rt_curve.copy()
    elastic_strain = plastic_curve["True_Stress_Pa"] / (
        props["E_GPa"] * 1e9
    )

    plastic_curve["Plastic_Strain"] = np.maximum(
        plastic_curve["True_Strain"] - elastic_strain,
        0,
    )

    plastic_curve = plastic_curve[
        [
            "Plastic_Strain",
            "True_Stress_Pa",
        ]
    ]

    plastic_curve.to_csv(
        alloy_dir / "ansys_multilinear_plasticity.csv",
        index=False,
    )

    txt_path = alloy_dir / f"{alloy_id}_material_property_summary.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Material Property Summary\n\n")
        f.write(f"Alloy ID: {alloy_id}\n")
        f.write(f"Alloy Family: {row['Alloy_Family']}\n")
        f.write(f"Composition: {row['Composition']}\n\n")

        f.write("Hardness\n\n")
        f.write(f"Hardness (HV): {props['HV']:.2f}\n\n")

        f.write("Elastic Properties\n\n")
        f.write(f"Elastic Modulus (GPa): {props['E_GPa']:.2f}\n")
        f.write(f"Shear Modulus (GPa): {props['G_GPa']:.2f}\n")
        f.write(f"Bulk Modulus (GPa): {props['K_GPa']:.2f}\n")
        f.write(f"Poisson Ratio: {props['Poisson']:.4f}\n")
        f.write(f"Density (kg/m3): {props['Density']:.2f}\n\n")

        f.write("Tensile Response\n\n")
        f.write(
            f"Derived Yield Strength (MPa): "
            f"{props['Yield_Strength_MPa']:.2f}\n"
        )
        f.write(
            f"Derived Ultimate Tensile Strength (MPa): "
            f"{props['UTS_MPa']:.2f}\n"
        )
        f.write(f"Failure Strain: {props['Failure_Strain']:.4f}\n")
        f.write(
            f"Tangent Modulus (GPa): "
            f"{props['Tangent_Modulus_GPa']:.2f}\n\n"
        )

        f.write("Fatigue Parameters\n\n")
        f.write(
            f"Fatigue Strength (MPa): "
            f"{props['Fatigue_Strength_MPa']:.2f}\n"
        )
        f.write(
            f"Fatigue Limit (MPa): "
            f"{props['Fatigue_Limit_MPa']:.2f}\n\n"
        )

        f.write("Temperature-Dependent Response\n\n")
        f.write("Temperature-dependent stress-strain curves generated for:\n")
        f.write("Room temperature, 200C, 400C, 600C and 800C\n\n")

        f.write("Generated Files\n\n")

        for temp_name in temperature_factors.keys():
            f.write(f"- true_stress_strain_{temp_name}.csv\n")

        f.write("- ansys_multilinear_plasticity.csv\n\n")

        f.write("Note\n\n")
        f.write(
            "Mechanical properties were estimated using the hardness-based "
            "property generation framework described in the methodology.\n"
        )

    master_summary.append({
        "FEM_Alloy_ID": alloy_id,
        "Alloy_Family": row["Alloy_Family"],
        "Composition": row["Composition"],
        "Predicted_HV": props["HV"],
        "E_GPa": props["E_GPa"],
        "G_GPa": props["G_GPa"],
        "K_GPa": props["K_GPa"],
        "Poisson_Ratio": props["Poisson"],
        "Density_kg_m3": props["Density"],
        "Yield_Strength_MPa": props["Yield_Strength_MPa"],
        "UTS_MPa": props["UTS_MPa"],
        "Failure_Strain": props["Failure_Strain"],
        "Tangent_Modulus_GPa": props["Tangent_Modulus_GPa"],
        "Fatigue_Strength_MPa": props["Fatigue_Strength_MPa"],
        "Fatigue_Limit_MPa": props["Fatigue_Limit_MPa"],
    })

master_df = pd.DataFrame(master_summary)
retention_df = pd.DataFrame(retention_summary)

summary_file = save_dir / "master_fem_material_summary.csv"
master_df.to_csv(summary_file, index=False)

retention_file = save_dir / "temperature_strength_retention.csv"
retention_df.to_csv(retention_file, index=False)

with open(summary_file, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()

print("\nMaterial property summaries generated.")
print(f"Output directory: {save_dir}")
print(f"SHA256 checksum: {sha256}")

print("\nGenerating mechanical property summary...")

mechanical_summary = master_df[
    [
        "FEM_Alloy_ID",
        "Predicted_HV",
        "E_GPa",
        "Yield_Strength_MPa",
        "UTS_MPa",
        "Density_kg_m3",
    ]
].copy()

mechanical_summary = mechanical_summary.rename(columns={
    "FEM_Alloy_ID": "Alloy",
    "Predicted_HV": "Hardness_HV",
    "E_GPa": "Elastic_Modulus_GPa",
    "Yield_Strength_MPa": "Yield_Strength_MPa",
    "UTS_MPa": "Ultimate_Tensile_Strength_MPa",
    "Density_kg_m3": "Density_kg_m3",
})

mechanical_summary_path = save_dir / "Mechanical_Property_Summary.csv"
mechanical_summary.to_csv(mechanical_summary_path, index=False)

supplementary_summary = master_df[
    [
        "FEM_Alloy_ID",
        "Alloy_Family",
        "G_GPa",
        "K_GPa",
        "Poisson_Ratio",
        "Failure_Strain",
        "Tangent_Modulus_GPa",
        "Fatigue_Strength_MPa",
        "Fatigue_Limit_MPa",
    ]
].copy()

supplementary_summary = supplementary_summary.rename(columns={
    "FEM_Alloy_ID": "Alloy",
})

supplementary_summary_path = save_dir / "Mechanical_Property_Supplementary.csv"
supplementary_summary.to_csv(supplementary_summary_path, index=False)

top_df = master_df.head(5).copy()

actual_property_cols = [
    "Predicted_HV",
    "Yield_Strength_MPa",
    "UTS_MPa",
]

actual_property_labels = [
    "HV",
    "YS",
    "UTS",
]

x = np.arange(len(top_df))
width = 0.24

plt.figure(figsize=(9, 5))

for i, label in enumerate(actual_property_labels):
    plt.bar(
        x + (i - 1) * width,
        top_df[actual_property_cols[i]],
        width=width,
        label=label,
        edgecolor="black",
        linewidth=0.7,
    )

plt.xticks(
    x,
    top_df["FEM_Alloy_ID"],
    rotation=0,
)

plt.ylabel("Property value")
plt.xlabel("Candidate alloy")
plt.legend(frameon=False, ncol=3)
plt.tight_layout()

plt.savefig(
    save_dir / "Mechanical_Property_Actual_Values.png",
    dpi=600,
)

plt.close()

comparison_cols = [
    "Predicted_HV",
    "Yield_Strength_MPa",
    "UTS_MPa",
    "E_GPa",
]

comparison_labels = [
    "HV",
    "YS",
    "UTS",
    "E",
]

comparison_values = top_df[comparison_cols].copy()
comparison_values = comparison_values / comparison_values.max(axis=0)

width = 0.18

plt.figure(figsize=(9, 5))

for i, label in enumerate(comparison_labels):
    plt.bar(
        x + (i - 1.5) * width,
        comparison_values.iloc[:, i],
        width=width,
        label=label,
        edgecolor="black",
        linewidth=0.7,
    )

plt.xticks(
    x,
    top_df["FEM_Alloy_ID"],
    rotation=0,
)

plt.ylabel("Normalized property value")
plt.xlabel("Candidate alloy")
plt.ylim(0, 1.15)
plt.legend(frameon=False, ncol=4)
plt.tight_layout()

plt.savefig(
    save_dir / "Mechanical_Property_Comparison.png",
    dpi=600,
)

plt.close()

temperature_order = [
    "Room_Temperature",
    "200C",
    "400C",
    "600C",
    "800C",
]

temperature_labels = [
    "RT",
    "200",
    "400",
    "600",
    "800",
]

temperature_positions = np.arange(len(temperature_order))

plt.figure(figsize=(8, 5))

for alloy_id, group in retention_df.groupby("FEM_Alloy_ID"):
    group = (
        group.set_index("Temperature")
        .loc[temperature_order]
        .reset_index()
    )

    plt.plot(
        temperature_positions,
        group["Normalized_Strength_Retention"],
        marker="o",
        linewidth=1.8,
        label=alloy_id,
    )

plt.xticks(
    temperature_positions,
    temperature_labels,
)

plt.xlabel("Temperature (C)")
plt.ylabel("Normalized strength retention")
plt.ylim(0, 1.05)
plt.legend(frameon=False, ncol=2)
plt.tight_layout()

plt.savefig(
    save_dir / "Temperature_Strength_Retention.png",
    dpi=600,
)

plt.close()

print(f"Mechanical summary saved to: {mechanical_summary_path}")
print(f"Supplementary summary saved to: {supplementary_summary_path}")
print(f"Temperature response summary saved to: {retention_file}")
print("Mechanical property summary generated successfully.")
