
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Paths setup
inp = Path("output/step6a_fem_material_cards/master_fem_material_summary.csv")
save_dir = Path("output/step6b_fatigue_sn_curves")
save_dir.mkdir(parents=True, exist_ok=True)

# Data Loading
print("Loading Step 6A master summary...")
df = pd.read_csv(inp)
print(f"Loaded alloys: {len(df)}")

# Standard fatigue life evaluation points
cycles = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])

# Processing Loop
summary = []

for _, row in df.iterrows():
    alloy_id = row["FEM_Alloy_ID"]
    uts = float(row["UTS_MPa"])
    ys = float(row["Yield_Strength_MPa"])

    # Consistent with Step 6A macroscopic property estimations
    fatigue_strength = 0.45 * uts
    fatigue_limit = 0.30 * uts

    # Basquin Equation Parameters calculation
    N1, N2 = 1e3, 1e7
    S1, S2 = fatigue_strength, fatigue_limit

    b = np.log10(S2 / S1) / np.log10(N2 / N1)
    A = S1 / (N1 ** b)
    stresses = A * (cycles ** b)

    # S-N Table Generation
    sn_df = pd.DataFrame({
        "Cycles": cycles.astype(int),
        "Stress_Amplitude_MPa": np.round(stresses, 3)
    })

    alloy_dir = save_dir / alloy_id
    alloy_dir.mkdir(exist_ok=True)

    # General and ANSYS-specific fatigue inputs
    sn_df.to_csv(alloy_dir / "sn_curve.csv", index=False)
    sn_df.to_csv(alloy_dir / "ansys_sn_curve.csv", index=False)

    summary.append({
        "FEM_Alloy_ID": alloy_id,
        "Yield_Strength_MPa": ys,
        "UTS_MPa": uts,
        "Fatigue_Strength_MPa": fatigue_strength,
        "Fatigue_Limit_MPa": fatigue_limit,
        "Basquin_b": round(b, 6)
    })

    print(f"Generated S-N curve: {alloy_id}")

# Save master summary for Step 6B
summary_df = pd.DataFrame(summary)
summary_df.to_csv(save_dir / "sn_curve_summary.csv", index=False)

print(f"\nStep 6B pipeline complete. Generated {len(df)} S-N curves.")
print(f"Output Directory: {save_dir}")
