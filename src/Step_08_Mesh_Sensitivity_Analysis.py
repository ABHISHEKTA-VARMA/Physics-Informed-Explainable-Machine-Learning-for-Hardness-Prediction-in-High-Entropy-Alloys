import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.0,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "xtick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "ytick.major.width": 1.2,
    "ytick.minor.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 600,
})

files = {
    "Indentation": Path("INDENTATION_REFINED.csv"),
    "Fatigue": Path("FATIGUE_REFINED.csv"),
    "Tensile": Path("TENSILE_REFINED.csv"),
}

y_labels = {
    "Indentation": "Equivalent elastic strain",
    "Fatigue": "Equivalent alternating stress (Pa)",
    "Tensile": "Equivalent stress (Pa)",
}

for name, filepath in files.items():
    if not filepath.exists():
        continue

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    mesh_col = df.columns[0]
    val_col = df.columns[1]

    df = df.sort_values(mesh_col, ascending=True).reset_index(drop=True)

    fine = df.head(3)
    h1, h2, h3 = fine[mesh_col].values
    f1, f2, f3 = fine[val_col].values

    r = h2 / h1
    e21 = f2 - f1
    e32 = f3 - f2

    if e21 == 0 or e32 == 0 or (e32 / e21) <= 0:
        p = 1.0
        GCI = 0.0
    else:
        p_calc = abs(np.log(abs(e32 / e21)) / np.log(r))
        p = max(p_calc, 1.0)
        GCI = 1.25 * abs((f1 - f2) / f1) / (r**p - 1) * 100

    val_ref = f1
    low = val_ref * (1 - GCI / 100)
    high = val_ref * (1 + GCI / 100)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.fill_between(
        df[mesh_col],
        low,
        high,
        color="#ff7f0e",
        alpha=0.15,
        edgecolor="none",
        label="GCI band",
    )

    ax.axhline(
        val_ref,
        linestyle="--",
        color="#ff7f0e",
        linewidth=2,
        label="Finest mesh",
    )

    ax.plot(
        df[mesh_col],
        df[val_col],
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        markersize=7,
        mfc="white",
        mew=1.5,
        zorder=3,
        label="Simulation",
    )

    ax.set_xlabel("Mesh size (mm)")
    ax.set_ylabel(y_labels[name])
    ax.autoscale(enable=True, axis="x", tight=False)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(f"Fig_{name}_Final.png", dpi=600, bbox_inches="tight")
    plt.close()
