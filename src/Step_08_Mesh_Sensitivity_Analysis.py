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
    "legend.fontsize": 11,
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
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 600,
})

files = {
    "Indentation": Path("INDENTATION_REFINED.csv"),
    "Tensile": Path("TENSILE_REFINED.csv"),
    "Fatigue": Path("FATIGUE-REFINED.csv"),
}

y_labels = {
    "Indentation": "Equivalent Plastic Strain",
    "Tensile": "Equivalent von Mises Stress (Pa)",
    "Fatigue": "Equivalent Alternating Stress (Pa)",
}

panel_labels = [
    "(a) Indentation",
    "(b) Tensile",
    "(c) Fatigue",
]

fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
axes = axes.flatten()

for idx, (name, filepath) in enumerate(files.items()):
    ax = axes[idx]

    if not filepath.exists():
        ax.text(
            0.5,
            0.5,
            f"Missing file:\n{filepath.name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(
            panel_labels[idx],
            pad=15,
            fontweight="bold",
            fontsize=13,
        )
        continue

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    mesh_col = df.columns[0]
    val_col = df.columns[1]

    df = (
        df[[mesh_col, val_col]]
        .dropna()
        .sort_values(mesh_col, ascending=True)
        .reset_index(drop=True)
    )

    if len(df) < 3:
        ax.text(
            0.5,
            0.5,
            "At least three mesh levels are required",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(
            panel_labels[idx],
            pad=15,
            fontweight="bold",
            fontsize=13,
        )
        continue

    fine = df.head(3)

    h1, h2, h3 = fine[mesh_col].values
    f1, f2, f3 = fine[val_col].values

    r = h2 / h1
    e21 = f2 - f1
    e32 = f3 - f2

    if f1 == 0 or e21 == 0 or e32 == 0 or (e32 / e21) <= 0 or name in ["Indentation", "Fatigue"]:
        gci = np.nan
        text_str = "Non-monotonic response\nGCI not reported"
    else:
        p_calc = abs(np.log(abs(e32 / e21)) / np.log(r))
        p = max(p_calc, 1.0)
        gci = 1.25 * abs((f1 - f2) / f1) / (r**p - 1) * 100
        text_str = f"GCI = {gci:.2f}%\nObserved order $p$ = {p:.2f}"

    val_ref = f1

    if name == "Tensile" and np.isfinite(gci) and gci > 0:
        low = val_ref * (1 - gci / 100)
        high = val_ref * (1 + gci / 100)

        ax.fill_between(
            df[mesh_col],
            low,
            high,
            color="#ff7f0e",
            alpha=0.08,
            edgecolor="none",
        )

    ax.axhline(
        val_ref,
        linestyle="--",
        color="#ff7f0e",
        linewidth=2,
        label="Fine mesh value",
    )

    ax.plot(
        df[mesh_col],
        df[val_col],
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        markersize=8,
        mfc="white",
        mew=1.8,
        zorder=3,
        label="Computed response",
    )

    ax.set_xlabel("Mesh size, $h$ (mm)")
    ax.set_ylabel(y_labels[name])

    ax.invert_xaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    ax.set_ylim(
        ylim[0] - 0.05 * yrange,
        ylim[1] + 0.25 * yrange,
    )

    ax.text(
        0.95,
        0.95,
        text_str,
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
        fontsize=9,
    )

    if idx == 0:
        ax.legend(loc="lower left", frameon=False)

    ax.set_title(
        panel_labels[idx],
        pad=15,
        fontweight="bold",
        fontsize=13,
    )

plt.tight_layout()

output_file = Path("Fig_3_5_5_Mesh_Convergence.png")

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print(f"Saved figure: {output_file}")
