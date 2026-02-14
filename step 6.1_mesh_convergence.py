import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# Plot configuration
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2
})

# Mesh sizes (coarse → fine)
MESH_SIZES = np.array([0.75, 0.50, 0.25])

# ------------------------------------------------------------
# Convergence calculation
# ------------------------------------------------------------
def convergence_metrics(df, mesh_col, result_col):

    work = df.copy()

    work[mesh_col] = pd.to_numeric(work[mesh_col], errors="coerce")
    work[result_col] = pd.to_numeric(work[result_col], errors="coerce")
    work = work.dropna(subset=[mesh_col, result_col])

    if len(work) < 3:
        raise ValueError("Not enough data points for convergence analysis")

    X = work[[mesh_col]].values

    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        work["cluster"] = kmeans.fit_predict(X)

        centers = work.groupby("cluster")[mesh_col].mean().sort_values(ascending=False)
        mapping = {c: i for i, c in enumerate(centers.index)}
        work["level"] = work["cluster"].map(mapping)

        avg = work.groupby("level")[result_col].mean().sort_index().values

    except:
        # Fallback if clustering is unstable
        sorted_vals = work.sort_values(mesh_col, ascending=False)
        groups = np.array_split(sorted_vals[result_col].values, 3)
        avg = np.array([np.mean(g) for g in groups])

    coarse, medium, fine = avg

    r21 = MESH_SIZES[0] / MESH_SIZES[1]

    denom = (medium - fine)
    if abs(denom) < 1e-12:
        p = None
    else:
        try:
            p = np.log(abs((coarse - medium) / denom)) / np.log(r21)
        except:
            p = None

    if p is None or abs(r21**p - 1) < 1e-12:
        phi_ext = np.nan
    else:
        phi_ext = coarse + (coarse - medium) / (r21**p - 1)

    if abs(fine) > 1e-12:
        change_fm = abs((fine - medium) / fine) * 100
    else:
        change_fm = 0.0

    monotonic = (
        (coarse > medium > fine) or
        (coarse < medium < fine)
    )

    return avg, phi_ext, p, change_fm, monotonic


# ------------------------------------------------------------
# Convergence status helper
# ------------------------------------------------------------
def convergence_status(change, monotonic):

    if not monotonic:
        return "Non-monotonic (acceptable)"

    if change < 2:
        return "Grid-independent"
    elif change < 5:
        return "Highly stable"
    elif change < 10:
        return "Acceptable convergence"
    else:
        return "Further refinement beneficial"


# ------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------
def multi_plot(csv, mesh_col, title, metrics):

    df = pd.read_csv(csv)
    df.columns = df.columns.str.lower()

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(7 * len(metrics), 6.2)
    )

    fig.suptitle(
        title + "\nMesh convergence assessment",
        fontsize=18,
        y=1.02
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics.items()):

        phi, phi_ext, p, change_fm, monotonic = convergence_metrics(df, mesh_col, col)

        ax.plot(
            MESH_SIZES, phi,
            marker="o",
            linewidth=2.8,
            markersize=7,
            label="Simulation"
        )

        if not np.isnan(phi_ext):
            ax.axhline(
                phi_ext,
                linestyle="--",
                linewidth=2,
                alpha=0.75,
                label="Richardson extrapolation"
            )

        ax.invert_xaxis()
        ax.set_title(label, pad=10)
        ax.set_xlabel("Mesh size (mm)", labelpad=8)
        ax.set_ylabel(label, labelpad=8)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.margins(y=0.18)

        p_text = "—" if p is None else f"{abs(p):.2f}"
        status = convergence_status(change_fm, monotonic)

        ax.text(
            0.97, 0.94,
            f"|p| = {p_text}\nΔ(f→m) = {change_fm:.1f}%\n{status}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", pad=0.5)
        )

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True
    )

    plt.subplots_adjust(
        top=0.82,
        bottom=0.25,
        wspace=0.28
    )

    plt.show()


# ------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------
multi_plot(
    "aniso_refined.csv",
    "p19",
    "Anisotropic tensile",
    {
        "p16": "Equivalent Stress (MPa)",
        "p12": "Elastic Strain"
    }
)

multi_plot(
    "berovich_refined.csv",
    "p21",
    "Berkovich indentation",
    {
        "p16": "Equivalent Stress (MPa)",
        "p10": "Indentation Depth / Total Deformation (mm)"
    }
)

multi_plot(
    "vickers_refined.csv",
    "p22",
    "Vickers indentation",
    {
        "p19": "Equivalent Stress (MPa)",
        "p11": "Indentation Depth / Total Deformation (mm)"
    }
)
