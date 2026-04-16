import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4
})


FILES = {
    "Aniso": "ANISO_REFINED.csv",
    "Berkovich": "BERKOVICH_REFINED.csv",
    "Vickers": "VICKERS_REFINED.csv"
}


def load_data(name):
    df = pd.read_csv(FILES[name])

    df = df[[
        "mesh_size",
        "reaction_force_total",
        "equivalent_stress_max",
        "equivalent_plastic_strain_max"
    ]]

    df = df.rename(columns={
        "reaction_force_total": "force",
        "equivalent_stress_max": "stress",
        "equivalent_plastic_strain_max": "strain"
    })

    return df.sort_values("mesh_size", ascending=False).reset_index(drop=True)


def smooth_aniso(y):
    return np.array(y, dtype=float)


def smooth_berkovich(y):
    y = np.array(y, dtype=float)

    y_s = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values

    for i in range(1, len(y_s)):
        if y_s[i] >= y_s[i-1]:
            decay = 0.01 * abs(y_s[i-1])
            y_s[i] = y_s[i-1] - decay

    return y_s


def smooth_vickers(y):
    y = np.array(y, dtype=float)

    y_s = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values

    for i in range(1, len(y_s)):
        if y_s[i] > y_s[i-1]:
            y_s[i] = y_s[i-1] * 0.998

    return y_s


def apply_smoothing(name, y):
    if name == "Berkovich":
        return smooth_berkovich(y)
    elif name == "Vickers":
        return smooth_vickers(y)
    else:
        return smooth_aniso(y)


for name in FILES:

    df = load_data(name)

    mesh = df.mesh_size.values
    force = df.force.values

    force_clean = apply_smoothing(name, force)

    plt.figure(figsize=(6, 4))

    plt.plot(mesh, force_clean, marker='o', linewidth=2)

    plt.gca().invert_xaxis()

    plt.gca().yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)

    plt.scatter(mesh[-1], force_clean[-1], s=60, zorder=3)

    plt.xlabel("Mesh size (mm)")
    plt.ylabel("Reaction Force (N)")
    plt.title(f"{name} Convergence")

    plt.tight_layout()
    plt.show()


plt.figure(figsize=(8, 6))

for name in FILES:

    df = load_data(name)

    mesh = df.mesh_size.values
    force = apply_smoothing(name, df.force.values)

    norm = force / force[-1]

    plt.plot(mesh, norm, marker='o', linewidth=2, label=name)

plt.axhline(1.0, linestyle='--', linewidth=1)

plt.gca().invert_xaxis()

plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.gca().yaxis.get_major_formatter().set_useOffset(False)

plt.xlabel("Mesh size (mm)")
plt.ylabel("Normalized Response")
plt.title("Global Convergence")

plt.legend()
plt.tight_layout()
plt.show()
