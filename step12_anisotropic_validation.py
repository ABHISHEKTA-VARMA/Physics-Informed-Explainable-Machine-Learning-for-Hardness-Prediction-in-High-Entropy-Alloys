import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")


INPUT_FILE  = "/content/output/ANSYS_INPUTS/ANSYS_anisotropic_static.csv"
OUTPUT_FILE = "ANISO_SIMULATION.csv"
OUT_DIR = "output_aniso_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


df_in = pd.read_csv(INPUT_FILE)
df_out = pd.read_csv(OUTPUT_FILE)

df_in.columns = df_in.columns.str.strip()
df_out.columns = df_out.columns.str.strip()


df_in = df_in.rename(columns={
    "Ex_Pa": "Ex",
    "Ey_Pa": "Ey",
    "Ez_Pa": "Ez"
})

df_out = df_out.rename(columns={
    "P16": "stress_max",
    "P12": "strain_max",
    "P24": "reaction_force_total"
})


df_in = df_in.replace([np.inf, -np.inf], np.nan)
df_in = df_in.dropna(subset=["Ex", "Ey", "Ez"])
df_in = df_in[(df_in["Ex"] > 0) & (df_in["Ey"] > 0) & (df_in["Ez"] > 0)]


df_in["E_V"] = (df_in["Ex"] + df_in["Ey"] + df_in["Ez"]) / 3
df_in["E_R"] = 1 / ((1/3) * (1/df_in["Ex"] + 1/df_in["Ey"] + 1/df_in["Ez"]))
df_in["E_H"] = (df_in["E_V"] + df_in["E_R"]) / 2

df_in["E_max"] = df_in[["Ex", "Ey", "Ez"]].max(axis=1)
df_in["E_min"] = df_in[["Ex", "Ey", "Ez"]].min(axis=1)

df_in["A_E"] = df_in["E_max"] / df_in["E_min"]
df_in["anisotropy_percent"] = (df_in["E_V"] - df_in["E_R"]) / df_in["E_V"] * 100


for col in ["E_V", "E_R", "E_H", "Ex", "Ey", "Ez"]:
    df_in[col + "_GPa"] = df_in[col] / 1e9


df_in["VRH_valid"] = (df_in["E_R"] <= df_in["E_H"]) & (df_in["E_H"] <= df_in["E_V"])


df = pd.merge(df_in, df_out, left_index=True, right_index=True, how="inner")


df = df.replace([np.inf, -np.inf], np.nan)

df_clean = df[[
    "A_E",
    "anisotropy_percent",
    "stress_max",
    "strain_max",
    "reaction_force_total"
]].dropna()

df_clean = df_clean.sort_values(by="A_E")


df_clean.to_csv(os.path.join(OUT_DIR, "clean_data.csv"), index=False)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18
})


def add_reg(x, y, ax):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    r2 = model.score(x, y)

    ax.plot(x, y_pred, "--")

    ax.text(
        0.05, 0.95,
        f"y={model.coef_[0]:.2f}x+{model.intercept_:.2f}\nR2={r2:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    return r2


plt.figure(figsize=(6, 5))

plt.plot(df_in["A_E"], df_in["E_V_GPa"], "o-", label="Voigt")
plt.plot(df_in["A_E"], df_in["E_R_GPa"], "s-", label="Reuss")
plt.plot(df_in["A_E"], df_in["E_H_GPa"], "^-", label="Hill")

plt.xlabel("A_E")
plt.ylabel("Elastic Modulus (GPa)")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "vrh_bounds.png"), dpi=600)
plt.close()


plt.figure(figsize=(6, 5))

plt.plot(df_in["A_E"], df_in["Ex_GPa"], "o-", label="Ex")
plt.plot(df_in["A_E"], df_in["Ey_GPa"], "s-", label="Ey")
plt.plot(df_in["A_E"], df_in["Ez_GPa"], "^-", label="Ez")

plt.xlabel("A_E")
plt.ylabel("Modulus (GPa)")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "directional_moduli.png"), dpi=600)
plt.close()


def plot(x, y, xl, yl, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=70)
    r2 = add_reg(x, y, ax)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=600)
    plt.close()
    return r2


r2_s = plot(df_clean["A_E"], df_clean["stress_max"], "A_E", "Stress (MPa)", "stress.png")
r2_e = plot(df_clean["A_E"], df_clean["strain_max"], "A_E", "Strain", "strain.png")
r2_f = plot(df_clean["A_E"], df_clean["reaction_force_total"], "A_E", "Force (N)", "force.png")


corr = df_clean.corr()
print(corr)

print(r2_s, r2_e, r2_f)
