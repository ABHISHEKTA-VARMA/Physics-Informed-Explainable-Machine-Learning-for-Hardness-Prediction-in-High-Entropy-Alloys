import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator
import warnings
from numpy.polynomial.polyutils import RankWarning

warnings.simplefilter("ignore", RankWarning)
sns.set_theme(style="whitegrid", context="talk")


vF_df = pd.read_csv("VICKERS_FORCE REACTION.csv")
vH_df = pd.read_csv("VICKERS_DEFORMATION PROBE.csv")

bF_df = pd.read_csv("BERKOVICH_FORCE REACTION.csv")
bH_df = pd.read_csv("BERKOVICH_DEFROMATION PROBE.csv")

ml_df = pd.read_csv("ml_predictions.csv")


def extract(force_df, disp_df):
    F = pd.to_numeric(force_df["Force Reaction (Total) [N]"], errors="coerce")
    h = pd.to_numeric(disp_df["Deformation Probe (Y) [mm]"], errors="coerce")

    df = pd.concat([F, h], axis=1).dropna()
    df.columns = ["F", "h"]

    df["h"] = df["h"].abs()
    df = df[(df["F"] > 0) & (df["h"] > 0)]
    df = df[df["h"] < 0.1]

    df = df.sort_values(by="h").reset_index(drop=True)
    df["F"] = np.maximum.accumulate(df["F"])

    return df


v = extract(vF_df, vH_df)
b = extract(bF_df, bH_df)


def stiffness(df):
    idx = df["F"].idxmax()
    seg = df.iloc[idx: idx + 8]

    if len(seg) < 5:
        return None

    h = seg["h"].values / 1000
    F = seg["F"].values

    order = np.argsort(h)
    h = h[order]
    F = F[order]

    try:
        slope, _ = np.polyfit(h, F, 1)
        if slope <= 0 or slope > 1e9:
            return None
        return slope
    except:
        return None


def compute_hardness(df, label):
    Pmax = df["F"].max()
    hmax = df["h"].max() / 1000

    S = stiffness(df)
    eps = 0.75

    if S is not None:
        hc = hmax - eps * (Pmax / S)
    else:
        hc = eps * hmax

    if hc <= 0 or hc > hmax:
        hc = eps * hmax

    if label == "Vickers":
        C = 27.5
        psi = 1.35
    else:
        C = 29.0
        psi = 1.45

    A = psi * C * hc**2
    H = Pmax / A
    HV = H / 9.807e6

    return HV


v_HV = compute_hardness(v, "Vickers")
b_HV = compute_hardness(b, "Berkovich")


ml_vals = pd.to_numeric(ml_df["ML_Hardness_HV"], errors="coerce").dropna()
ml_mean = ml_vals.mean()
ml_std = ml_vals.std()


def metrics(HV):
    err = abs(HV - ml_mean) / ml_mean * 100
    z = (HV - ml_mean) / ml_std
    return err, z


v_err, _ = metrics(v_HV)
b_err, _ = metrics(b_HV)


def smooth_curve(df):
    h = df["h"].values
    F = df["F"].values

    order = np.argsort(h)
    h = h[order]
    F = F[order]

    h_unique, idx = np.unique(h, return_index=True)
    F_unique = F[idx]

    h_new = np.linspace(h_unique.min(), h_unique.max(), 300)
    interpolator = PchipInterpolator(h_unique, F_unique)
    F_new = interpolator(h_new)

    return h_new, F_new


plt.figure(figsize=(7, 5), dpi=600)

h_v, F_v = smooth_curve(v)
h_b, F_b = smooth_curve(b)

plt.plot(h_v, F_v, linewidth=2.5, label="Vickers")
plt.plot(h_b, F_b, linewidth=2.5, label="Berkovich")

plt.scatter(v["h"], v["F"], s=35)
plt.scatter(b["h"], b["F"], s=35)

plt.xlabel("Indentation Depth (mm)")
plt.ylabel("Load (N)")
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5), dpi=600)

vals = [v_HV, b_HV]
labels = ["Vickers", "Berkovich"]

plt.bar(labels, vals, edgecolor="black")

plt.axhline(ml_mean, linestyle="--", linewidth=2, color="black", label="ML mean")

plt.fill_between([-0.5, 1.5],
                 ml_mean - ml_std,
                 ml_mean + ml_std,
                 alpha=0.2,
                 color="gray",
                 label="ML ±1σ")

for i, val in enumerate(vals):
    err = [v_err, b_err][i]
    plt.text(i, val + 10,
             f"{val:.1f} HV\n({err:.1f}%)",
             ha="center", fontsize=11)

plt.ylabel("Hardness (HV)")
plt.title("Hardness comparison")
plt.legend()

plt.tight_layout()
plt.show()


print("\nResults:")
print(f"ML mean        : {ml_mean:.2f} HV")
print(f"Vickers (FEM)  : {v_HV:.2f} HV  | Error: {v_err:.2f}%")
print(f"Berkovich (FEM): {b_HV:.2f} HV  | Error: {b_err:.2f}%")
