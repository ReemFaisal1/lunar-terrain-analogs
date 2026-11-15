"""Polar regime maps: where each lunar terrain regime concentrates."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT, RES = Path("data/patches"), Path("results")
BIN = 12.0                                    # km per cell
meta = pd.read_csv(PAT / "patches_meta.csv")
M = np.load(PAT / "matches.npz")
reg, DM, T, mi_all = M["reg"], M["DM"], float(M["T"]), M["mi_all"]
m = (meta.body == "moon").values
x, y, r = meta.x.values[m] / 1000, meta.y.values[m] / 1000, reg[m]
cov = {k: 100 * (DM[reg[mi_all] == k][:, 0] <= T).mean() for k in range(4)}

names = {0: "R0  moderate relief", 1: "R1  gentle terrain",
         2: "R2  steep ridges and rims", 3: "R3  basins and depressions"}
lim = np.ceil(max(np.abs(x).max(), np.abs(y).max()) / BIN) * BIN
edges = np.arange(-lim, lim + BIN, BIN)
tot, _, _ = np.histogram2d(x, y, bins=[edges, edges])

fig, axes = plt.subplots(2, 2, figsize=(11, 11.6))
for ax, k in zip(axes.ravel(), range(4)):
    h, _, _ = np.histogram2d(x[r == k], y[r == k], bins=[edges, edges])
    frac = np.where(tot >= 3, h / np.maximum(tot, 1), np.nan)
    im = ax.imshow(frac.T, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="magma", vmin=0, vmax=max(np.nanpercentile(frac, 98), 0.05))
    for rad in (100, 200, 300):
        ax.add_artist(plt.Circle((0, 0), rad, fill=False, color="cyan", lw=0.6, alpha=0.45, ls=":"))
    ax.scatter([0], [0], marker="+", c="cyan", s=70, lw=1.2)
    ax.text(0.02, 0.98, f"{100 * (r == k).mean():.0f}% of lunar patches\n{cov[k]:.1f}% have an Earth match",
            transform=ax.transAxes, va="top", fontsize=8.5, color="white",
            bbox=dict(fc="black", alpha=0.55, lw=0))
    ax.set_title(names[k], fontsize=11)
    ax.set_xlabel("km from pole", fontsize=9); ax.set_ylabel("km from pole", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045, label="share of patches in cell")
fig.suptitle("Lunar south pole terrain regimes, learned from elevation alone\n"
             "dotted rings: 86.7\u00b0S, 83.4\u00b0S, 80.1\u00b0S", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(RES / "quicklook/polar_regimes.png", dpi=140); plt.close()
print(f"saved results/quicklook/polar_regimes.png  (bin {BIN:.0f} km, {m.sum()} patches)")