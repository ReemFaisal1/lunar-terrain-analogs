"""Lunar baseline for the coverage numbers.

The match threshold T is the median distance from a held-out lunar patch to its nearest
lunar patch in other regions. So by construction about 50% of held-out lunar patches are
"matched" by the Moon itself, and Earth's coverage has to be read against that, not
against 100%. This script measures the baseline per regime and puts Earth's coverage
beside it, on the same held-out patches and the same threshold.

It also cuts the Earth pool down to the size of the lunar pool, because nearest-neighbour
distances shrink as the reference set grows and Earth has the bigger pool.

Run from the project root:  python src/baseline_check.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

PAT, RES = Path("data/patches"), Path("results")
rng = np.random.default_rng(0)
E = np.load(PAT / "embeddings.npy").astype("float32")
E -= E.mean(0)
is_val = np.load(PAT / "is_val.npy")
meta = pd.read_csv(PAT / "patches_meta.csv")
M = np.load(PAT / "matches.npz")
reg, DM, T = M["reg"], M["DM"], float(M["T"])
mi_all, ei_all, mi_tr = M["mi_all"], M["ei_all"], M["mi_tr"]
moon = (meta.body == "moon").values
mi_va = np.where(moon & is_val)[0]


def knn1(Q, R, chunk=1000):
    R = R.astype("float32")
    r2 = (R ** 2).sum(1)
    out = np.empty(len(Q), "float32")
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d2 = (q ** 2).sum(1)[:, None] + r2[None, :] - 2 * q @ R.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


d_moon = knn1(E[mi_va], E[mi_tr])                       # held-out Moon -> Moon, other regions
d_earth = DM[np.searchsorted(mi_all, mi_va), 0]         # same patches -> all Earth patches
pool = rng.choice(ei_all, min(len(mi_tr), len(ei_all)), replace=False)
d_earth_eq = knn1(E[mi_va], E[pool])                    # Earth pool cut to the lunar pool's size

r = reg[mi_va]
names = {1: "R1 gentle", 0: "R0 moderate", 3: "R3 basins", 2: "R2 ridges/rims"}
rows = []
for k, m in [(k, r == k) for k in (1, 0, 3, 2)] + [("all", np.ones(len(r), bool))]:
    mo = 100 * (d_moon[m] <= T).mean()
    ea = 100 * (d_earth[m] <= T).mean()
    eq = 100 * (d_earth_eq[m] <= T).mean()
    rows.append(dict(
        regime=names.get(k, "all held-out"), n_heldout=int(m.sum()),
        moon_matches_itself_pct=round(mo, 1),
        earth_pct=round(ea, 1),
        earth_equal_pool_pct=round(eq, 1),
        earth_over_moon=round(eq / mo, 2) if mo > 0 else np.nan,
        median_d_moon_over_T=round(float(np.median(d_moon[m])) / T, 2),
        median_d_earth_over_T=round(float(np.median(d_earth[m])) / T, 2)))

tab = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"\nThreshold T = {T:.3f}. Held-out lunar patches: {len(mi_va)}; "
      f"lunar pool {len(mi_tr)}, Earth pool {len(ei_all)} (cut to {len(pool)} for the equal-pool column)\n")
print(tab.to_string(index=False))
tab.to_csv(RES / "baseline_check.csv", index=False)
print("\nRead earth_over_moon as: how good a source of this terrain Earth is, relative to the Moon itself.")
print("1.0 = as good as other lunar regions; 0.5 = half as good; near 0 = Earth genuinely lacks it.")
print("saved results/baseline_check.csv")