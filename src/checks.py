"""Robustness checks: block bootstrap on the shadow gain, site-size confound on the ranking."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score

PAT, RES = Path("data/patches"), Path("results")
rng = np.random.default_rng(0)
E = np.load(PAT / "embeddings.npy").astype("float32"); E -= E.mean(0)
is_val = np.load(PAT / "is_val.npy")
meta = pd.read_csv(PAT / "patches_meta.csv"); desc = pd.read_csv(PAT / "descriptors.csv")
base = ["slope_p50", "slope_p90", "slope_p99", "rough_m", "relief_m"]

# ---- 1. spatial-block bootstrap on the shadow_always gain
have = desc.shadow_always.notna().values
tr_i, va_i = np.where(have & ~is_val)[0], np.where(have & is_val)[0]
y = desc.shadow_always.values
Fb = desc[base].values.astype("float32"); Fboth = np.hstack([Fb, E])
p1 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Fb[tr_i], y[tr_i]).predict(Fb[va_i])
p2 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Fboth[tr_i], y[tr_i]).predict(Fboth[va_i])
yt = y[va_i]
print(f"\npoint estimate: R2(both) - R2(slope/relief) = "
      f"{r2_score(yt, p2) - r2_score(yt, p1):.3f}  ({len(yt)} test patches)")

blk = (meta.site + "_" + (meta.row // 256).astype(str) + "_" + (meta.col // 256).astype(str)).values[va_i]
ublk = np.unique(blk); by = {b: np.where(blk == b)[0] for b in ublk}
print(f"these fall in {len(ublk)} spatial blocks (~20 km each)")

def boot(sampler, n=2000):
    out = []
    for _ in range(n):
        r = sampler()
        if np.var(yt[r]) < 1e-12:
            continue
        out.append(r2_score(yt[r], p2[r]) - r2_score(yt[r], p1[r]))
    return np.array(out)

d_pat = boot(lambda: rng.integers(0, len(yt), len(yt)))
d_blk = boot(lambda: np.concatenate([by[b] for b in rng.choice(ublk, len(ublk), replace=True)]))
print("\n[1] Bootstrap 95% CI on the shadow gain")
for nm, d in [("patch-level (too narrow)", d_pat), ("block-level (honest)", d_blk)]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"    {nm:26s} mean {d.mean():+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(gain>0) = {(d > 0).mean():.3f}")

# ---- 2. is the site ranking just site size?
tab = pd.read_csv(RES / "validation_site_ranking.csv", index_col=0)
x, v = np.log(tab.n_patches.values), tab.emb_per1000.values
resid = v - np.polyval(np.polyfit(x, v, 1), x)
print(f"\n[2] Site size vs coverage score: Spearman = {spearmanr(x, v)[0]:+.2f}")
print(f"    mean patches - known analogs {tab.n_patches[tab.role == 'known_analog'].mean():.0f}, "
      f"others {tab.n_patches[~tab.role.isin(['known_analog', 'saudi'])].mean():.0f}")
m = tab.role.isin(["known_analog", "control", "background"]).values
kn = (tab.role.values == "known_analog")[m]
for nm, val in [("raw score", v[m]), ("size-adjusted residual", resid[m])]:
    obs = val[kn].mean() - val[~kn].mean()
    null = np.array([(lambda p: val[p].mean() - val[~p].mean())(rng.permutation(kn)) for _ in range(10000)])
    print(f"    {nm:24s} AUC = {roc_auc_score(kn, val):.2f}  permutation p = {(null >= obs).mean():.3f}")
tab["size_adjusted"] = resid.round(0)
print("\n    size-adjusted top 10:")
print(tab.sort_values("size_adjusted", ascending=False)[["role", "n_patches", "emb_per1000", "size_adjusted"]].head(10).to_string())
tab.to_csv(RES / "validation_site_ranking.csv")