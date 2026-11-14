from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score

PAT = Path("data/patches"); rng = np.random.default_rng(0)
E = np.load(PAT / "embeddings.npy").astype("float32"); E -= E.mean(0)
is_val = np.load(PAT / "is_val.npy")
meta = pd.read_csv(PAT / "patches_meta.csv"); desc = pd.read_csv(PAT / "descriptors.csv")
M = np.load(PAT / "matches.npz"); DM, IM, T, ei_all = M["DM"], M["IM"], float(M["T"]), M["ei_all"]
moon = (meta.body == "moon").values
base = ["slope_p50", "slope_p90", "slope_p99", "rough_m", "relief_m"]
H = np.log1p(desc[base].clip(lower=0).values.astype("float32")); H = (H - H.mean(0)) / H.std(0)

# A. Is the Moon-vs-Earth gap real terrain difference, or something beyond slope/relief?
def bal(idx, n):
    m, e = idx[moon[idx]], idx[~moon[idx]]
    return np.concatenate([rng.choice(m, min(n, len(m)), replace=False), rng.choice(e, min(n, len(e)), replace=False)])
a, b = bal(np.where(~is_val)[0], 10000), bal(np.where(is_val)[0], 4000)
print("\n[A] Moon-vs-Earth AUC (gradient boosting, held-out regions)")
for name, F in [("5 slope/relief descriptors", H), ("embedding", E)]:
    c = HistGradientBoostingClassifier(random_state=0).fit(F[a], moon[a])
    print(f"    {name:28s} AUC = {roc_auc_score(moon[b], c.predict_proba(F[b])[:, 1]):.3f}")
print("    If descriptors alone separate the bodies about as well, the gap is mostly real terrain difference.")

# B. Bootstrap CI: does the embedding add to slope/relief for shadow_always?
have = desc.shadow_always.notna().values
tr_i, va_i = np.where(have & ~is_val)[0], np.where(have & is_val)[0]
y = desc.shadow_always.values
Fb = desc[base].values.astype("float32"); Fboth = np.hstack([Fb, E])
p1 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Fb[tr_i], y[tr_i]).predict(Fb[va_i])
p2 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Fboth[tr_i], y[tr_i]).predict(Fboth[va_i])
yt = y[va_i]; d = []
for _ in range(1000):
    s = rng.integers(0, len(yt), len(yt)); d.append(r2_score(yt[s], p2[s]) - r2_score(yt[s], p1[s]))
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\n[B] shadow_always: R2(both) - R2(slope/relief) = {np.mean(d):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print("    CI above 0 = the gain is not noise (resamples patches only, not regions).")

# C. Ranking sensitivity to the match threshold
sites = meta.loc[meta.body == "earth", ["site", "role"]].drop_duplicates().set_index("site").role
n_site = meta[meta.body == "earth"].groupby("site").size()
site_of = np.asarray(meta.site, dtype=object)[ei_all[IM]]
out = {}
for f in (0.8, 1.0, 1.2):
    good = DM <= T * f
    out[f] = pd.Series({s: ((site_of == s) & good).any(1).sum() / n_site[s] * 1000 for s in sites.index})
tab = pd.DataFrame(out).round(0); tab.columns = [f"T x{c}" for c in tab.columns]; tab.insert(0, "role", sites)
print("\n[C] Per-1000 coverage at three thresholds:")
print(tab.sort_values("T x1.0", ascending=False).to_string())
for c in tab.columns[1:]:
    d_ = tab[tab.role.isin(["known_analog", "control", "background"])]
    print(f"    {c}: AUC known-vs-rest = {roc_auc_score(d_.role == 'known_analog', d_[c]):.2f}, "
          f"Spearman vs T x1.0 = {spearmanr(tab[c], tab['T x1.0'])[0]:.2f}")