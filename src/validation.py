from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

PAT, RES = Path("data/patches"), Path("results")
rng = np.random.default_rng(0)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

E = np.load(PAT / "embeddings.npy").astype("float32"); E -= E.mean(0)
is_val = np.load(PAT / "is_val.npy")
meta = pd.read_csv(PAT / "patches_meta.csv"); desc = pd.read_csv(PAT / "descriptors.csv")
M = np.load(PAT / "matches.npz")
reg, DM, IM, T = M["reg"], M["DM"], M["IM"], float(M["T"])
mi_all, ei_all, mi_tr = M["mi_all"], M["ei_all"], M["mi_tr"]
moon = (meta.body == "moon").values
mi_va = np.where(moon & is_val)[0]

def knn(Q, R, k, chunk=1000):
    R = R.astype("float32"); r2 = (R ** 2).sum(1)
    D = np.empty((len(Q), k), "float32"); I = np.empty((len(Q), k), "int64")
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk].astype("float32")
        d2 = (q ** 2).sum(1)[:, None] + r2[None, :] - 2 * q @ R.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d2, idx, 1); o = np.argsort(dk, 1)
        I[s:s + chunk] = np.take_along_axis(idx, o, 1)
        D[s:s + chunk] = np.sqrt(np.maximum(np.take_along_axis(dk, o, 1), 0))
    return D, I

# 1. can a classifier tell Moon from Earth in the embedding?
tr, va = np.where(~is_val)[0], np.where(is_val)[0]
def bal(idx, n):
    m, e = idx[moon[idx]], idx[~moon[idx]]
    return np.concatenate([rng.choice(m, min(n, len(m)), replace=False), rng.choice(e, min(n, len(e)), replace=False)])
a, b = bal(tr, 10000), bal(va, 4000)
clf = LogisticRegression(max_iter=2000).fit(E[a], moon[a])
auc = roc_auc_score(moon[b], clf.predict_proba(E[b])[:, 1])
print(f"\n[1] Moon-vs-Earth classifier on embeddings (held-out regions): AUC = {auc:.3f}")
print("    0.5 = bodies fully mixed; 1.0 = trivially separable.")

# 2. does the embedding explain more than slope/relief?
base = ["slope_p50", "slope_p90", "slope_p99", "rough_m", "relief_m"]
have = desc.shadow_always.notna().values & desc.route_ratio.notna().values
tr_i, va_i = np.where(have & ~is_val)[0], np.where(have & is_val)[0]
Fb = desc[base].values.astype("float32")
feats = {"slope/relief only": Fb, "embedding only": E, "both": np.hstack([Fb, E])}
rows = []
for t in ["shadow_always", "route_ratio"]:
    y_all = desc[t].values
    for name, F in feats.items():
        mdl = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(F[tr_i], y_all[tr_i])
        p = mdl.predict(F[va_i]); y = y_all[va_i]; mm = moon[va_i]
        rows.append(dict(target=t, features=name, R2_all=round(r2_score(y, p), 3),
                         R2_moon_only=round(r2_score(y[mm], p[mm]), 3) if mm.sum() > 20 else np.nan,
                         R2_earth_only=round(r2_score(y[~mm], p[~mm]), 3) if (~mm).sum() > 20 else np.nan))
pred = pd.DataFrame(rows)
print(f"\n[2] Predicting independent descriptors on held-out regions ({len(tr_i)} train / {len(va_i)} test; small, so noisy):")
print(pred.to_string(index=False))
print("    Does 'both' beat 'slope/relief only'? If not, the embedding adds nothing beyond slope for that target.")
pred.to_csv(RES / "validation_predictability.csv", index=False)

# 3. site contributions: embedding vs hand-built features
H = np.log1p(desc[base].clip(lower=0).values.astype("float32"))
H = (H - H.mean(0)) / (H.std(0) + 1e-6)
Dmm_h, _ = knn(H[mi_va], H[mi_tr], 1); Th = float(np.median(Dmm_h[:, 0]))
DMh, IMh = knn(H[mi_all], H[ei_all], 5)
sites = meta.loc[meta.body == "earth", ["site", "role"]].drop_duplicates().set_index("site").role
n_site = meta[meta.body == "earth"].groupby("site").size()

def contrib(Dx, Ix, Tx):
    good = Dx <= Tx; site_of = np.asarray(meta.site, dtype=object)[ei_all[Ix]]
    return pd.Series({s: int(((site_of == s) & good).any(1).sum()) for s in sites.index})

ce, ch = contrib(DM, IM, T), contrib(DMh, IMh, Th)
tab = pd.DataFrame(dict(role=sites, n_patches=n_site,
                        emb_per1000=(ce / n_site * 1000).round(1), hand_per1000=(ch / n_site * 1000).round(1)))
tab["emb_rank"] = rankdata(-tab.emb_per1000, method="min").astype(int)
tab["hand_rank"] = rankdata(-tab.hand_per1000, method="min").astype(int)
tab = tab.sort_values("emb_rank")
print("\n[3] Distinct lunar patches covered per 1000 Earth patches of each site:")
print(tab.to_string())
tab.to_csv(RES / "validation_site_ranking.csv")
print(f"    Spearman agreement, embedding vs hand-built ranking: {spearmanr(tab.emb_per1000, tab.hand_per1000)[0]:.2f}")

def perm_test(col, n=10000):
    d = tab[tab.role.isin(["known_analog", "control", "background"])]
    y = (d.role == "known_analog").values; v = d[col].values.astype(float)
    obs = v[y].mean() - v[~y].mean()
    null = [(lambda p: v[p].mean() - v[~p].mean())(rng.permutation(y)) for _ in range(n)]
    return obs, float((np.array(null) >= obs).mean()), roc_auc_score(y, v)
print("\n[4] Are known analogs enriched vs controls/background (Saudi excluded)?")
for col, nm in [("emb_per1000", "embedding"), ("hand_per1000", "hand-built features")]:
    obs, p, a_ = perm_test(col)
    print(f"    {nm:20s} AUC = {a_:.2f}  mean difference = {obs:.1f}  permutation p = {p:.3f}")
ctrl = tab[tab.role == "control"]
print("    control ranks (out of", len(tab), ") - embedding:", dict(ctrl.emb_rank), "| hand-built:", dict(ctrl.hand_rank))

# 5. regime coverage by method
rr = []
for r in sorted(set(reg[mi_all])):
    m = reg[mi_all] == r
    rr.append(dict(regime=r, emb_covered_pct=round(100 * (DM[m][:, 0] <= T).mean(), 1),
                   hand_covered_pct=round(100 * (DMh[m][:, 0] <= Th).mean(), 1)))
print("\n[5] Regime coverage by method:")
print(pd.DataFrame(rr).to_string(index=False))