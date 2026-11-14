from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT, RES = Path("data/patches"), Path("results")
(RES / "quicklook").mkdir(parents=True, exist_ok=True)
K_OVERRIDE = None            # set e.g. K_OVERRIDE = 6 to force the number of regimes
rng = np.random.default_rng(0)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

E = np.load(PAT / "embeddings.npy").astype("float32")
E -= E.mean(0)
is_val = np.load(PAT / "is_val.npy")
meta = pd.read_csv(PAT / "patches_meta.csv")
desc = pd.read_csv(PAT / "descriptors.csv")
moon = (meta.body == "moon").values
mi_all, ei_all = np.where(moon)[0], np.where(~moon)[0]
mi_tr, mi_va = np.where(moon & ~is_val)[0], np.where(moon & is_val)[0]
known_zone = ((meta.role == "known_analog") & meta.in_zone).values
is_saudi = (meta.role == "saudi").values
r_km = np.hypot(meta.x, meta.y).values / 1000.0

def knn(Q, R, k, chunk=1000):
    R = R.astype("float32"); r2 = (R ** 2).sum(1)
    D = np.empty((len(Q), k), "float32"); I = np.empty((len(Q), k), "int64")
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk].astype("float32")
        d2 = (q ** 2).sum(1)[:, None] + r2[None, :] - 2 * q @ R.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d2, idx, 1)
        o = np.argsort(dk, 1)
        I[s:s + chunk] = np.take_along_axis(idx, o, 1)
        D[s:s + chunk] = np.sqrt(np.maximum(np.take_along_axis(dk, o, 1), 0))
    return D, I

# 1. how many regimes?
sub = rng.choice(mi_tr, min(6000, len(mi_tr)), replace=False)
perm = rng.permutation(mi_tr); A, B = perm[:len(perm) // 2], perm[len(perm) // 2:]
rows = []
for k in range(3, 11):
    km = KMeans(k, n_init=5, random_state=0).fit(E[sub])
    sil = silhouette_score(E[sub], km.labels_)
    la = KMeans(k, n_init=3, random_state=1).fit(E[A]).predict(E[mi_va])
    lb = KMeans(k, n_init=3, random_state=2).fit(E[B]).predict(E[mi_va])
    rows.append((k, sil, adjusted_rand_score(la, lb)))
ktab = pd.DataFrame(rows, columns=["k", "silhouette", "stability_ARI"])
print("\nChoosing k (silhouette = separation, ARI = stability across independent halves):")
print(ktab.round(3).to_string(index=False))
ok = ktab[(ktab.k >= 4) & (ktab.stability_ARI >= 0.6)]
K = int(K_OVERRIDE or (ok.loc[ok.silhouette.idxmax(), "k"] if len(ok) else ktab.loc[ktab.stability_ARI.idxmax(), "k"]))
print(f"-> using k = {K}")
km = KMeans(K, n_init=10, random_state=0).fit(E[mi_tr])
reg = np.full(len(meta), -1); reg[mi_all] = km.predict(E[mi_all])

# 2. regime profiles from the independent descriptors
d = desc.copy(); d["regime"] = reg; d["r_km"] = r_km
cols = ["slope_p50", "slope_p90", "relief_m", "rough_m", "shadow_always", "route_ratio"]
prof = d[d.regime >= 0].groupby("regime")[cols].median()
prof.insert(0, "n_patches", pd.Series(reg[mi_all]).value_counts().sort_index())
prof["pole_dist_km"] = d[d.regime >= 0].groupby("regime")["r_km"].median()
print("\nLunar terrain regimes (median descriptors; never shown to the model):")
print(prof.round(2).to_string())
prof.to_csv(RES / "regime_profiles.csv")

# 3. what counts as a good match?
Dmm, _ = knn(E[mi_va], E[mi_tr], 1)
T = float(np.median(Dmm[:, 0]))
print(f"\nMatch threshold T = {T:.3f} (median distance from a held-out lunar patch to its nearest lunar "
      f"patch from other regions). 'Good match' = Earth patch at least this close.")

# 4. lunar-centric coverage
DM, IM = knn(E[mi_all], E[ei_all], 5)
good = DM <= T
earth_idx = ei_all[IM]
cov_rows = []
for r in range(K):
    m = reg[mi_all] == r
    g = good[m]; sites = meta.site.values[earth_idx[m]][g]
    top = pd.Series(sites).value_counts(normalize=True).head(3)
    cov_rows.append(dict(
        regime=r, n_patches=int(m.sum()), covered_pct=round(100 * g[:, 0].mean(), 1),
        dist_vs_T=round(float(np.median(DM[m][:, 0]) / T), 2),
        known_zone_share_pct=round(100 * known_zone[earth_idx[m]][g].mean(), 1) if g.any() else np.nan,
        saudi_share_pct=round(100 * is_saudi[earth_idx[m]][g].mean(), 1) if g.any() else np.nan,
        top_sites=", ".join(f"{s} {100*v:.0f}%" for s, v in top.items())))
cov = pd.DataFrame(cov_rows)
print("\nCoverage per lunar regime (covered_pct = % of its patches with an Earth patch within T; "
      "dist_vs_T <= 1 means typically well matched):")
print(cov.to_string(index=False))
cov.to_csv(RES / "regime_coverage.csv", index=False)

# 5. Earth-centric summary per site
DE, IE = knn(E[ei_all], E[mi_tr], 10)
vote = pd.DataFrame(reg[mi_tr][IE]).mode(axis=1)[0].values.astype(int)
es = pd.DataFrame(dict(site=meta.site.values[ei_all], role=meta.role.values[ei_all],
                       good=DE[:, 0] <= T, regime=vote, d_over_T=DE[:, 0] / T))
def site_row(g):
    gg = g[g.good]
    return pd.Series(dict(n_patches=len(g), good_pct=round(100 * g.good.mean(), 1),
                          median_d_over_T=round(g.d_over_T.median(), 2),
                          main_regime=int(gg.regime.mode()[0]) if len(gg) else -1))
ssum = (es.groupby(["role", "site"]).apply(site_row, include_groups=False).reset_index()
        .sort_values(["role", "good_pct"], ascending=[True, False]))
print("\nEarth sites: % of patches that are good lunar matches, and the lunar regime they best match (-1 = none):")
print(ssum.to_string(index=False))
ssum.to_csv(RES / "site_match_summary.csv", index=False)
np.savez(PAT / "matches.npz", reg=reg, DM=DM, IM=IM, DE=DE, IE=IE, T=T,
         mi_all=mi_all, ei_all=ei_all, mi_tr=mi_tr)

# 6. figures
pca = PCA(2).fit(E[np.concatenate([rng.choice(mi_all, 6000, replace=False), rng.choice(ei_all, 6000, replace=False)])])
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
se, sm = rng.choice(ei_all, 6000, replace=False), rng.choice(mi_all, 6000, replace=False)
pe, pm = pca.transform(E[se]), pca.transform(E[sm])
ax[0].scatter(*pe.T, s=3, c="lightgray")
sc = ax[0].scatter(*pm.T, s=3, c=reg[sm], cmap="tab10", vmin=0, vmax=9)
ax[0].set_title("Embedding space (PCA): Earth grey, Moon coloured by regime")
ax[0].legend(*sc.legend_elements(), title="regime", loc="best", fontsize=7)
ax[1].scatter(meta.x[moon] / 1000, meta.y[moon] / 1000, c=reg[moon], cmap="tab10", vmin=0, vmax=9, s=1)
ax[1].set_aspect("equal"); ax[1].set_title("Lunar south pole (80-90 S): regimes (km from pole)")
plt.tight_layout(); plt.savefig(RES / "quicklook/regimes_overview.png", dpi=110); plt.close()

X = np.load(PAT / "patches.npy", mmap_mode="r")
cent = km.cluster_centers_
fig, ax = plt.subplots(K, 10, figsize=(17, 1.75 * K)); ax = np.atleast_2d(ax)
for r in range(K):
    mc = mi_all[reg[mi_all] == r]
    near_m = mc[np.argsort(((E[mc] - cent[r]) ** 2).sum(1))[:5]]
    near_e = ei_all[np.argsort(((E[ei_all] - cent[r]) ** 2).sum(1))[:5]]
    for j, i in enumerate(list(near_m) + list(near_e)):
        a = ax[r, j]; p = np.asarray(X[i]).astype("float32")
        a.imshow(p, cmap="terrain"); a.axis("off")
        a.set_title(("moon" if j < 5 else meta.site[i][:10]) + f" {np.ptp(p):.0f}m", fontsize=7)
    ax[r, 0].text(-0.15, 0.5, f"R{r}", transform=ax[r, 0].transAxes, fontsize=10, ha="right", va="center")
plt.tight_layout(); plt.savefig(RES / "quicklook/regime_examples.png", dpi=100); plt.close()
print("\nSaved results/regime_profiles.csv, regime_coverage.csv, site_match_summary.csv, "
      "quicklook/regimes_overview.png, quicklook/regime_examples.png")