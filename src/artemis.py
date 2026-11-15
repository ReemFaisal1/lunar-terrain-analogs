"""Artemis III candidate landing regions: terrain regime, Earth coverage, Earth attribution."""
import glob
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT, RES = Path("data/patches"), Path("results")
R_MOON = 1737400.0
PATCH_KM = 64 * 80 / 1000.0

REGIONS = [                       # centres from the USGS/IAU Gazetteer of Planetary Nomenclature
    ("Peak near Shackleton", "Shackleton",    -89.90,   0.00, 19.0),
    ("de Gerlache Rim",      "de Gerlache",   -88.50, -87.10, 32.4),
    ("Faustini Rim A",       "Faustini",      -87.18,  85.02, 42.5),
    ("Haworth",              "Haworth",       -86.90,  -4.00, 51.4),
    ("Malapert Massif",      "Mons Malapert", -85.98,  -1.67, 67.0),
    ("Nobile Rim",           "Nobile",        -85.28,  53.27, 79.3),
]

meta = pd.read_csv(PAT / "patches_meta.csv")
M = np.load(PAT / "matches.npz")
reg, DM, IM, T, mi_all, ei_all = M["reg"], M["DM"], M["IM"], float(M["T"]), M["mi_all"], M["ei_all"]
site_of = np.asarray(meta.site, dtype=object)
earth_share = meta[meta.body == "earth"].site.value_counts(normalize=True)
glob_cov = 100 * (DM[:, 0] <= T).mean()

import rasterio
from pyproj import Transformer
f = glob.glob("data/**/*LDEM*80S*.tif*", recursive=True)[0]
with rasterio.open(f) as src:
    crs = src.crs
tf = Transformer.from_crs(f"+proj=longlat +R={R_MOON} +no_defs", crs, always_xy=True)
xy = [tf.transform(lon, lat) for _, _, lat, lon, _ in REGIONS]

mx, my = meta.x.values[mi_all], meta.y.values[mi_all]
rows = []
for (name, feat, lat, lon, diam), (px, py) in zip(REGIONS, xy):
    d = np.hypot(mx - px, my - py) / 1000.0
    sel = d <= diam / 2 + PATCH_KM / 2
    rg, dm, good = reg[mi_all][sel], DM[sel], DM[sel] <= T
    s = pd.Series(site_of[ei_all[IM[sel]]][good]).value_counts(normalize=True)
    enr = (s / earth_share).dropna().sort_values(ascending=False).head(3)
    rows.append(dict(
        region=name, feature=feat, lat=lat, lon=lon, radius_km=diam / 2,
        x_km=px / 1000, y_km=py / 1000, n_patches=int(sel.sum()),
        pole_km=round(np.hypot(px, py) / 1000, 1),
        R2_share_pct=round(100 * (rg == 2).mean(), 1),
        covered_pct=round(100 * good[:, 0].mean(), 1),
        median_d_over_T=round(float(np.median(dm[:, 0]) / T), 2),
        top_earth=", ".join(f"{k} {100*v:.0f}%" for k, v in s.head(3).items()),
        top_earth_enriched=", ".join(f"{k} {v:.1f}x" for k, v in enr.items())))

tab = pd.DataFrame(rows).sort_values("covered_pct")
print(f"\nGlobal lunar coverage for reference: {glob_cov:.1f}%\n")
print(tab.drop(columns=["x_km", "y_km", "lon", "radius_km"]).to_string(index=False))
tab.to_csv(RES / "artemis_regions.csv", index=False)

OFF = {"Peak near Shackleton": (0, -34), "de Gerlache Rim": (-10, 20), "Faustini Rim A": (34, -20),
       "Haworth": (-42, 6), "Malapert Massif": (-20, 26), "Nobile Rim": (30, 16)}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 6.6))
sc = a1.scatter(tab.x_km, tab.y_km, s=np.maximum(tab.radius_km, 14) ** 1.75, c=tab.covered_pct,
                cmap="RdYlGn", vmin=0, vmax=45, edgecolors="black", linewidths=1.3, zorder=3)
for _, r in tab.iterrows():
    dx, dy = OFF[r.region]
    a1.annotate(f"{r.region}\n{r.covered_pct:.0f}% matched", (r.x_km, r.y_km), fontsize=9,
                xytext=(dx, dy), textcoords="offset points", ha="center", zorder=4,
                bbox=dict(fc="white", ec="0.7", lw=0.5, pad=0.3, alpha=0.92),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))
for rad, lab in ((50, "88.3\u00b0S"), (100, "86.7\u00b0S"), (150, "85.0\u00b0S")):
    a1.add_artist(plt.Circle((0, 0), rad, fill=False, color="gray", lw=0.6, ls=":", alpha=0.8))
    a1.text(rad * 0.71, -rad * 0.71, lab, fontsize=7, color="gray", ha="center")
a1.scatter([0], [0], marker="+", c="black", s=80, lw=1.3, zorder=3)
a1.text(4, -8, "south pole", fontsize=7.5, color="0.3")
a1.set_xlim(-110, 200); a1.set_ylim(-80, 185); a1.set_aspect("equal")
a1.set_xlabel("km from pole"); a1.set_ylabel("km from pole")
a1.set_title("Artemis III candidate regions\nmarker size = feature diameter", fontsize=11)
fig.colorbar(sc, ax=a1, fraction=0.045, label="% of region's patches matched")

a2.barh(tab.region, tab.covered_pct, color=plt.cm.RdYlGn(tab.covered_pct / 45), edgecolor="black", lw=0.7)
a2.axvline(glob_cov, color="black", ls="--", lw=1)
a2.text(glob_cov + 0.4, len(tab) - 0.35, f"whole south pole {glob_cov:.1f}%", fontsize=8, va="center")
for i, r in enumerate(tab.itertuples()):
    a2.text(r.covered_pct + 0.6, i, f"R2 {r.R2_share_pct:.0f}%", va="center", fontsize=8.5, color="#555555")
a2.set_xlim(0, 48)
a2.set_xlabel("% of the region's patches with a terrestrial match")
a2.set_title("Grey: share of the region that is R2,\nthe regime Earth cannot supply", fontsize=11)
plt.tight_layout(); plt.savefig(RES / "quicklook/artemis_regions.png", dpi=150); plt.close()
print("\nsaved results/artemis_regions.csv and results/quicklook/artemis_regions.png")