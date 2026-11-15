"""Export matched Moon/Earth patch pairs as hillshade thumbnails + JSON for the web demo.

Run from the project root:  python src/export_web.py
Writes:  docs/assets/pairs/*.png  and  docs/data/pairs.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT, RES, OUT = Path("data/patches"), Path("results"), Path("docs/assets/pairs")
OUT.mkdir(parents=True, exist_ok=True)
Path("docs/data").mkdir(parents=True, exist_ok=True)

X = np.load(PAT / "patches.npy", mmap_mode="r")
meta = pd.read_csv(PAT / "patches_meta.csv")
desc = pd.read_csv(PAT / "descriptors.csv")
M = np.load(PAT / "matches.npz")
reg, DM, IM, T = M["reg"], M["DM"], M["IM"], float(M["T"])
mi_all, ei_all = M["mi_all"], M["ei_all"]
art = pd.read_csv(RES / "artemis_regions.csv")

NAMES = {0: "moderate relief", 1: "gentle terrain",
         2: "steep ridges and rims", 3: "basins and depressions"}
PRETTY = {
    "atacama": "Atacama, Chile", "hawaii": "Hawaii, USA",
    "haughton": "Haughton Crater, Canada", "death_valley": "Death Valley, USA",
    "askja": "Askja, Iceland", "hekla": "Hekla, Iceland", "etna": "Mount Etna, Italy",
    "craters_moon": "Craters of the Moon, USA", "dry_valleys": "Dry Valleys, Antarctica",
    "grand_canyon": "Grand Canyon, USA", "tibesti": "Tibesti, Chad",
    "ahaggar": "Ahaggar, Algeria", "tuwayq": "Tuwayq Escarpment, Saudi Arabia",
    "asir": "Asir Mountains, Saudi Arabia", "rahat": "Harrat Rahat, Saudi Arabia",
    "khaybar": "Harrat Khaybar, Saudi Arabia", "namib": "Namib Desert, Namibia",
    "gobi": "Gobi Desert", "sinai": "Sinai, Egypt", "andes": "Andes", "alps": "Alps",
    "himalaya": "Himalaya", "svalbard": "Svalbard, Norway",
    "wahbah": "Wahbah Crater, Saudi Arabia", "rub_al_khali": "Rub' al Khali, Saudi Arabia",
    "ctrl_plains": "Great Plains, USA", "ctrl_smokies": "Great Smoky Mountains, USA",
}


def hillshade(z, az=315.0, alt=40.0, res=80.0):
    gy, gx = np.gradient(np.asarray(z, dtype="float64"), res)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    a, z0 = np.radians(alt), np.radians(az)
    return np.clip(np.sin(a) * np.cos(slope)
                   + np.cos(a) * np.sin(slope) * np.cos(z0 - aspect), 0, 1)


def save(i, name):
    plt.imsave(OUT / f"{name}.png", hillshade(np.asarray(X[i]).astype("float32")),
               cmap="gray", vmin=0, vmax=1)


def stats(i):
    d = desc.loc[i]
    s = dict(relief_m=round(float(d.relief_m)), slope_p50=round(float(d.slope_p50), 1),
             rough_m=round(float(d.rough_m), 2))
    if "shadow_always" in desc.columns and pd.notna(d.shadow_always):
        s["enclosure"] = round(float(d.shadow_always), 3)
    return s


pairs, seen = [], set()


def add(gi, label, group, note):
    k = int(np.where(mi_all == gi)[0][0])
    ei = int(ei_all[IM[k, 0]])
    d = float(DM[k, 0])
    pid = f"p{len(pairs):02d}"
    save(gi, f"{pid}_moon")
    save(ei, f"{pid}_earth")
    pairs.append(dict(
        id=pid, group=group, label=label, note=note, hero=False,
        regime=int(reg[gi]), regime_name=NAMES[int(reg[gi])],
        d_over_T=round(d / T, 2), matched=bool(d <= T),
        moon=stats(gi), earth=stats(ei),
        earth_site=meta.site[ei], earth_place=PRETTY.get(meta.site[ei], meta.site[ei]),
        earth_role=meta.role[ei]))
    seen.add(gi)


def pick(idx, q):
    """global patch index at quantile q of match distance within idx (indices into mi_all)"""
    o = np.asarray(idx)[np.argsort(DM[np.asarray(idx), 0])]
    for j in o[min(int(q * len(o)), len(o) - 1):]:
        if int(mi_all[j]) not in seen:
            return int(mi_all[j])
    return int(mi_all[o[-1]])


# --- a typical patch from each Artemis candidate region
mx, my = meta.x.values[mi_all], meta.y.values[mi_all]
for _, r in art.sort_values("covered_pct", ascending=False).iterrows():
    sel = np.where(np.hypot(mx - r.x_km * 1000, my - r.y_km * 1000) / 1000
                   <= r.radius_km + 2.56)[0]
    if len(sel):
        add(pick(sel, 0.5), r.region, "artemis", "typical patch in this region")

# --- each regime: its closest terrestrial match, and a typical member
for k in range(4):
    inreg = np.where(reg[mi_all] == k)[0]
    if not len(inreg):
        continue
    add(pick(inreg, 0.0), f"R{k}  {NAMES[k]}", "regime", "closest match found anywhere")
    add(pick(inreg, 0.5), f"R{k}  {NAMES[k]}", "regime", "typical member of this regime")

# --- lunar terrain whose nearest neighbour is a Saudi site
for site in ("tuwayq", "asir"):
    es = set(np.where(meta.site.values == site)[0])
    hit = [j for j in range(len(mi_all)) if int(ei_all[IM[j, 0]]) in es]
    if hit:
        for q in (0.0, 0.3):
            add(pick(hit, q), f"Lunar terrain nearest {PRETTY[site].split(',')[0]}",
                "saudi", "its closest terrestrial neighbour")

# --- hero: structured terrain that is nonetheless well matched
best = None
for p in pairs:
    if p["group"] == "regime" and p["regime"] in (0, 2, 3) and p["matched"]:
        if best is None or p["d_over_T"] < best["d_over_T"]:
            best = p
(best or pairs[0])["hero"] = True

json.dump(pairs, open("docs/data/pairs.json", "w"), indent=1)
print(f"\nwrote {len(pairs)} pairs -> docs/assets/pairs/*.png and docs/data/pairs.json\n")
for p in pairs:
    flag = "" if p["matched"] else "   NO MATCH"
    print(f"  {p['id']:4s} {p['group']:8s} {p['label'][:28]:30s} R{p['regime']}  "
          f"d/T {p['d_over_T']:.2f}  -> {p['earth_place']}{flag}{'   [hero]' if p['hero'] else ''}")