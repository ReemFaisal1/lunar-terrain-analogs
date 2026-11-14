import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from sites import SITES

PROC, OUT = Path("data/processed"), Path("data/patches")
OUT.mkdir(parents=True, exist_ok=True)
P, STRIDE = 64, 32
MAX_NAN, MAX_WATER = 0.02, 0.05
yy, xx = np.mgrid[0:P, 0:P]
A = np.c_[xx.ravel(), yy.ravel(), np.ones(P*P)].astype("float64")
PINV = np.linalg.pinv(A)

def detrend(a):                      # remove best-fit plane; keep relief amplitude
    coef = PINV @ a.ravel().astype("float64")
    return (a - (A @ coef).reshape(P, P)).astype("float32")

def water_mask(z):                   # near sea level AND locally flat
    zz = np.nan_to_num(z.astype("float64"), nan=1e4)
    m = uniform_filter(zz, 3)
    s = np.sqrt(np.maximum(uniform_filter(zz*zz, 3) - m*m, 0))
    return (zz > -25) & (zz < 5) & (s < 1.0)

def extract(name, z, meta, body, coastal):
    x0, dx, _, y0, _, dy = meta["transform"]
    bad = water_mask(z) if coastal else None
    H, W = z.shape
    out, rows = [], []
    for r in range(0, H-P+1, STRIDE):
        for c in range(0, W-P+1, STRIDE):
            a = z[r:r+P, c:c+P]
            if np.isnan(a).mean() > MAX_NAN: continue
            if bad is not None and bad[r:r+P, c:c+P].mean() > MAX_WATER: continue
            a = np.where(np.isnan(a), np.nanmean(a), a)
            out.append(detrend(a).astype("float16"))
            rows.append((body, name, r, c, x0+(c+P/2)*dx, y0+(r+P/2)*dy))
    return out, rows

if __name__ == "__main__":
    X, R = [], []
    z = np.load(PROC/"moon_south_pole.npy"); meta = json.load(open(PROC/"moon_south_pole.json"))
    o, r = extract("moon_south_pole", z, meta, "moon", False); X += o; R += r
    for n, s in SITES.items():
        f = PROC/f"{n}.npy"
        if not f.exists(): print("skip", n); continue
        z = np.load(f); meta = json.load(open(PROC/f"{n}.json"))
        o, r = extract(n, z, meta, "earth", s["coastal"]); X += o; R += r
        print(f"{n:14s} {len(o):6d} patches")
    X = np.stack(X)
    df = pd.DataFrame(R, columns=["body","site","row","col","x","y"])
    df["role"] = df.site.map(lambda n: SITES[n]["role"] if n in SITES else "moon")
    df["dist_km"] = np.where(df.body=="earth", np.hypot(df.x, df.y)/1000, np.nan)
    df["zone_km"] = df.site.map(lambda n: SITES[n]["r"] if n in SITES else 0)
    df["in_zone"] = (df.body=="earth") & (df.zone_km > 0) & (df.dist_km <= df.zone_km)
    np.save(OUT/"patches.npy", X); df.to_csv(OUT/"patches_meta.csv", index=False)
    print("TOTAL", X.shape, df.groupby("body").size().to_dict(), "in-zone:", int(df.in_zone.sum()))