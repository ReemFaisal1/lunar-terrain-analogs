import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.graph import MCP_Geometric

PROC, PAT = Path("data/processed"), Path("data/patches")
P, RES = 64, 80.0
RADIUS = {"moon": 1737400.0, "earth": 6371000.0}
SUN_ELEV_DEG = 1.5
N_AZ, K_STEPS = 16, 120          # ray directions, ray length in pixels (9.6 km)
IMPASSABLE_DEG = 20.0
N_PAIRS = 12
SAMPLE_PER_BODY = 1500
EARTH_SIGMA = 1.5

def slope_deg(a, res=RES):
    gy, gx = np.gradient(a.astype("float64"), res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def basic_descriptors(patch):
    a = patch.astype("float32")
    s = slope_deg(a)
    rough = np.sqrt(np.mean((a - uniform_filter(a, 5)) ** 2))
    return dict(slope_p50=np.percentile(s, 50), slope_p90=np.percentile(s, 90),
                slope_p99=np.percentile(s, 99), slope_mean=s.mean(), rough_m=rough,
                relief_m=np.percentile(a, 95) - np.percentile(a, 5),
                steep_frac=(s > IMPASSABLE_DEG).mean())

def shadow_descriptors(ctx, body, sun_deg=SUN_ELEV_DEG, n_az=N_AZ, K=K_STEPS):
    """ctx: (P+2K)x(P+2K) elevation window around the patch. A pixel is shadowed along an
    azimuth if terrain on that ray subtends a higher angle than the sun (with curvature drop)."""
    R = RADIUS[body]
    z = np.where(np.isnan(ctx), np.nanmean(ctx), ctx).astype("float32")
    if body == "earth" and EARTH_SIGMA > 0:
      z = gaussian_filter(z, EARTH_SIGMA, mode="nearest")
    zp = np.pad(z, K, constant_values=-1e9)
    z0 = z[K:K + P, K:K + P]
    tan_sun = np.tan(np.radians(sun_deg))
    hits = np.zeros((n_az, P, P), dtype=bool)
    for i, az in enumerate(np.linspace(0, 2 * np.pi, n_az, endpoint=False)):
        dy, dx = -np.cos(az), np.sin(az)
        best = np.full((P, P), -np.inf, dtype="float32")
        seen = set()
        for k in range(1, K + 1):
            oy, ox = int(round(k * dy)), int(round(k * dx))
            if (oy, ox) in seen: continue
            seen.add((oy, ox))
            d = np.hypot(oy, ox) * RES
            zt = zp[2 * K + oy:2 * K + oy + P, 2 * K + ox:2 * K + ox + P]
            np.maximum(best, (zt - z0 - d * d / (2 * R)) / d, out=best)
        hits[i] = best > tan_sun
    frac = hits.mean(axis=0)
    return dict(shadow_mean=float(frac.mean()), shadow_always=float((frac > 0.999).mean()),
                shadow_often=float((frac > 0.5).mean()))

def route_descriptors(patch, rng, n_pairs=N_PAIRS):
    s = slope_deg(patch)
    cost = 1.0 + (s / 10.0) ** 2
    cost[s > IMPASSABLE_DEG] = np.inf
    mcp = MCP_Geometric(cost, fully_connected=True)
    ratios, fails = [], 0
    for _ in range(n_pairs):
        a = rng.integers(2, P - 2, 2); b = rng.integers(2, P - 2, 2)
        dist = np.hypot(*(a - b))
        if dist < 25 or np.isinf(cost[tuple(a)]) or np.isinf(cost[tuple(b)]): continue
        cum, _ = mcp.find_costs([tuple(a)])
        c = cum[tuple(b)]
        if np.isfinite(c): ratios.append(c / dist)
        else: fails += 1
    n = len(ratios) + fails
    return dict(route_ratio=float(np.mean(ratios)) if ratios else np.nan,
                route_fail=(fails / n) if n else np.nan, route_n=n)

def context_window(grid, r, c, K=K_STEPS):
    H, W = grid.shape
    r0, r1, c0, c1 = r - K, r + P + K, c - K, c + P + K
    out = np.full((r1 - r0, c1 - c0), np.nan, dtype="float32")
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)
    out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = grid[rr0:rr1, cc0:cc1]
    return out

if __name__ == "__main__":
    X = np.load(PAT / "patches.npy", mmap_mode="r")
    meta = pd.read_csv(PAT / "patches_meta.csv")
    rng = np.random.default_rng(0)
    print("basic descriptors for", len(meta), "patches ...")
    t = time.time()
    basic = pd.DataFrame([basic_descriptors(np.asarray(X[i])) for i in range(len(meta))])
    print(f"  done in {time.time() - t:.0f}s")
    sample = np.concatenate([rng.choice(meta.index[meta.body == b],
                             min(SAMPLE_PER_BODY, (meta.body == b).sum()), replace=False)
                             for b in ("moon", "earth")])
    extra, grids, t = {i: {} for i in sample}, {}, time.time()
    for n, i in enumerate(sample):
        row = meta.loc[i]
        g = grids.setdefault(row.site, np.load(PROC / f"{row.site}.npy", mmap_mode="r"))
        ctx = context_window(g, int(row.row), int(row.col))
        extra[i].update(shadow_descriptors(ctx, row.body))
        extra[i].update(route_descriptors(np.asarray(X[i]).astype("float32"), rng))
        if n % 250 == 0: print(f"  shadow+route {n}/{len(sample)} ({time.time() - t:.0f}s)")
    ex = pd.DataFrame.from_dict(extra, orient="index")
    out = pd.concat([basic, ex.reindex(meta.index)], axis=1)
    out.to_csv(PAT / "descriptors.csv", index=False)
    out["body"] = meta.body
    print(out.groupby("body")[["slope_p50", "slope_p90", "rough_m", "relief_m",
          "shadow_mean", "shadow_always", "route_ratio", "route_fail"]].median().round(3).to_string())
    print("saved", PAT / "descriptors.csv")