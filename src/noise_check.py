import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

X = np.load("data/patches/patches.npy", mmap_mode="r")
meta = pd.read_csv("data/patches/patches_meta.csv")
rng = np.random.default_rng(0)

def hf_ratio(a, pre=0.0):             # pixel-scale energy relative to 1-4 px energy
    a = a.astype("float32")
    if pre > 0: a = gaussian_filter(a, pre)
    g1, g4 = gaussian_filter(a, 1.0), gaussian_filter(a, 4.0)
    return np.std(a - g1) / (np.std(g1 - g4) + 1e-6)

mi = rng.choice(meta.index[meta.body == "moon"], 2000, replace=False)
ei = rng.choice(meta.index[meta.body == "earth"], 2000, replace=False)
q = lambda v: np.round(np.percentile(v, [25, 50, 75]), 3)
print("moon  (25/50/75%):", q([hf_ratio(X[i]) for i in mi]))
for s in (0, 0.75, 1.0, 1.5, 2.0):
    print(f"earth smoothed sigma={s}:", q([hf_ratio(X[i], s) for i in ei]))