import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = np.load("data/patches/patches.npy"); df = pd.read_csv("data/patches/patches_meta.csv")
rng = np.random.default_rng(0)
idx = list(rng.choice(df.index[df.body=="moon"], 6, replace=False)) + \
      list(rng.choice(df.index[df.body=="earth"], 6, replace=False))
fig, ax = plt.subplots(2, 6, figsize=(16, 6))
for a, i in zip(ax.ravel(), idx):
    a.imshow(X[i].astype("float32"), cmap="terrain", vmin=-300, vmax=300)
    a.set_title(df.site[i][:12], fontsize=8); a.axis("off")
plt.savefig("results/quicklook/patch_samples.png", dpi=110, bbox_inches="tight")