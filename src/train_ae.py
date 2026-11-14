import time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PAT = Path("data/patches"); Path("results/quicklook").mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)
SCALE, Z, STEPS, BS = 50.0, 64, 4000, 64          # BS per body -> 128 per step

X = np.load(PAT / "patches.npy").astype("float32")
X = np.arcsinh(X / SCALE).astype("float16")
Xt = torch.from_numpy(X)
meta = pd.read_csv(PAT / "patches_meta.csv")

# spatial block split (~20 km blocks), 20% of blocks held out
blk = meta.site + "_" + (meta.row // 256).astype(str) + "_" + (meta.col // 256).astype(str)
ub = blk.unique()
val_blocks = set(rng.choice(ub, int(0.2 * len(ub)), replace=False))
is_val = blk.isin(val_blocks).values
moon = (meta.body == "moon").values
tr_m, tr_e = np.where(~is_val & moon)[0], np.where(~is_val & ~moon)[0]
va_m, va_e = np.where(is_val & moon)[0], np.where(is_val & ~moon)[0]
print(f"train moon {len(tr_m)} earth {len(tr_e)} | val moon {len(va_m)} earth {len(va_e)} | device {dev}")

def blockc(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
def blockd(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

class AE(nn.Module):
    def __init__(s, z=Z):
        super().__init__()
        s.enc = nn.Sequential(blockc(1, 32), blockc(32, 64), blockc(64, 128), blockc(128, 256),
                              nn.Flatten(), nn.Linear(256 * 16, z))
        s.fc = nn.Linear(z, 256 * 16)
        s.dec = nn.Sequential(blockd(256, 128), blockd(128, 64), blockd(64, 32),
                              nn.ConvTranspose2d(32, 1, 4, 2, 1))
    def encode(s, x): return s.enc(x)
    def forward(s, x):
        z = s.enc(x); return s.dec(s.fc(z).view(-1, 256, 4, 4)), z

def loss_fn(a, b):
    gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return F.mse_loss(a, b) + 10 * (F.mse_loss(gx(a), gx(b)) + F.mse_loss(gy(a), gy(b)))

def batch(idx): return Xt[idx].float().unsqueeze(1).to(dev)
def aug(x):
    x = torch.rot90(x, int(rng.integers(4)), (2, 3))
    return torch.flip(x, (3,)) if rng.random() < 0.5 else x

model = AE().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# fixed validation subsets, so the validation curve is comparable across steps
vm = rng.choice(va_m, 1000, replace=False)
ve = rng.choice(va_e, 1000, replace=False)
hist = {"step": [], "tr_m": [], "tr_e": []}
val_hist = {"step": [], "va_m": [], "va_e": []}
t0 = time.time()
for step in range(1, STEPS + 1):
    if step == int(0.7 * STEPS):
        for g in opt.param_groups: g["lr"] = 3e-4
    idx = np.concatenate([rng.choice(tr_m, BS), rng.choice(tr_e, BS)])
    x = aug(batch(idx))
    rec, _ = model(x)
    loss = loss_fn(rec, x)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                       # train loss split by body (first half Moon, second half Earth)
        hist["step"].append(step)
        hist["tr_m"].append(loss_fn(rec[:BS], x[:BS]).item())
        hist["tr_e"].append(loss_fn(rec[BS:], x[BS:]).item())
    if step % 50 == 0:
        model.eval()
        with torch.no_grad():
            lm = loss_fn(model(batch(vm))[0], batch(vm)).item()
            le = loss_fn(model(batch(ve))[0], batch(ve)).item()
        model.train()
        val_hist["step"].append(step); val_hist["va_m"].append(lm); val_hist["va_e"].append(le)
        if step % 250 == 0:
            print(f"step {step:5d} train {loss.item():.4f} | val moon {lm:.4f} earth {le:.4f} | {time.time()-t0:.0f}s")

# loss curves
pd.DataFrame(hist).to_csv("results/train_loss.csv", index=False)
pd.DataFrame(val_hist).to_csv("results/val_loss.csv", index=False)
def smooth(v, w=50): return np.convolve(np.asarray(v), np.ones(w) / w, mode="valid")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for a, name, tr, va in [(ax[0], "Moon", "tr_m", "va_m"), (ax[1], "Earth", "tr_e", "va_e")]:
    s, t = np.array(hist["step"]), np.array(hist[tr])
    a.plot(s, t, color="C0", alpha=0.2)
    a.plot(s[49:], smooth(t), color="C0", label="train (50-step average)")
    a.plot(val_hist["step"], val_hist[va], color="C3", label="validation (held-out regions)")
    a.set_yscale("log"); a.set_title(f"{name} patches"); a.set_xlabel("training step")
    a.grid(alpha=0.3); a.legend()
ax[0].set_ylabel("loss (MSE + gradient term, log scale)")
plt.tight_layout(); plt.savefig("results/quicklook/loss_curves.png", dpi=120); plt.close()


# embeddings: average over the 8 rotations/flips
model.eval(); N = len(Xt); E = np.zeros((N, Z), dtype="float32")
with torch.no_grad():
    for s in range(0, N, 1024):
        x = batch(np.arange(s, min(s + 1024, N))); acc = 0
        for k in range(4):
            for f in (False, True):
                v = torch.rot90(x, k, (2, 3)); v = torch.flip(v, (3,)) if f else v
                acc = acc + model.encode(v)
        E[s:s + 1024] = (acc / 8).cpu().numpy()
np.save(PAT / "embeddings.npy", E); torch.save(model.state_dict(), "results/ae.pt")
np.save(PAT / "is_val.npy", is_val)

# reconstruction check
sel = list(rng.choice(va_m, 4)) + list(rng.choice(va_e, 4))
with torch.no_grad(): rec = model(batch(sel))[0].cpu().numpy()[:, 0]
fig, ax = plt.subplots(2, 8, figsize=(18, 5))
for j, i in enumerate(sel):
    ax[0, j].imshow(Xt[i].float().numpy(), cmap="terrain"); ax[0, j].set_title(meta.site[i][:12], fontsize=8)
    ax[1, j].imshow(rec[j], cmap="terrain"); ax[0, j].axis("off"); ax[1, j].axis("off")
plt.savefig("results/quicklook/recon.png", dpi=100, bbox_inches="tight")
print("saved embeddings", E.shape)