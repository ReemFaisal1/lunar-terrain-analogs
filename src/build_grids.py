import json, zipfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import reproject, calculate_default_transform, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sites import SITES

RAW, INTERIM, OUT = Path("data/raw"), Path("data/interim"), Path("data/processed")
QL = Path("results/quicklook")
for p in (INTERIM, OUT, QL): p.mkdir(parents=True, exist_ok=True)
RES = 80.0



def find_dem_files(site_dir):
    files = []
    for z in site_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.lower().endswith(("_dem.tif", ".hgt")):
                    dest = INTERIM / site_dir.name
                    zf.extract(n, dest)
                    files.append(dest / n)
    for f in site_dir.iterdir():
        if f.name.lower().endswith(("_dem.tif", ".hgt")):
            files.append(f)
    return sorted(set(files))

def build_earth(name, lon, lat):
    files = find_dem_files(RAW / "earth" / name)
    if not files:
        print(f"!! {name}: no DEM files found"); return
    srcs = [rasterio.open(f) for f in files]
    arr, tr = merge(srcs, nodata=-32768)
    for s in srcs: s.close()
    z = arr[0].astype("float32")
    z[z < -1000] = np.nan                     # voids (-32768, -9999)
    src_crs = CRS.from_epsg(4326)
    proj = f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs"
    dst_crs = CRS.from_proj4(proj)
    h, w = z.shape
    dst_tr, dw, dh = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, tr), resolution=RES)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(z, dst, src_transform=tr, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_tr, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=Resampling.average)
    save(name, dst, {"body":"earth","res":RES,"crs":proj,"transform":list(dst_tr.to_gdal()),
                     "tiles":[f.name for f in files]})

def build_moon():
    with rasterio.open(RAW / "moon" / "LDEM_80S_80MPP_ADJ.tiff") as d:
        print("MOON:", d.shape, d.res, d.dtypes, d.nodata, d.scales, d.offsets)
        z = d.read(1).astype("float32")
        if d.nodata is not None: z[z == d.nodata] = np.nan
        z = z * d.scales[0] + d.offsets[0]
        meta = {"body":"moon","res":float(d.res[0]),"crs":d.crs.to_string(),
                "transform":list(d.transform.to_gdal())}
    save("moon_south_pole", z, meta)

def save(name, z, meta):
    np.save(OUT / f"{name}.npy", z)
    json.dump(meta, open(OUT / f"{name}.json", "w"))
    res = meta["res"]
    zz = np.where(np.isnan(z), np.nanmean(z), z)
    gx, gy = np.gradient(zz, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    ok = ~np.isnan(z)
    print(f"{name:16s} shape={z.shape} valid={ok.mean():.2f} "
          f"elev[min/mean/max]={np.nanmin(z):.0f}/{np.nanmean(z):.0f}/{np.nanmax(z):.0f} m "
          f"slope p50/p90={np.percentile(slope[ok],50):.1f}/{np.percentile(slope[ok],90):.1f} deg")
    step = max(1, max(z.shape)//1200)
    s = zz[::step, ::step]
    gy2, gx2 = np.gradient(s, res*step)
    sl = np.pi/2 - np.arctan(np.hypot(gx2, gy2)); asp = np.arctan2(-gx2, gy2)
    az, alt = np.radians(360-315+90), np.radians(45)
    hs = np.sin(alt)*np.sin(sl) + np.cos(alt)*np.cos(sl)*np.cos((az-np.pi/2)-asp)
    plt.figure(figsize=(7,7)); plt.imshow(hs, cmap="gray"); plt.axis("off")
    plt.title(name); plt.savefig(QL / f"{name}.png", dpi=110, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    if not (OUT / "moon_south_pole.npy").exists():
        build_moon()
    for n, s in SITES.items():
        if (OUT / f"{n}.npy").exists(): continue
        build_earth(n, s["lon"], s["lat"])