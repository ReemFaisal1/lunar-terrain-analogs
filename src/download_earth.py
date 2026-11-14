import os, earthaccess
from sites import SITES
earthaccess.login()
for name, s in SITES.items():
    out = f"data/raw/earth/{name}"
    if s["pad"] is None and os.path.isdir(out) and os.listdir(out):
        print("have", name); continue
    os.makedirs(out, exist_ok=True)
    p = s["pad"] or 0.1
    res = earthaccess.search_data(short_name=s["prod"],
          bounding_box=(s["lon"]-p, s["lat"]-p, s["lon"]+p, s["lat"]+p))
    print(name, s["prod"], len(res), "granule(s)")
    if res: earthaccess.download(res, out)