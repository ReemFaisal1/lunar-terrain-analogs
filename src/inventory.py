from pathlib import Path
root = Path("data/raw")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(root)}  {p.stat().st_size/1e6:.1f} MB")