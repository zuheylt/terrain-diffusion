import json
import numpy as np
import rasterio
from pathlib import Path
RAW_DIR = Path("data/raw")
TILE_DIR = Path("data/tiles")
TILE_SIZE = 256
STRIDE = 128

def process_file(tif_path, biome, manifest):
    
    with rasterio.open(tif_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    h, w = dem.shape
    count = 0
    for row in range(0, h - TILE_SIZE + 1, STRIDE):
        for col in range(0, w - TILE_SIZE + 1, STRIDE):
            patch = dem[row:row + TILE_SIZE, col:col + TILE_SIZE]
            patch_min = float(patch.min())
            patch_max = float(patch.max())
            if np.isnan(patch).any() or patch_max == patch_min:
                continue
            normalized = (patch - patch_min) / (patch_max - patch_min)
            normalized = normalized * 2 - 1
            tile_id = f"{biome}_{tif_path.stem}_{row}_{col}"
            np.save(TILE_DIR / f"{tile_id}.npy", normalized.astype(np.float32))
            manifest.append({
                "id": tile_id,
                "source": tif_path.name,
                "biome": biome,
                "min_elev": patch_min,
                "max_elev": patch_max,
            })
            count += 1
    return count
def main():
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    biome_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not biome_dirs:
        print("No biome subdirectories found in data/raw/")
        return
    for biome_dir in biome_dirs:
        biome = biome_dir.name
        tif_files = sorted(biome_dir.glob("*.tif"))
        if not tif_files:
            print(f"  {biome}: no .tif files, skipping")
            continue
        print(f"\n--- {biome} ({len(tif_files)} files) ---")
        total = 0
        for tif_path in tif_files:
            n = process_file(tif_path, biome, manifest)
            print(f"  {tif_path.name} → {n} tiles")
            total += n
        print(f"  {biome} total: {total} tiles")
    with open(TILE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {len(manifest)} tiles saved to {TILE_DIR}")
if __name__ == "__main__":
    main()