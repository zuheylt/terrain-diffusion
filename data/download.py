import urllib.request
from pathlib import Path
RAW_DIR = Path("data/raw")
BIOME_TILES = {
    "alpine": [
        (46, 7), (46, 8), (46, 9), (47, 7), (47, 8),
        (39, -106), (40, -106), (40, -105),
        (27, 86), (27, 87), (28, 86),
        (-33, -70), (-32, -70),
    ],
    "dunes": [
        (30, -5), (31, -4), (29, -5),
        (-24, 15), (-25, 15),
        (20, 50), (21, 51),
        (40, 93), (41, 94),
    ],
    "coastal": [
        (62, 5), (63, 6), (62, 6),
        (53, -9), (54, -9),
        (44, -124), (45, -124),
        (40, 14),
    ],
    "fluvial": [
        (-3, -60), (-4, -60),
        (29, -89), (29, -90),
        (25, 85), (25, 86),
    ],
    "glacial": [
        (68, -50), (69, -50),
        (-49, -73), (-50, -73),
        (64, -18), (65, -19),
    ],
    "volcanic": [
        (64, -17), (63, -19),
        (19, -155), (20, -155),
        (37, 15),
    ],
    "karst": [
        (25, 110), (24, 110),
        (20, -89), (20, -90),
        (45, 14),
    ],
    "plains": [
        (38, -98), (39, -98), (39, -99),
        (48, 34), (49, 34),
        (-35, -63), (-34, -63),
    ],
    "archipelagos": [
        (37, 25), (36, 25), (37, 26),
        (11, 124), (10, 124),
        (18, -67), (18, -66),
    ],
    "canyon": [
        (36, -112), (36, -113),
        (37, -112),
        (-15, -72),
        (38, 100),
    ],
    "tundra": [
        (70, -153), (71, -153),
        (68, 130), (69, 130),
        (78, 16), (78, 17),
    ],
    "badlands": [
        (43, -102), (44, -102),
        (-47, -68), (-46, -68),
        (36, -108),
    ],
    "fjord": [
        (61, 6), (61, 7), (62, 7),
        (-50, -74), (-51, -74),
        (-44, 168),
    ],
    "savanna": [
        (-2, 34), (-3, 35),
        (-24, 31), (-25, 31),
        (-15, -47),
    ],
    "mesa": [
        (37, -110), (36, -110), (37, -111),
        (-33, 22),
    ],
    "rift_valley": [
        (0, 36), (1, 36),
        (11, 41), (12, 41),
        (31, 35),
    ],
    "highland_plateau": [
        (32, 88), (33, 88), (31, 88),
        (9, 38), (10, 38),
        (-17, -68), (-16, -68),
    ],
}
def tile_url(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_str = f"{ns}{abs(lat):02d}"
    lon_str = f"{ew}{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
def download_tiles(biome_tiles=BIOME_TILES):
    for biome, tiles in biome_tiles.items():
        biome_dir = RAW_DIR / biome
        biome_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {biome} ({len(tiles)} tiles) ---")
        for lat, lon in tiles:
            url = tile_url(lat, lon)
            filename = url.split("/")[-1]
            dest = biome_dir / filename
            if dest.exists() and dest.stat().st_size > 1_000_000:
                print(f"  skip {filename}")
                continue
            print(f"  downloading {filename} ...")
            try:
              urllib.request.urlretrieve(url, dest)
              print(f"  saved → {dest}")
            except Exception as e:
              dest.unlink(missing_ok=True)
              print(f"  FAILED {filename}: {e}")
if __name__ == "__main__":
    download_tiles()