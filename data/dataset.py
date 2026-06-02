import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class HeightmapDataset(Dataset):
    def __init__(self, tile_dir="data/tiles"):
        tile_dir = Path(tile_dir)
        with open(tile_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.tile_dir = tile_dir
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        arr = np.load(self.tile_dir / f"{entry['id']}.npy")
        return torch.from_numpy(arr).unsqueeze(0)