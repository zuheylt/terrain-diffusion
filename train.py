import yaml
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from data.dataset import HeightmapDataset
from pathlib import Path
import argparse
from tqdm import tqdm
from model.unet import UNet
from model.edm import EDMPrecond, edm_loss
from model.ema import EMA
def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    dataset = HeightmapDataset()
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    unet = UNet(base_ch=cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"]),
                use_checkpoint=cfg.get("use_checkpoint", False))
    model = EDMPrecond(unet).to(device)
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    ckpt_dir = Path("checkpoints/heightmap_64_edm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    last_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))
    if last_ckpt:
        ckpt = torch.load(last_ckpt[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            x = x.to(device)
            with autocast("cuda"):
                loss = edm_loss(model, x)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
        if epoch % cfg["checkpoint_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")
            print(f"  checkpoint saved")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/heightmap_64.yaml")
    main(p.parse_args().config)