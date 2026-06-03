import os
import yaml
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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

    # --- Distributed setup -------------------------------------------------
    # torchrun sets RANK / LOCAL_RANK / WORLD_SIZE. If they're absent we fall
    # back to a normal single-process run (e.g. local debugging).
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1

    is_main = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if is_main:
        print(f"device: {device} | world_size: {world_size}")

    # --- Data --------------------------------------------------------------
    dataset = HeightmapDataset()
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                            num_workers=2, pin_memory=True, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,num_workers=4, pin_memory=True, drop_last=True)

    # --- Model / EMA / optimizer ------------------------------------------
    unet = UNet(base_ch=cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"]),
                use_checkpoint=cfg.get("use_checkpoint", False))
    model = EDMPrecond(unet).to(device)
    ema = EMA(model, decay=0.9999)            # tracks the raw (unwrapped) module
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()
    if is_main:
        print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ckpt_dir = Path(f"checkpoints/heightmap_{cfg['image_size']}_edm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Download from HF if no local checkpoint (cross-session resume) ---
    if not any(ckpt_dir.glob("epoch_*.pt")) and os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN_REPO"):
        if is_main:
            try:
                from huggingface_hub import hf_hub_download
                print("No local checkpoint — downloading latest.pt from HF...")
                hf_hub_download(
                    repo_id=os.environ["HF_TOKEN_REPO"],
                    filename="latest.pt",
                    token=os.environ["HF_TOKEN"],
                    local_dir=str(ckpt_dir),
                    local_dir_use_symlinks=False,
                )
                probe = torch.load(ckpt_dir / "latest.pt", map_location="cpu")
                renamed = ckpt_dir / f"epoch_{probe['epoch']:04d}.pt"
                (ckpt_dir / "latest.pt").rename(renamed)
                print(f"  saved as {renamed.name}")
            except Exception as e:
                print(f"  HF download failed (starting fresh): {e}")
        if ddp:
            dist.barrier()   # rank 1 waits here until rank 0 finishes the download

    # --- Resume (every rank loads the same state) -------------------------
    start_epoch = 1
    last_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))
    if last_ckpt:
        ckpt = torch.load(last_ckpt[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}")

    # Wrap AFTER loading so DDP broadcasts the loaded weights to every rank.
    train_model = DDP(model, device_ids=[local_rank]) if ddp else model

    # --- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_model.train()
        if ddp:
            sampler.set_epoch(epoch)          # different shuffle each epoch
        total_loss = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}", disable=not is_main):
            x = x.to(device)
            with autocast("cuda"):
                loss = edm_loss(train_model, x)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)                 # update EMA from the raw module
            total_loss += loss.item()
        if is_main:
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

        # Only rank 0 writes checkpoints and uploads to the Hub.
        if is_main and epoch % cfg["checkpoint_every"] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            print(f"  checkpoint saved")
            if os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN_REPO"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    # Upload milestone (every 10 epochs) and always overwrite latest
                    if epoch % 10 == 0:
                        api.upload_file(
                            path_or_fileobj=str(ckpt_path),
                            path_in_repo=ckpt_path.name,
                            repo_id=os.environ["HF_TOKEN_REPO"],
                            token=os.environ["HF_TOKEN"],
                            )
                    api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo="latest.pt",
                        repo_id=os.environ["HF_TOKEN_REPO"],
                        token=os.environ["HF_TOKEN"],
                    )
                    print(f"  pushed to Hub")
                    # Delete local file to free disk (latest.pt is on HF)
                    ckpt_path.unlink()
                    print(f"  local checkpoint removed")
                except Exception as e:
                    print(f"  HF upload failed (continuing): {e}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/heightmap_64.yaml")
    main(p.parse_args().config)