import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import argparse
from tqdm import tqdm

from model.unet import UNet

def linear_schedule(T, beta_start, beta_end, device):
    betas = torch.linspace(beta_start,beta_end, T, device=device)
    alphas_bar = torch.cumprod(1-betas, dim=0)
    return betas, alphas_bar

def q_samle(x0, t, alphas_bar,  noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alphas_bar[t][:, None, None, None]
    return ab.sqrt() * x0 + (1-ab).sqrt() * noise, noise
def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])
    
    dataset = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers= 4, pin_memory=True, drop_last=True)
    
    betas, alphas_bar = linear_schedule(cfg["T"], cfg["beta_start"], cfg["beta_end"], device)
    
    model = UNet(base_ch=cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    ckpt_dir = Path("checkpoints/mnist")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 1
    last_ckpt= sorted(ckpt_dir.glob("epoch_*.pt"))
    
    if last_ckpt:
        ckpt = torch.load(last_ckpt[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"]+1
        print (f"Resumed from epoch {ckpt['epoch']}")
        
    for epoch in range (start_epoch, cfg["epochs"]+1):
        model.train()
        total_loss = 0.0
        for x, _ in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            x = x.to(device)
            t = torch.randint(0, cfg["T"], (x.size(0),), device=device)
            xt, noise = q_samle(x, t, alphas_bar)
            
            loss = F.mse_loss(model(xt,t), noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")
        
        if epoch % cfg["checkpoint_every"] == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                          "opt": opt.state_dict(), "cfg": cfg},
                         ckpt_dir / f"epoch_{epoch:04d}.pt")
            print(f"  checkpoint saved")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mnist.yaml")
    main(p.parse_args().config)