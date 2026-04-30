import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model.unet import UNet

@torch.no_grad()
def ddpm_sample(model, betas, alphas_bar, T, n, img_size, device):
    alphas = 1 - betas
    x= torch.randn(n, 1, img_size, img_size, device=device)
    
    for t_idx in tqdm(reversed(range(T)), total=T, desc="Sampling"):
        t = torch.full((n,), t_idx, device=device, dtype=torch.long)
        eps = model(x,t)
        
        ab_t = alphas_bar[t_idx]
        x0_pred = ((x - (1-ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1,1)
        
        if t_idx > 0:
            ab_prev = alphas_bar[t_idx-1]
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            mean = (
                (ab_prev.sqrt() * beta_t) / (1-ab_t) * x0_pred
                + (alpha_t.sqrt() * (1-ab_prev)) / (1-ab_t) * x
            )
            var = beta_t * (1 - ab_prev) / (1 - ab_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0_pred
    return x

def main(ckpt_path, n = 16):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(base_ch = cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    betas = torch.linspace(cfg["beta_start"], cfg["beta_end"],cfg["T"], device=device)
    alphas_bar = torch.cumprod(1-betas, dim=0)
    
    samples = ddpm_sample(model, betas, alphas_bar, cfg["T"], n, cfg["image_size"],device)
    samples = (samples *0.5 +0.5).clamp(0,1).cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i,0], cmap = "gray")
        ax.axis("off")
    plt.tight_layout()
    out = Path(ckpt_path).with_suffix(".png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args() 
    main(args.ckpt,args.n)   