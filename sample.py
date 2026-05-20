import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model.unet import UNet
from model.edm import EDMPrecond, SIGMA_MIN, SIGMA_MAX

@torch.no_grad()
def heun_sample(model, n, img_size, device, steps=35, rho=7.0):
    # Build sigma schedule (Karras et al. eq. 5)                                                                                                                                              
    step_indices = torch.arange(steps, device=device)                                                                                                                                         
    t_steps = (                                                                                                                                                                               
        SIGMA_MAX ** (1/rho)                                                                                                                                                                  
        + step_indices / (steps - 1) * (SIGMA_MIN**(1/rho) - SIGMA_MAX**(1/rho))
    ) ** rho                                                                                                                                                                                  
    # Append sigma=0 at the end                                                                                                                                                               
    sigmas = torch.cat([t_steps, torch.zeros(1, device=device)])                                                                                                                              
                                                                                                                                                                                              
    x = torch.randn(n, 1, img_size, img_size, device=device) * sigmas[0]                                                                                                                      
                                                                                                                                                                                              
    for i in tqdm(range(steps), desc="Sampling"):                                                                                                                                             
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
                                                                                                                                                                                              
        # Derivative at current point
        s_vec = sigma.expand(n)                                                                                                                                                               
        d = (x - model(x, s_vec)) / sigma                                                                                                                                                     
                                                                                                                                                                                              
        # Euler step (predictor)                                                                                                                                                              
        x_next = x + (sigma_next - sigma) * d                                                                                                                                                 
                                                                                                                                                                                              
        if sigma_next > 0:
            # Derivative at predicted point (corrector)                                                                                                                                       
            s_next_vec = sigma_next.expand(n)
            d_next = (x_next - model(x_next, s_next_vec)) / sigma_next                                                                                                                        
            # Heun correction: average the two derivatives
            x_next = x + (sigma_next - sigma) * 0.5 * (d + d_next)                                                                                                                            
                                                                                                                                                                                              
        x = x_next                                                                                                                                                                            
                                                                                                                                                                                              
    return x

def main(ckpt_path, n=16, steps=35):                                                                                                                                                          
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]                                                                                                                                                                         
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                              
    unet = UNet(base_ch=cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"]))                                                                                                           
    model = EDMPrecond(unet).to(device)                                                                                                                                                       
    model.load_state_dict(ckpt.get("ema",ckpt["model"]))                                                                                                                                                      
    model.eval()
    samples = heun_sample(model, n, cfg["image_size"], device, steps=steps)                                                                                                                   
    samples = (samples * 0.5 + 0.5).clamp(0, 1).cpu()
                                                                                                                                                                                              
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))                                                                                                                                            
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")                                                                                                                                                 
        ax.axis("off")                                                                                                                                                                        
    plt.tight_layout()
    out = Path(ckpt_path).with_suffix(".png")                                                                                                                                                 
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
                                                                                                                                                                                              
if __name__ == "__main__":
    p = argparse.ArgumentParser()                                                                                                                                                             
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--steps", type=int, default=35)                                                                                                                                           
    args = p.parse_args()
    main(args.ckpt, args.n, args.steps)