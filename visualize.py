import torch                                                                             
import argparse
import numpy as np                                                                       
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.unet import UNet                                                              
from model.edm import EDMPrecond, SIGMA_MIN, SIGMA_MAX
                                                                                         
@torch.no_grad()
def heun_sample_one(model, img_size, device, steps=35, rho=7.0):                         
    step_indices = torch.arange(steps, device=device)                                    
    t_steps = (                                                                          
        SIGMA_MAX ** (1/rho)                                                             
        + step_indices / (steps - 1) * (SIGMA_MIN**(1/rho) - SIGMA_MAX**(1/rho))         
    ) ** rho                                                                             
    sigmas = torch.cat([t_steps, torch.zeros(1, device=device)])                         
                                                                                         
    x = torch.randn(1, 1, img_size, img_size, device=device) * sigmas[0]                 
                                                                                         
    for i in tqdm(range(steps), desc="Sampling"):                                        
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]                                                       
        s_vec = sigma.expand(1)
        d = (x - model(x, s_vec)) / sigma                                                
        x_next = x + (sigma_next - sigma) * d                                            
        if sigma_next > 0:                                                               
            s_next_vec = sigma_next.expand(1)                                            
            d_next = (x_next - model(x_next, s_next_vec)) / sigma_next                   
            x_next = x + (sigma_next - sigma) * 0.5 * (d + d_next)
        x = x_next                                                                       
                
    return x[0, 0].cpu().numpy()                                                         
                
def main(ckpt_path):                                                                     
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]                                                                    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                                                         
    unet = UNet(base_ch=cfg["base_channels"], ch_mults=tuple(cfg["channel_mults"]))
    model = EDMPrecond(unet).to(device)                                                  
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))                                                 
    model.eval()
                                                                                         
    heightmap = heun_sample_one(model, cfg["image_size"], device)                        
 
    x = np.arange(heightmap.shape[1])                                                    
    y = np.arange(heightmap.shape[0])
    x, y = np.meshgrid(x, y)
                                                                                         
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")                                           
    ax.plot_surface(x, y, heightmap, cmap="terrain", linewidth=0, antialiased=True)
    ax.set_axis_off()                                                                    
    plt.tight_layout()
    plt.show()                                                                           
                
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")                                                               
    main(p.parse_args().ckpt)