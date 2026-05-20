import torch
import torch.nn as nn

SIGMA_DATA = 0.5
P_MEAN = -1.2
P_STD = 1.2
SIGMA_MIN =0.002
SIGMA_MAX = 80.0

class EDMPrecond(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(self, x_noisy, sigma):
        # sigma shape: (B,) — broadcast to (B,1,1,1)
        s = sigma[:, None, None, None]
        
        c_skip = SIGMA_DATA**2 / (s**2 + SIGMA_DATA**2)
        c_out = s * SIGMA_DATA / (s**2 + SIGMA_DATA**2).sqrt()
        c_in =1.0 / (s**2 + SIGMA_DATA**2).sqrt()
        c_noise  = sigma.log() / 4.0
        
        F_x = self.unet(c_in * x_noisy, c_noise)
        return c_skip*x_noisy + c_out * F_x

def edm_loss(precond, x0, rng=None):
    device = x0.device
    B = x0.shape[0]
    
    # Sample log-normal sigma for each item in the batch                                                                                                                                      
    log_sigma = torch.randn(B, device=device) * P_STD + P_MEAN
    sigma = log_sigma.exp() # shape (B,) 
    
    noise = torch.randn_like(x0)
    x_noisy = x0 +sigma[:, None, None, None] * noise
    
    # Effective weight from the paper (lambda)                                                                                                                                                
    weight = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA)**2
    
    D_x = precond(x_noisy, sigma)
    loss = weight[:, None,None,None] * (D_x - x0)**2
    return loss.mean()