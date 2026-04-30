import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half-1)
        )
    args = t[:,None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim =1)

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch,3, padding=1)
        self.time_proj = nn.Linear(time_dim,out_ch)
        self.norm2 = nn.GroupNorm(8,out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch,1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb):
        h= self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:,:, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
class UNet(nn.Module):
    
    """tiny Unet for MNIST DDPM INPUT: (B,1,32,32). Output: (B,1,32,32)."""
    def __init__(self,base_ch=64, ch_mults=(1,2)):
        super().__init__()
        self.emb_dim = base_ch
        time_dim = base_ch * 4
        chs = [base_ch*m for m in ch_mults]
        
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim),
        )
        
        #encoder
        self.input_conv = nn.Conv2d(1,chs[0], 3, padding=1)
        self.down1 =  ResBlock(chs[0],chs[0],time_dim)
        self.pool1 = nn.Conv2d(chs[0], chs[0], 3, stride=2, padding=1)
        self.down2 = ResBlock(chs[0],chs[1],time_dim)
        self.pool2 = nn.Conv2d(chs[1],chs[1], 3,stride=2, padding=1)
        
        #Bottleneck
        self.mid = ResBlock(chs[1], chs[1], time_dim)
        
        #Decoder
        self.up2 = nn.ConvTranspose2d(chs[1], chs[1], 2, stride=2)
        self.dec2 = ResBlock(chs[1] + chs[1], chs[0], time_dim)                          
        self.up1 = nn.ConvTranspose2d(chs[0], chs[0], 2, stride=2)       # 16→32
        self.dec1 = ResBlock(chs[0] + chs[0], chs[0], time_dim)
        
        self.output_conv = nn.Conv2d(chs[0],1, 1)
    def forward(self, x, t):
          t_emb = sinusoidal_embedding(t.float(), self.emb_dim)
          t_emb = self.time_mlp(t_emb)                                                     
  
          x0 = self.input_conv(x)                                                          
          h1 = self.down1(x0, t_emb)
          h2 = self.down2(self.pool1(h1), t_emb)                                           
                                                                                           
          h = self.mid(self.pool2(h2), t_emb)
                                                                                           
          h = self.dec2(torch.cat([self.up2(h), h2], dim=1), t_emb)                        
          h = self.dec1(torch.cat([self.up1(h), h1], dim=1), t_emb)
                                                                                           
          return self.output_conv(h)