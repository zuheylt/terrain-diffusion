import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)

class ResBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
    
class SelfAttention(nn.Module):
    
    def __init__(self, ch, heads=8):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.head_dim = ch // heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, H * W, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, H * W, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H * W, self.heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, H * W, C)
        h = self.proj(h).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + h
    
class UNet(nn.Module):
    
    """UNet for EDM terrain diffusion. Input: (B,1,H,W). Output: (B,1,H,W)."""
    def __init__(self, base_ch=64, ch_mults=(1, 2, 4, 8), use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.emb_dim = base_ch
        time_dim = base_ch * 4
        chs = [base_ch * m for m in ch_mults]
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        self.input_conv = nn.Conv2d(1, chs[0], 3, padding=1)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = chs[0]
        skip_chs = []
        for out_ch in chs:
            self.downs.append(ResBlock(in_ch, out_ch, time_dim))
            self.pools.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            skip_chs.append(out_ch)
            in_ch = out_ch
            
        # Bottleneck — two ResBlocks sandwiching attention at lowest resolution
        self.mid1 = ResBlock(chs[-1], chs[-1], time_dim)
        self.attn = SelfAttention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], time_dim)
        
        # Decoder
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_ch = chs[-1]
        for out_ch, skip_ch in zip(reversed(chs), reversed(skip_chs)):
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2))
            self.decs.append(ResBlock(in_ch + skip_ch, out_ch, time_dim))
            in_ch = out_ch
            
        self.output_conv = nn.Conv2d(chs[0], 1, 1)
        
    def _run(self, block, x, t_emb):
        if self.use_checkpoint:
            return cp.checkpoint(block, x, t_emb, use_reentrant=False)
        return block(x, t_emb)
    
    def forward(self, x, noise_cond):
        t_emb = sinusoidal_embedding(noise_cond, self.emb_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.input_conv(x)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = self._run(down, x, t_emb)
            skips.append(x)
            x = pool(x)
        x = self._run(self.mid1, x, t_emb)
        x = self.attn(x)
        x = self._run(self.mid2, x, t_emb)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            x = torch.cat([up(x), skip], dim=1)
            x = self._run(dec, x, t_emb)
        return self.output_conv(x)