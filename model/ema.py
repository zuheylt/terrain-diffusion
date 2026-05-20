import copy
import torch

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
            
    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha = 1 - self.decay)
    
    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
    
    