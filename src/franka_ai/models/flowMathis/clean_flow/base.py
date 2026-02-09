import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class Flow():
    def __init__(self, model, device="cpu", dim: int = 1):
        super().__init__()
        self.dim = dim
        self.device = device
        self.model = model

    def step(self, x_t: Tensor, c, t_start: Tensor, t_end: Tensor) -> Tensor:
        """
        x_t : (B, dim)
        c : {mod1:(d1,...,dn), mod2: ..., ...}
        t_start : (B, 1)
        """
        return x_t + (t_end - t_start)*self.model(x_t, c, t_start)

    def solve(self, c, n_steps, n_samples=1, return_intermediary_steps=False):
        """
        c : {mod1:(d1,...,dn), mod2: ..., ...}
        n_steps : int
        """
        device, dtype = None, None
        for key in c.keys():
            device, dtype = c[key].device, c[key].dtype

        timesteps = np.linspace(0, 1, n_steps+1)
        current_x = torch.randn(n_samples, self.dim, device=device, dtype=dtype)

        x_s = []

        for i in range(n_steps):
            if return_intermediary_steps:
                x_s.append(current_x.detach())
            
            current_x = self.step(current_x, c, timesteps[i]*torch.ones_like(current_x)[...,0,None], timesteps[i+1]*torch.ones_like(current_x)[...,0,None])
        
        if return_intermediary_steps:
            return current_x.detach(), x_s
        else:
            return current_x.detach()

    def compute_loss(self, c, target, num_train_steps=100):
        """
        c : {mod1:(d1,...,dn), mod2: ..., ...}
        target : (B, dim)
        """
        timesteps = torch.rand(num_train_steps, 1, 1, device=target.device)

        source_dist = torch.randn_like(target)

        trajectory = target[None,...]*timesteps + (1-timesteps)*source_dist[None,...]

        for key in c.keys():
            #print(torch.ones_like(timesteps.view(-1, *[1 for i in range(len(c[key].shape))])).shape)

            c[key] = torch.flatten(c[key][None,...]*torch.ones_like(timesteps.view(-1, *[1 for i in range(len(c[key].shape))])), end_dim=1)

        pred_target = self.model(torch.flatten(trajectory, end_dim=1), c, torch.flatten(timesteps*torch.ones_like(target[None,...,0,None]), end_dim=1))

        return F.mse_loss(pred_target, torch.flatten((target-source_dist)[None,...]*torch.ones_like(timesteps), end_dim=1))