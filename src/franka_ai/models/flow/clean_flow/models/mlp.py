import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP_v1(nn.Module):
    def __init__(self, input_dim, fc_dims, act_fct="relu"):
        super().__init__()

        modules = [nn.Linear(input_dim, fc_dims[0])]
        for i in range(len(fc_dims)-1):

            if act_fct == "elu":
                modules.append(nn.ELU())
            else:
                modules.append(nn.ReLU())
            modules.append(nn.Linear(fc_dims[i], fc_dims[i+1]))

        #print(modules)
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)
    
class FlowMLP(nn.Module):
    def __init__(self, input_dim, dims, global_cond_dim={"state":1}, obs_sequence_length=1):
        super().__init__()

        new_global_cond_dim = 0
        for key in global_cond_dim.keys():
            new_global_cond_dim += global_cond_dim[key]*obs_sequence_length

        self.mlp = MLP_v1(input_dim+new_global_cond_dim+1, [dim for dim in dims]+[input_dim])

    def forward(self, x, global_cond, t):
        gcond = [torch.flatten(global_cond[key], start_dim=1) for key in global_cond.keys()]

        #print(x.shape, t.shape, [cond.shape for cond in gcond])
        return self.mlp(torch.cat([x]+[t]+gcond, dim=-1))