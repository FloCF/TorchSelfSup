from typing import Union

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias: bool = True,
                 use_batchnorm: bool = True,
                 batchnorm_last: bool = False):
        super().__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        
        mlp = [nn.Linear(in_dim, hidden_dims[0], bias = bias)]
        for i in range(len(hidden_dims) - 1):
            if use_batchnorm:
                mlp.append(nn.BatchNorm1d(hidden_dims[i]))
            mlp.extend([nn.ReLU(inplace=True),
                        nn.Linear(hidden_dims[i], hidden_dims[i+1], bias = bias)])
        if batchnorm_last:
            # for simplicity, remove gamma in last BN
            mlp.append(nn.BatchNorm1d(hidden_dims[-1], affine=False)
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)