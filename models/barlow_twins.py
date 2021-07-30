from typing import Union

import torch
import torch.nn as nn

from .utils import MLP

class BarlowTwins(nn.Module):
    """
    Code from https://github.com/facebookresearch/barlowtwins
    """
    def __init__(self, backbone_net,
                 projector_hidden: Union[int, tuple] = (8192,8192,8192),
                 λ: float = 0.0051):
        super().__init__()
        
        self.lambd = λ
        
        self.backbone_net = backbone_net
        repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        self.projector = MLP(repre_dim, projector_hidden, bias = False)

        # Normalization layer for representations z1 + z2
        self.bn = nn.BatchNorm1d(projector_hidden[-1], affine=False)
    
    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2, λ):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        
        return on_diag + λ * off_diag
    
    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        return self.loss_fn(z1, z2, self.lambd)