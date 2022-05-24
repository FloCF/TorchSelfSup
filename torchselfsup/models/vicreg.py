from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP

class VICReg(nn.Module):
    """
    Code adapted from https://github.com/facebookresearch/barlowtwins
    """
    def __init__(self,
                 backbone_net: nn.Module, repre_dim: int,
                 projector_hidden: Union[int, tuple] = (8192,8192,8192),
                 λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super().__init__()
        
        self.backbone_net = backbone_net
        self.repre_dim = repre_dim
        
        # Loss Hyperparams
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ
        
        self.projector = MLP(self.repre_dim, projector_hidden, bias = False)
    
    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
        # Get batch size and dim of rep
        N,D = z1.shape
        
        # invariance loss
        sim_loss = F.mse_loss(z1, z2)
        
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() + torch.relu(γ - std_z2).mean()
        
        # covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / D
        
        return λ*sim_loss + μ*std_loss + ν*cov_loss
    
    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        return self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps)