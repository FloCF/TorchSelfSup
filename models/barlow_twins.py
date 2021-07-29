from typing import Union

import torch
import torch.nn as nn

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
        
        if isinstance(projector_hidden, int):
            projector_hidden = (projector_hidden,)
        
        # Projector
        projector = [nn.Linear(repre_dim, projector_hidden[0], bias=False)]
        for i in range(len(projector_hidden) - 1):
            projector.extend([nn.BatchNorm1d(projector_hidden[i]),
                              nn.ReLU(inplace=True),
                              nn.Linear(projector_hidden[i], projector_hidden[i+1], bias=False)])
            
        self.projector = nn.Sequential(*projector)

        # Normalization layer for representations z1 + z2
        self.bn = nn.BatchNorm1d(projector_hidden[-1], affine=False)
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    
    def forward(self, x1, x2):
        bz = x1.size(0)
        
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(bz)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss