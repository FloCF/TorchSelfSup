import copy

from math import cos,pi
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import NTXentLoss
from .utils import MLP

class SimCLR(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 projector_hidden: Union[int, tuple] = (2048, 2048, 256),
                 temperature: float = 0.2,
                 direct_dim: Optional[int] = None):
        super().__init__()
        
        # DirectCLR appoach from https://arxiv.org/abs/2110.09348 (default 360 for ResNet50)
        self.direct_dim = int(direct_dim)
        self.nt_xent_loss = NTXentLoss(temperature)
        
        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define projector and init memory bank
        if projector_hidden:
            self.projector = MLP(self.repre_dim, projector_hidden, batchnorm_last=True)
        else: # Use no projector
            self.projector = nn.Identity()  

    def forward(self, x1, x2):
        # Encoder
        p1, p2 = self.backbone_net(x1), self.backbone_net(x2)
        # Projector
        if self.direct_dim:
            p1, p2 = p1[:self.direct_dim], p2[:self.direct_dim]
        else:
            p1, p2 = self.projector(p1), self.projector(p2)
            
        return self.nt_xent_loss(p1, p2)
