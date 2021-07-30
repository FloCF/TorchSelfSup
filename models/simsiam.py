from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP

class SimSiam(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 projector_hidden: Union[int, tuple] = (2048, 2048, 2048),
                 predictor_hidden: Optional[Union[int, tuple]] = (512, 2048)):
        super().__init__()
        
        self.backbone_net = backbone_net
        repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define projector
        self.projector = MLP(repre_dim, projector_hidden, batchnorm_last = True)
            
        # Define online predictor
        if predictor_hidden:
            self.predictor = MLP(projector_hidden[-1], predictor_hidden)
        else: # use same as projection mlp
            self.predictor = MLP(projector_hidden[-1], projector_hidden)
        
    def loss_fn(self, p1, p2, z1, z2):
        loss = F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss += F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        return 1 - 0.5 * loss # add 1 so that loss = 0 for perfect allignment

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))
        
        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        return self.loss_fn(p1, p2, z1, z2)