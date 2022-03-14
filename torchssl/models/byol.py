import copy

from math import cos,pi
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP

class BYOL(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 projector_hidden: Union[int, tuple] = (4096, 256),
                 predictor_hidden: Optional[Union[int, tuple]] = None):
        super().__init__()
        
        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define projector
        self.projector = MLP(self.repre_dim, projector_hidden)
        
        # Define online predictor
        if predictor_hidden:
            self.predictor = MLP(projector_hidden[-1], predictor_hidden)
        else: # use same as projection mlp
            self.predictor = MLP(projector_hidden[-1], projector_hidden)
            
        # Define target backbone and projector
        self.encoder_target = copy.deepcopy(self.backbone_net)
        self.projector_target = copy.deepcopy(self.projector)
        # Turn of requires grad
        for model in [self.encoder_target, self.projector_target]:
            for p in model.parameters():
                p.requires_grad = False
        
    def get_tau(self, k, K, base_tau: float=0.996):
        return 1 - 0.5 * (1 - base_tau) * (cos(pi*k/K)+1)
    
    @torch.no_grad()
    def update_moving_average(self, τ):
        # Update backbone encoder
        for online, target in zip(self.backbone_net.parameters(), self.encoder_target.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data
        # Update projector
        for online, target in zip(self.projector.parameters(), self.projector_target.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data 
         
    def loss_fn(self, p1, p2, z1_hat, z2_hat):
        loss = F.cosine_similarity(p1, z2_hat.detach(), dim=-1).mean()
        loss += F.cosine_similarity(p2, z1_hat.detach(), dim=-1).mean()
        return 1 - 0.5 * loss # add 1 so that loss = 0 for perfect allignment

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))
        
        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        with torch.no_grad():
            z1_hat = self.projector_target(self.encoder_target(x1))
            z2_hat = self.projector_target(self.encoder_target(x2))
        
        return self.loss_fn(p1, p2, z1_hat, z2_hat)