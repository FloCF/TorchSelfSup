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
        repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define encoders
        self.encoder_online = nn.Sequential(self.backbone_net, MLP(repre_dim, projector_hiddenprojector_hidden))
        self.encoder_target = copy.deepcopy(self.encoder_online)
        # Turn of requires grad
        for p in self.encoder_target.parameters():
            p.requires_grad = False
            
        # Define online predictor
        if predict_hiddens:
            self.predictor_online = MLP(projector_hidden[-1], predictor_hidden)
        else: # use same as projection mlp
            self.predictor_online = MLP(projector_hidden[-1], projector_hidden)
        
    def get_tau(self, k, K, base_tau: float=0.996):
        return 1 - 0.5 * (1 - base_tau) * (cos(pi*k/K)+1)
    
    @torch.no_grad()
    def update_moving_average(self, τ):
        for online, target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data
            
    def loss_fn(self, p1, p2, z1, z2):
        loss = F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss += F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        return 1 - 0.5 * loss # add 1 so that loss = 0 for perfect allignment

    def forward(self, x1, x2):
        p1 = self.predictor_online(self.encoder_online(x1))
        p2 = self.predictor_online(self.encoder_online(x2))
        
        with torch.no_grad():
            z1, z2 = self.encoder_target(x1), self.encoder_target(x2)
        
        return self.loss_fn(p1, p2, z1, z2)