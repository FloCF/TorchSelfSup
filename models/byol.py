import copy

from math import cos,pi

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_dim:int = 1024):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class BYOL(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 project_dim: int, project_hidden: int):
        super().__init__()
        
        self.backbone_net = backbone_net
        repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        self.projector = MLP(repre_dim, project_dim, project_hidden)
        
        # Define encoders
        self.encoder_online = nn.Sequential(self.backbone_net, self.projector)
        self.encoder_target = copy.deepcopy(self.encoder_online)
        # Turn of requires grad
        for p in self.encoder_target.parameters():
            p.requires_grad = False
            
        # Define online predictor
        self.predictor_online = MLP(project_dim, project_dim, project_hidden)
        
    def get_tau(self, k, K, base_tau: float=0.996):
        return 1 - 0.5 * (1 - base_tau) * (cos(pi*k/K)+1)
    
    @torch.no_grad()
    def update_moving_average(self, τ):
        for online, target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data
            
    def loss_fn(self, p1, p2, z1_t, z2_t):
        loss = F.cosine_similarity(p1, z1_t.detach(), dim=-1).mean()
        loss += F.cosine_similarity(p2, z2_t.detach(), dim=-1).mean()
        return 4 - 2 * loss

    def forward(self, x1, x2):
        p1 = self.predictor_online(self.encoder_online(x1))
        p2 = self.predictor_online(self.encoder_online(x2))
        
        with torch.no_grad():
            z1_t, z2_t = self.encoder_target(x2), self.encoder_target(x1)
        
        return self.loss_fn(p1, p2, z1_t, z2_t)