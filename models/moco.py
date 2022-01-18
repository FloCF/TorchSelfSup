import copy

from math import cos,pi
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import NTXentLoss
from .utils import MLP

class MoCo(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 projector_hidden: Union[int, tuple] = (2048, 128),
                 predictor_hidden: Optional[Union[int, tuple]] = None,
                 temperature: float = 0.2,
                 memory_bank_size: int = 65536):
        super().__init__()
        
        self.nt_xent_loss = NTXentLoss(temperature, memory_bank_size)
        
        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define projector and init memory bank
        if projector_hidden:
            self.projector = MLP(self.repre_dim, projector_hidden, batchnorm_last=True)
        else: # Use no projector
            self.projector = nn.Identity()

        # Define online predictor
        if predictor_hidden:
            if projector_hidden:
                self.predictor = MLP(projector_hidden[-1], predictor_hidden)
            else:
                self.predictor = MLP(self.repre_dim, predictor_hidden)
        else: # Don't use predictor
            self.predictor = nn.Identity()   
            
        # Define momentum backbone and projector
        self.encoder_momentum = copy.deepcopy(self.backbone_net)
        self.projector_momentum = copy.deepcopy(self.projector)
        # Turn of requires grad
        for model in [self.encoder_momentum, self.projector_momentum]:
            for p in model.parameters():
                p.requires_grad = False
    
    # get tau based on BYOL schedule
    def get_tau(self, k, K, base_tau: float=0.996):
        return 1 - 0.5 * (1 - base_tau) * (cos(pi*k/K)+1)
    
    @torch.no_grad()
    def update_moving_average(self, τ):
        # Update backbone encoder
        for online, momentum in zip(self.backbone_net.parameters(), self.encoder_momentum.parameters()):
            momentum.data = τ * momentum.data + (1 - τ) * online.data
        # Update projector
        for online, momentum in zip(self.projector.parameters(), self.projector_momentum.parameters()):
            momentum.data = τ * momentum.data + (1 - τ) * online.data 
        
    # Code copied from https://github.com/lightly-ai/lightly/blob/master/lightly/models/utils.py
    @torch.no_grad()
    def batch_shuffle(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle
    
    @torch.no_grad()
    def batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]
    
    def loss_fn(self, p1, p2, z1, z2, update_bank):
        loss1 = self.nt_xent_loss(p1, z2, update_bank)
        loss2 = self.nt_xent_loss(p2, z1, update_bank)
                
        return 0.5 * (loss1 + loss2)

    def forward(self, x1, x2, shuffle = True):
        # Encoder
        p1, p2 = self.backbone_net(x1), self.backbone_net(x2)
        # Projector
        p1, p2 = self.projector(p1), self.projector(p2)
        # Predictor
        p1, p2 = self.predictor(p1), self.predictor(p2)
        
        with torch.no_grad():
            if shuffle:
                x1, shuffle = self.batch_shuffle(x1)
                z1 = self.projector_momentum(self.encoder_momentum(x1))
                z1 = self.batch_unshuffle(z1, shuffle)
            
                x2, shuffle = self.batch_shuffle(x2)
                z2 = self.projector_momentum(self.encoder_momentum(x2))
                z2 = self.batch_unshuffle(z2, shuffle)
            else:
                z1 = self.projector_momentum(self.encoder_momentum(x1))
                z2 = self.projector_momentum(self.encoder_momentum(x2))
            
        return self.loss_fn(p1, p2, z1, z2, self.training)