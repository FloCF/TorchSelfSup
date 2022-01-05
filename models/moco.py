import copy

from math import cos,pi
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP

class MoCo(nn.Module):
    def __init__(self, backbone_net: nn.Module,
                 projector_hidden: Union[int, tuple] = (2048, 128),
                 memory_bank_size: int = 65536,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        
        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Identity()
        
        # Define projector and init memory bank
        if projector_hidden:
            self.projector = MLP(self.repre_dim, projector_hidden)
            self.memory_bank = F.normalize(torch.randn(memory_bank_size, projector_hidden[-1]), dim=1)
        else: # Use no projector
            self.projector = nn.Identity()
            self.memory_bank = F.normalize(torch.randn(memory_bank_size, self.repre_dim), dim=1) 
            
        # Define target backbone and projector
        self.encoder_momentum = copy.deepcopy(self.backbone_net)
        self.projector_momentum = copy.deepcopy(self.projector)
        # Turn of requires grad
        for model in [self.encoder_target, self.projector_target]:
            for p in model.parameters():
                p.requires_grad = False
    
    # get tau based on BYOL schedule
    def get_tau(self, k, K, base_tau: float=0.996):
        return 1 - 0.5 * (1 - base_tau) * (cos(pi*k/K)+1)
    
    @torch.no_grad()
    def update_moving_average(self, τ):
        # Update backbone encoder
        for online, target in zip(self.backbone_net.parameters(), self.encoder_momentum.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data
        # Update projector
        for online, target in zip(self.projector.parameters(), self.projector_momentum.parameters()):
            target.data = τ * target.data + (1 - τ) * online.data 
    
    @torch.no_grad()
    def _add_to_memory_bank(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        # dequeue
        self.memory_bank = self.memory_bank[batch_size:]
        # enqueue
        self.memory_bank = torch.cat([self.memory_bank, batch.detach()])
        
    # Code copied from https://github.com/lightly-ai/lightly/blob/master/lightly/models/utils.py
    @torch.no_grad()
    def batch_shuffle(batch: torch.Tensor):
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle
    
    @torch.no_grad()
    def batch_unshuffle(batch: torch.Tensor, shuffle: torch.Tensor):
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]
    
    def _NTXentLoss(self, p, z, temp):
        # Extract stats
        device=p.device # Device
        bs = p.shape[0] # Batch_size
        # Normalize
        p, z = F.normalize(p, dim=1), F.normalize(z, dim=1)
        
        # b,k: batch size of p and z, f: embedding_size, m: memory bank size
        if len(self.memory_bank)>0:
            sim_pos = torch.einsum('bf,bf->b', p, z).unsqueeze(-1)
            sim_neg = torch.einsum('bf,mf->bm', p, self.memory_bank)
            logits = torch.cat([sim_pos, sim_neg], dim=1) / temp
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
        else: # If no memory, use other samples from batch as negatives
            output = torch.cat([p, z], dim=0)
            logits = torch.einsum('bf,kf->bk', output, output) / temp
            logits = logits[~torch.eye(2*bs, dtype=torch.bool, device=device)].view(2*bs, -1)
            labels = torch.arange(bs, device=device, dtype=torch.long)
            labels = torch.cat([labels + bs - 1, labels])
            
        return F.cross_entropy(logits, labels), z
    
    def loss_fn(self, p1, p2, z1, z2, temp):
        loss1, z2 = self._NTXentLoss(p1, z2, temp)
        loss2, z1 = self._NTXentLoss(p2, z1, temp)
        
        self._add_to_memory_bank(torch.cat([z1,z2]))
        
        return 0.5 * (loss1 + loss2)

    def forward(self, x1, x2):
        p1 = self.projector(self.backbone_net(x1))
        p2 = self.projector(self.backbone_net(x2))
        
        x1, shuffle = self.batch_shuffle(x1)
        z1 = self.projector_momentum(self.encoder_momentum(x1))
        z1 = batch_unshuffle(z1, shuffle)
        
        x2, shuffle = self.batch_shuffle(x2)
        z2 = self.projector_momentum(self.encoder_momentum(x2))
        z2 = batch_unshuffle(z2, shuffle)
                
        return self.loss_fn(p1, p2, z1, z2, self.temperature)