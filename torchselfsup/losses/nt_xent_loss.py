import torch
import torch.nn as nn
from torch.nn import functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5,
                 memory_bank_size: int = 0):
        super().__init__()
        self.temp = temperature
        self.memory_bank_size = memory_bank_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        
        # init
        self.memory_bank = None
        
    def init_memory_bank(self, feature_dim:int, device):
        if self.memory_bank_size > 0:
            self.memory_bank = F.normalize(torch.randn(self.memory_bank_size,
                                                       feature_dim,
                                                       device=device), dim=1)
        else:
            raise Warning("Memory bank size is set to 0, so memory bank can not be initiated")
    
    @torch.no_grad()
    def add_to_memory_bank(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        # dequeue
        self.memory_bank = self.memory_bank[batch_size:]
        # enqueue
        self.memory_bank = torch.cat([self.memory_bank, batch.detach()])
    
    def forward(self, out1, out2, update_bank: bool = True):
        # get batch size, dim of embeddings and device
        bs, dim_feat = out1.shape
        device = out1.device
        
        # Normalize over feature dimensin
        out1, out2 = F.normalize(out1, dim=1), F.normalize(out2, dim=1) # [b, f]
        
        if self.memory_bank is None and self.memory_bank_size>0:
            self.init_memory_bank(dim_feat, device)
        
        # Code from https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
        if self.memory_bank is not None:
            self.memory_bank.to(device)
            # get positive similarities
            sim_pos = torch.einsum('bf,bf->b', out1, out2).unsqueeze(-1) # [b, 1]
            # get negative similarities
            sim_neg = torch.einsum('bf,mf->bm', out1, self.memory_bank) # [b, m]
            # get logits and labels
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temp # [b, 1+m]
            # All positives are on the first collumn
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
            # update memory bank
            if update_bank:
                self.add_to_memory_bank(out2)
        else:
            # Concat -> [2*b, f]
            reps = torch.cat([out1, out2], dim=0)
            # get logits
            logits = torch.einsum('bf,kf->bk', reps, reps) / self.temp #[2*b, 2*b]
            # Filter out similarities of samples with themself
            logits = logits[~torch.eye(2*bs, dtype=torch.bool, device=device)].view(2*bs, 2*bs-1) #[2*b, 2*b-1]
            
            # The labels point from a sample in out1 to its equivalent in out2
            labels = torch.arange(bs, device=device, dtype=torch.long)
            labels = torch.cat([labels + bs - 1, labels])
        
        return self.cross_entropy(logits, labels)