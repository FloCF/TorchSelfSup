from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

class Linear_Protocoler(object):
    def __init__(self, encoder, repre_dim: int, device : str = 'cuda'):
        self.device = device
        
        # Copy net
        self.encoder = deepcopy(encoder)
        # Turn off gradients
        for p in self.encoder.parameters():
            p.requires_grad = False    
        
        self.repre_dim = repre_dim
        
    def knn_accuracy(self, train_dl, test_dl, knn_k: int = 200, knn_t: float=0.1):        
        # get classes
        num_classes = len(train_dl.dataset.classes)
        
        # extract train
        train_features = ()
        train_labels = ()
        for x,labels in train_dl:
            feats = self.encoder(x.to(self.device))
            train_features += (F.normalize(feats, dim=1),)
            train_labels += (labels,)
        train_features = torch.cat(train_features).t().contiguous()
        train_labels = torch.cat(train_labels).to(self.device)

        # Test
        total_top1, total_num = 0., 0
        for x,target in test_dl:
            x, target = x.to(self.device), target.to(self.device)
            features = self.encoder(x)
            features = F.normalize(features, dim=1)
            
            # Get knn predictions
            pred_labels = knn_predict(features, train_features, train_labels, num_classes, knn_k, knn_t)
            
            total_num += x.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
        
        return total_top1 / total_num * 100
    
    def train(self, dataloader, num_epochs, lr, milestones = None):
        # get classes
        num_classes = len(dataloader.dataset.classes)
        
        # Add classification layer
        self.classifier = nn.Sequential(self.encoder, nn.Linear(self.repre_dim, num_classes))
        
        # Send to device
        self.classifier = self.classifier.to(self.device)
        
        # Define optimizer
        optimizer = opt.Adam(self.classifier.parameters(), lr)
        # Define loss
        ce_loss = nn.CrossEntropyLoss()
        # Define scheduler
        if milestones:
            scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones)
        else:
            scheduler = None
        
        # Train
        for epoch in range(num_epochs):
            for x,y in dataloader:
                x,y = x.to(self.device), y.to(self.device)
                # forward
                loss = ce_loss(self.classifier(x), y)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
                
    def linear_accuracy(self, dataloader):
        total_top1, total_num = 0., 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.classifier(x)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total_num += y.size(0)
                total_top1 += (predicted == y).float().sum().item()
            self.classifier.train()
        
        return total_top1 / total_num * 100
    

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(feature: torch.Tensor,
                feature_bank: torch.Tensor,
                feature_labels: torch.Tensor, 
                num_classes: int,
                knn_k: int,
                knn_t: float) -> torch.Tensor:
    """Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature: 
            Tensor of shape [N, D] for which you want predictions
        feature_bank: 
            Tensor of a database of features used for kNN
        feature_labels: 
            Labels for the features in our feature_bank
        num_classes: 
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k: 
            Number of k neighbors used for kNN
        knn_t: 
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, num_classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels