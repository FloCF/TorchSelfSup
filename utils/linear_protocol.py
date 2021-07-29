from copy import deepcopy
from typing import Optional

from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.optim as opt

class Linear_Protocoler(object):
    def __init__(self, backbone_net, num_classes: int = 10, out_dim: Optional[int] = None, device : str = 'cpu'):
        self.device = device
        # Copy net
        self.classifier = deepcopy(backbone_net)
        # Turn off gradients
        for p in self.classifier.parameters():
            p.requires_grad = False
        # get out dimension
        if out_dim:
            out_dim = out_dim
        else:
            out_dim = p.shape[0]
        # Add classification layer
        self.classifier.fc = nn.Sequential(nn.Linear(out_dim, num_classes))
        # Send to device
        self.classifier = self.classifier.to(self.device)
    
    def knn_accuracy(self, train_dl, test_dl, n_neighbors: int = 8):
        # Turn off last layer
        last_layer_saved = deepcopy(self.classifier.fc)
        self.classifier.fc = nn.Flatten()
        
        # extract train
        X_train = ()
        Y_train = ()
        for x,y in train_dl:
            X_train += (self.classifier(x.to(self.device)).detach(),)
            Y_train += (y,)
        X_train = torch.cat(X_train).cpu().numpy()
        Y_train = torch.cat(Y_train).cpu().numpy()

        # extract test
        X_test = ()
        Y_test = ()
        for x,y in test_dl:
            X_test += (self.classifier(x.to(self.device)).detach(),)
            Y_test += (y,)
        X_test = torch.cat(X_test).cpu().numpy()
        Y_test = torch.cat(Y_test).cpu().numpy()
        
        # Restore last layer
        self.classifier.fc = last_layer_saved
        
        # Run K- NearestNeighbor
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X_train, Y_train)
        preds = neigh.predict(X_test)
        
        return sum(preds==Y_test)/len(preds)
    
    def train(self, dataloader, num_epochs, lr : float = 1e-3, milestones : Optional[list] = None):
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
                
    def get_accuracy(self, dataloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            self.classifier.eval()
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.classifier(x)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            self.classifier.train()
        
        return correct / total