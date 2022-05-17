from os import path
import time

import torch
from torch.optim import lr_scheduler

from torchssl.utils import check_existing_model, Linear_Protocoler

class SSL_Trainer(object):
    def __init__(self, model, ssl_data, device='cuda',
                 use_momentum: bool = False, momentum_tau: float = None,
                 use_memory_bank: bool = False):
        # Define device
        self.device = torch.device(device)
        
        # Define if use momentum
        self.use_momentum = use_momentum
        self.momentum_tau = momentum_tau
        self.use_memory_bank = use_memory_bank
        
        # Init
        self.loss_hist = []
        self.eval_acc = {'lin': [], 'knn': []}
        self._iter_scheduler = False
        self._hist_lr = []
        
        # Model
        self.model = model.to(self.device)
        
        # Define data
        self.data = ssl_data

    def train_epoch(self, epoch_id):
        for i, ((x1,x2), _) in enumerate(self.data.train_dl):
            x1,x2 = x1.to(self.device), x2.to(self.device)
        
            # Forward pass
            loss = self.model(x1,x2)
        
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update momentum encoder
            if self.use_momentum:
                if self.momentum_tau: # Use predefined momentum
                    tau = self.momentum_tau
                else: # get τ based on BYOL schedule
                    tau = self.model.get_tau(1+i+self._train_len*epoch_id, self._total_iters)
                self.model.update_moving_average(tau)
            
            # save learning rate
            self._hist_lr.append(self.scheduler.get_last_lr())

            if self.scheduler and self._iter_scheduler:
                # Scheduler every iteration, e.g. for cosine deday
                self.scheduler.step()
        
            # Save loss
            self._epoch_loss += loss.item()
    
    def evaluate(self, num_epochs, lr, milestones = None):
        # Linear protocol
        evaluator = Linear_Protocoler(self.model.backbone_net, repre_dim=self.model.repre_dim, device=self.device)
        # knn accuracy
        self.eval_acc['knn'].append(evaluator.knn_accuracy(self.data.train_eval_dl, self.data.test_dl))
        # linear protocol
        evaluator.train(self.data.train_eval_dl, num_epochs, lr, milestones)
        self.eval_acc['lin'].append(evaluator.linear_accuracy(self.data.test_dl))

    def train(self, save_root, num_epochs, optimizer,
              scheduler, optim_params, scheduler_params, eval_params,
              iter_scheduler=True, evaluate_at=[100,200,400], verbose=True):
        
        # Check and Load existing model
        epoch_start, optim_state, sched_state = self.load_model(save_root, return_vals=True)
        
        # Extract training length
        self._train_len = len(self.data.train_dl)
        self._total_iters = num_epochs * self._train_len
        
        # Define Optimizer
        self.optimizer = optimizer(self.model.parameters(), **optim_params)
        # Load existing optimizer
        if optim_state:
            self.optimizer.load_state_dict(optim_state)
        
        # Define Scheduler
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
            self._iter_scheduler = iter_scheduler
            # Load existing scheduler
            if sched_state:
                self.scheduler.load_state_dict(sched_state
        else: # scheduler = None
            self.scheduler = scheduler

        # Run Training
        for epoch in range(epoch_start, num_epochs):
            self._epoch_loss = 0
            start_time = time.time()
            
            self.train_epoch(epoch)
            
            if self.scheduler and not self._iter_scheduler:
                # Scheduler only every epoch
                self.scheduler.step()
    
            # Log
            self.loss_hist.append(self._epoch_loss/self._train_len)
            if verbose:
                print(f'Epoch: {epoch}, Loss: {self.loss_hist[-1]}, Time epoch: {time.time() - start_time}')
    
            # Run evaluation
            if (epoch+1) in evaluate_at:
                self.evaluate(**eval_params)
                # print
                print(f'Accuracy after epoch {epoch}: KNN:{self.eval_acc["knn"][-1]}, Linear: {self.eval_acc["lin"][-1]}')
            
                # Save model
                self.save_model(save_root, epoch+1)
        
        # Evaluate after Training
        self.evaluate(**eval_params)
        # print
        print(f'Accuracy after full Training: KNN:{self.eval_acc["knn"][-1]}, Linear: {self.eval_acc["lin"][-1]}')
        
        # Save final model
        self.save_model(save_root, num_epochs)
        
    def save_model(self, save_root, epoch):
        save_data = {'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sched': self.scheduler.state_dict() if self.scheduler else None,
                    'loss_hist': self.loss_hist,
                    'eval_acc': self.eval_acc,
                    'lr_hist': self._hist_lr}
        # Add the memory bank to the save data
        if self.use_memory_bank:
            save_data['memory_bank'] = self.model.nt_xent_loss.memory_bank
        
        torch.save(save_data, path.join(save_root, f'epoch_{epoch:03}.tar'))
    
    def load_model(self, save_root, return_vals=False):
        # Check for trained model
        epoch_start, saved_data = check_existing_model(save_root, self.device)
        
        if saved_data is None:
            if return_vals:
                return epoch_start, None, None
        else:
            self.model.load_state_dict(saved_data['model'])
            self.loss_hist = saved_data['loss_hist']
            self.eval_acc = saved_data['eval_acc']
            self._hist_lr = saved_data['lr_hist']
            if self.use_memory_bank:
                self.model.nt_xent_loss.memory_bank = saved_data['memory_bank']
                
            if return_vals:
                return epoch_start, saved_data['optim'], saved_data['sched']
        
        