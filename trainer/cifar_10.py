from os import path
import time

from torch.optim import lr_scheduler

from optimizer import LARS
from utils import check_existing_model, Linear_Protocoler

def cifar10_trainer(save_root, model, ssl_data, optim_params, train_params,
                    eval_params, verbose = True):    
    
    # Extract device
    device = next(model.parameters()).device
    # Init
    eval_acc = {'lin': [], 'knn': []}
    loss_hist = []
    # Define optimizer
    optimizer = LARS(model.parameters(), **optim_params)
    # Define scheduler for warmup
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/(train_params['warmup_epchs']*len(ssl_data.train_dl)))
    
    # Check for trained model
    train_params['epoch_start'], saved_data = check_existing_model(save_root, device)
    # Extract data
    if saved_data:
        barlow_twins.load_state_dict(saved_data['model'])
        optimizer.load_state_dict(saved_data['optim'])
        if epoch_start >= train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_data.train_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       iters_left,
                                                       eta_min=train_params['eta_min'])
        scheduler.load_state_dict(saved_data['sched'])
        eval_acc = saved_data['eval_acc']
        loss_hist = saved_data['loss_hist']
        
    if scheduler is None:
        # Define scheduler for warmup
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/(train_params['warmup_epchs']*len(ssl_data.train_dl)))
    
    # Run Training
    for epoch in range(train_params['epoch_start'], train_params['num_epochs']):
        epoch_loss = 0
        start_time = time.time()
        for (x1,x2), _ in ssl_data.train_dl:
            x1,x2 = x1.to(device), x2.to(device)
        
            # Forward pass
            loss = model(x1,x2)
        
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Scheduler every iteration for cosine deday
            scheduler.step()
        
            # Save loss
            epoch_loss += loss.item()
    
        # Switch to Cosine Decay after warmup period
        if epoch+1==train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_data.train_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       iters_left,
                                                       eta_min=train_params['eta_min'])
    
        # Log
        loss_hist.append(epoch_loss/len(ssl_data.train_dl))
        if verbose:
            print(f'Epoch: {epoch}, Loss: {loss_hist[-1]}, Time epoch: {time.time() - start_time}')
    
        # Run evaluation
        if (epoch+1) in eval_params['evaluate_at']:
            # Linear protocol
            evaluator = Linear_Protocoler(model.backbone_net, repre_dim=model.repre_dim)
            # knn accuracy
            eval_acc['knn'].append(evaluator.knn_accuracy(ssl_data.train_eval_dl, ssl_data.test_dl))
            print(f'KNN Accuracy after epoch {epoch}: {eval_acc["knn"][-1]}')
            # linear protocol
            evaluator.train(ssl_data.train_eval_dl, eval_params)
            eval_acc['lin'].append(evaluator.linear_accuracy(ssl_data.test_dl))
                        
            print(f'Accuracy after epoch {epoch}: KNN:{eval_acc["knn"][-1]}, Linear: {eval_acc["lin"][-1]}')
        
            torch.save({'model':barlow_twins.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict()},
                       path.join(save_root, f'epoch_{epoch+1:03}.tar'))