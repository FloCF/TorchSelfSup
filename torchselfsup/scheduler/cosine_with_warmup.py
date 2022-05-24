from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

def CosineAnnealingWithWarmupLR(optimizer, num_epochs:int, len_traindl:int,
                                warmup_epochs:int = 10, min_lr:float = 1e-3,
                                iter_scheduler: bool = True):
    """
        optimizer (Optimizer): Wrapped optimizer.
        num_epochs (int): Total number of epochs.
        len_traindl (int): Total number of batches in the training Dataloader.
        warmup_epochs (int): Number of epochs for linear warmup. Default: 10.
        min_lr (float): Min learning rate. Default: 0.001.
        iter_scheduler (bool): If scheduler is used every iteration (True) or
                               every epoch (False). Default: True.
    """
    # Declare iteration or epoch-wise scheduler
    if iter_scheduler:
        train_len = len_traindl
    else:
        train_len = 1
        
    # Define Warmup scheduler   
    scheduler1 = LambdaLR(optimizer, lambda it: (it+1)/(warmup_epochs*train_len))
    # Define Cosine Annealing Scheduler
    scheduler2 = CosineAnnealingLR(optimizer, T_max=(num_epochs-warmup_epochs)*train_len, eta_min=min_lr)
    
    # Combine both
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs*train_len])
    
    return scheduler