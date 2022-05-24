from os import listdir
from os.path import isfile, join

import torch

def check_existing_model(save_root, device):
    """
    Epochs must be saved in format: f'epoch_{epoch:03}.tar'.
    E.g. for epoch 20: epoch_020.tar.
    """
    # Get all files
    files = [f for f in listdir(save_root) if isfile(join(save_root, f))]
    
    # init
    epoch_start = 0
    saved_data = None
    if len(files)>0:
        user_answer = "Users_answer"
        while user_answer not in ["y","n"]:
            user_answer = input("Pretrained model available, use it?[y/n]: ").lower()[0]
        if user_answer=="y":
            epoch_start = max([int(file[-7:-4]) for file in files])
            # Load data
            saved_data = torch.load(join(save_root, f'epoch_{epoch_start:03}.tar'), map_location=device)
    
    return epoch_start, saved_data