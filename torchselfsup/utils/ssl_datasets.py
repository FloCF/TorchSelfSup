from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

from torchselfsup.augmentation import *

class SSL_CIFAR10(object):
    
    def __init__(self, data_root, augmentation, dl_kwargs):
        assert augmentation in ['BYOL', 'SimSiam', 'VICReg'], "augmentation must be in ['BYOL', 'SimSiam', 'VICReg']"
        # Cifar10 Mean and Std
        CIFAR10_NORM = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
        
        # Define Augmentations    
        if augmentation=='BYOL':
            train_transf = BYOL_augmentaions(image_size=32, normalize=CIFAR10_NORM)
        elif augmentation=='SimSiam':
            train_transf = SimSiam_augmentaions(image_size=32, normalize=CIFAR10_NORM)
        elif augmentation=='VICReg':
            train_transf = VICReg_augmentaions(image_size=32, normalize=CIFAR10_NORM)
            
        train_eval_transf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR10_NORM)
        ])
        
        test_transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*CIFAR10_NORM)])

        # Define Datasets
        train_ds = CIFAR10(root=data_root, train = True, download = True, transform = train_transf)
        train_eval_ds = CIFAR10(root=data_root, train = True, transform = train_eval_transf, download = True)
        test_ds  = CIFAR10(root=data_root, train = False, transform = test_transf, download = True)

        # Define Dataloaders
        self.train_dl = DataLoader(train_ds, drop_last=True, **dl_kwargs)
        self.train_eval_dl = DataLoader(train_eval_ds, drop_last=False, **dl_kwargs)
        self.test_dl  = DataLoader(test_ds, drop_last=False, **dl_kwargs)