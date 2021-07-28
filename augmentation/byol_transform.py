from torchvision import transforms

IMAGENET_NORM = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class BYOL_transform(object):
    def __init__(self, image_size, normalize=IMAGENET_NORM):

        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode('bicubic')),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # Kernel size given in SimCLR paper. Kernel size has to be odd
            transforms.GaussianBlur(image_size//20*2+1),
            transforms.ToTensor(),
            transforms.Normalize(*IMAGENET_NORM)
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode('bicubic')),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(image_size//20*2+1)], p=0.1),
            transforms.RandomSolarize(0.5, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*IMAGENET_NORM)
        ])

    def __call__(self, x):
        x1 = self.transform1(x) 
        x2 = self.transform2(x) 
        return x1, x2