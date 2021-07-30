import torchvision.transforms as T

IMAGENET_NORM = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

# SimCLR Transformations
def simclr_transforms(image_size: int, jitter: tuple = (0.4, 0.4, 0.2, 0.1),
                      p_blur: float = 1.0, p_solarize: float = 0.0,
                      normalize: list = IMAGENET_NORM):
    
    trans_list = [T.RandomResizedCrop(image_size, interpolation=T.InterpolationMode('bicubic')),
                  T.RandomHorizontalFlip(),
                  T.RandomApply([T.ColorJitter(*jitter)], p=0.8),
                  T.RandomGrayscale(p=0.2)]
    
    # Turn off blur for small images
    if image_size<=32:
        p_blur = 0.0
    # Add Gaussian blur
    if p_blur==1.0:
        trans_list.append(T.GaussianBlur(image_size//20*2+1))
    elif p_blur>0.0:
        trans_list.append(T.RandomApply([T.GaussianBlur(image_size//20*2+1)], p=p_blur))
    # Add RandomSolarize
    if p_solarize>0.0:
        trans_list.append(T.RandomSolarize(0.5, p=p_solarize))
    
    trans_list.extend([T.ToTensor(), T.Normalize(*normalize)])
    
    return T.Compose(trans_list)


# A wrapper that performs returns two augmented images
class TwoTransform(object):
    """Applies data augmentation two times."""

    def __init__(self, base_transform, sec_transform = None):
        self.base_transform = base_transform
        self.sec_transform = base_transform if sec_transform is None else sec_transform
        
    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.sec_transform(x)
        return x1,x2