from .utils import ssl_transforms, TwoTransform


IMAGENET_NORM = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class BYOL_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = ssl_transforms(image_size, normalize=normalize)
        trans2 = ssl_transforms(image_size, p_blur=0.1, p_solarize=0.2, normalize=normalize)
        
        super().__init__(trans1, trans2)
        

class SimSiam_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = ssl_transforms(image_size,
                                jitter = (0.4,0.4,0.4,0.1),
                                p_blur = 0.5,
                                normalize=normalize)
        
        super().__init__(trans1)
        
        
class VICReg_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=IMAGENET_NORM):
        trans1 = ssl_transforms(image_size,
                                p_blur = 0.5,
                                p_solarize = 0.1,
                                normalize=normalize)
        
        super().__init__(trans1)
        

class OneChannel_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=None):
        trans1 = ssl_transforms(image_size, p_gray = 0., p_blur = 0.,
                                normalize=normalize)
        trans2 = ssl_transforms(image_size, p_gray = 0., p_blur = 0.,
                                p_solarize=0.2, normalize=normalize)

        super().__init__(trans1, trans2)