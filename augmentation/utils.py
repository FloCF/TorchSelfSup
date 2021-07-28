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