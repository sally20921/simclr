'''
resulting  in two  correlated view of the same example 
denoted  x~_i and x~_j, which we consider as a positive pair.

we sequentially apply three simple augmentations:
   RANDOM CROPPING followed by 
   resize back to the original size (224 * 224) 
   RANDOM COLOR DISTORTIONS
   RANDOM GAUSSIAN BLUR 

'''

from  torchvision import transforms

class Augmentation:
    def __init__(self, size):

    s = 1 
    color_jitter =  torchvision.transforms.ColorJitter(
            0.8*s, 0.8*s, 0.8*s, 0.2*s
    )

    self.train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(), 
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RnadomGrayscale(p=0.2), 
            torchvision.transforms.ToTensor(),
        ]
    )

    def __call__(self, x):
        return self.train_transform(x)

