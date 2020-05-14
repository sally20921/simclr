import torchvision

'''
a stochastic data augmentation  module that 
transforms any given data example randomly
resulting  in two correlated views of the same example.
x_i, x_j which we consider as a positive pair
'''

class Aug:
    def __init__(self, size):
        s = 1 
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        self.aug = torchvision.transforms.Compose()

    def __call__(self, x):
        return self.aug(x)
