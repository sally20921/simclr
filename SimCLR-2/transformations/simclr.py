import torchvision

from transformations.gaussian_blur import GaussianBlur


class TransformsSimCLR:
    """
    Implementation of the data transformation for SimCLR accordingly to the published
    paper https://arxiv.org/abs/2002.05709
    """

    def __init__(self, size):
        # strentgh of the color distortion. denoted es s in the paper
        s = 1

        # color destortian based on the pseudo-code in the paper appendix A
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        color_jitter_prob = 0.8

        # gaussian blur should be applied with a probability of 0.5
        gaussian_blur = GaussianBlur(0.1, 0.1, 2.0)
        gaussian_blur_prob = 0.5

        gray_scale_prob = 0.2

        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=color_jitter_prob),
                torchvision.transforms.RandomGrayscale(p=gray_scale_prob),
                # TODO: atm this has to be the last transform prior to ToTensor(). find a better approach?
                torchvision.transforms.RandomApply([gaussian_blur], p=gaussian_blur_prob),
                torchvision.transforms.ToTensor(),
            ]
        )

        # test transformations used to test the trained features by simclr
        self.train_test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ]
        )

        # do not apply the transformations in the test case. only perform a resize
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.CenterCrop(size=size),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        # apply the randomized transformation 2 times in order to obtain two correlated views
        x_i = self.train_transform(x)
        x_j = self.train_transform(x)

        return x_i, x_j
