import cv2
import numpy as np


class GaussianBlur(object):
    """
    Implementation of Gaussian blur as described in the SimCLR paper https://arxiv.org/abs/2002.05709.
    Uses OpenCV to perform the blur operation
    """

    def __init__(self, kernel_size=0.1, min_sigma=0.1, max_sigma=2.0):
        # sigma set to random value in [0.1, 2.0] as stated in the paper
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # kernel size equals 10% of the image size in the paper
        self.kernel_size = kernel_size

    def __call__(self, sample):
        image = np.array(sample)

        kernel_size = int(min(sample.size) * self.kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        # print('kernel_size: {}'.format(kernel_size))

        rnd_sigma = np.random.random_sample()
        sigma = (self.max_sigma - self.min_sigma) * rnd_sigma + self.min_sigma
        # print('sigma: {}'.format(sigma))

        sample = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        return sample
