import  torch.nn as nn
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args
