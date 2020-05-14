'''
a neural network  base encoder f 
that extracts  representation vectors 
from augmented data examples.
We adopt the commonly used ResNet to obtain h_i=f(x_i)=ResNet(x_i)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class Encoder(nn.Module):
    def __init__(self, out_dim = 64):
        super(Encoder, self).__init__()

        
