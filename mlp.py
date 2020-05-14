import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers):
        super(MLP, self).__init__()

    layers = []
    prev_dim = in_dim
    
    self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
