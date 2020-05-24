'''
the  contrastive prediction task 

aims to identify x_j in x_k for a given x_i

for convenience, we term it NT-Xent 
(the  normalized temperature-scaled cross entropy loss)
(optimized using LARS with linear learning rate  scaling 

LearningRate = 0.3 * BatchSize/256
weight decay of 10**-6)


'''

import torch 
import torch.nn as nn
import numpy as np

class NTXentLoss(torch.nn.Module):
    def __init__(self, device,  batch_size, temperature):
        super(NTXentLoss, self).__init__()

        self.batch_size = batch_size
        self.temperature =  temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity =nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        p = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity(p.unsqueeze(1), p.unsqueeze(0))/self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        #positive samples 
        #negative  samples 

        loss = self.criterion()
        loss /=2*self.batch_size
        return loss 
