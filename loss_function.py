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
