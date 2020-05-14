'''
the normalized temperature-scaled cross entropy loss
'''

import torch
import torch.nn as nn

class NT_Xent(nn.module):
    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.device =  device

        '''
        cosine similarity
        '''
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        '''
        we do not sample negative examples explicitly.
        given a positive pair, we treat the other 2(N-1)
        augmented examples within a minibatch as negative examples.
        '''
