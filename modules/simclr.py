'''
for sampled minibatch x_k 1~N do 
(train at batch size 4096 for 100 epochs) 
(linear warmup for the first  10 epochs)

#the first augmentation  
#representation
#projection

#the second augmentation
#representation
#projection


for all 1~2N

#pairwise similarity  
#NT-Xent

#update  f and g 
'''
import os 
import torch.nn as nn
import torchvision


class SimCLR(nn.Module):
    def __init__(self, args):
    
        super(SimCLR, self).__init__()

        self.args = args

        self.encoder = self.get_resnet(args.resnet)
        self.projector =  nn.Sequential(
                nn.Linear(),
                nn.ReLU(), 
                nn.Linear(),
        )

    def get_resnet(self,  name):
        resnets = {
                "resnet18": torchvision.models.resnet18(),
                "resnet50": torchvision.models.resnet50(),
            }
        return resnets[name]

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        
        z =  nn.functional.normalize(z, dim=1)

        return h,z



