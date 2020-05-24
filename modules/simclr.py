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

'''
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

'''


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args

        self.encoder = self.get_resnet(args.resnet)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, args.projection_dim, bias=False),
        )


    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]


    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z

