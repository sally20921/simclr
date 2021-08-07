import torch.nn as nn
import torchvision


class Identity(nn.Module):
    """
    helper class. a nn module which just outputs the input. can be used to disable a layer
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):

    def __init__(self, config):
        super(SimCLR, self).__init__()

        self.config = config

        self.encoder = self.get_resnet_model(config.simclr.model.resnet)

        self.num_features = self.encoder.fc.in_features

        # remove fully connected layer as described in the paper
        self.encoder.fc = Identity()

        # projection head denoted as g in the paper
        self.projector = nn.Sequential(
            nn.Linear(self.num_features, self.num_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.num_features, config.simclr.model.projection_dim, bias=False),
        )

    @staticmethod
    def get_resnet_model(name):

        if name == 'resnet18':
            model = torchvision.models.resnet18()
        elif name == 'resnet50':
            model = torchvision.models.resnet50()
        else:
            raise KeyError('invalid resnet name: {}'.format(name))

        return model

    def forward(self, x):
        # representation denoted as h in the paper
        h = self.encoder(x)

        # projection denoted as z in the paper
        z = self.projector(h)

        if self.config.simclr.model.normalize:
            z = nn.functional.normalize(z, dim=1)

        return h, z
