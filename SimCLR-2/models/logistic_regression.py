import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Simple logistic regression implementation. Same as a linear regression without further processing.
    nn.CrossEntropyLoss() computes softmax internally and therefore is used to transform
    the linear output of this module
    """

    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
