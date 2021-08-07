import torch
import torch.nn as nn


class NTXent(nn.Module):
    """
    Implementation of the normalized temperature-scaled cross entropy loss accordingly the proposed loss in the
    paper https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, batch_size, temperature, device):
        super(NTXent, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.calculate_correlated_mask(batch_size)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_function = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        representations = torch.cat((z_i, z_j), dim=0)

        # create a similarity matrix
        similarity_matrix = self.similarity_function(representations.unsqueeze(1),
                                                     representations.unsqueeze(0)) / self.temperature

        # filter positive scores
        sim_i_j = torch.diag(similarity_matrix, self.batch_size)
        sim_j_i = torch.diag(similarity_matrix, -self.batch_size)

        # create the positive samples
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            2 * self.batch_size, 1
        )

        # treat the other augmented examples within a minibatch as negative examples
        negative_samples = similarity_matrix[self.mask].reshape(2 * self.batch_size, -1)

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        # adjustment of the loss based on the batch size
        loss /= 2 * self.batch_size

        return loss

    @staticmethod
    def calculate_correlated_mask(batch_size):

        # prepare the prefilled mask
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)

        # adjust the mask content to the batch_size
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask
