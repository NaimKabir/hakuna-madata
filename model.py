from collections import OrderedDict
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CUDA_AVAILABLE = torch.cuda.is_available()


def TotalLogLoss(predicted, labels):
    """
        Take the binary log loss for each predicted label and sum them.
        This results in the equivalent of training multiple binary classifiers, a useful formulation
        for our multi-label problem.
    """

    labels = labels.bool()  # for logical indexing

    # take complement of classes that are not present,
    # in a way that maintains the computation graph

    subtracted_from = (~labels).float()
    sign_indicator = labels.float() * 2 - 1  # [0,1] -> [-1, 1]
    signed = sign_indicator * predicted
    complemented = subtracted_from + signed

    # take negative log and sum
    negative_log = -torch.log(complemented)

    return negative_log.sum()


def HingeLoss(predicted, labels):

    # multilabel margin loss is straight-up stupid in its implementation
    # it requires labels to be class indices, padded with negative int values

    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)
        predicted = predicted.unsqueeze(0)

    idxes = torch.ones(labels.shape) * -1  # preallocate space for labels
    if CUDA_AVAILABLE:
        idxes = idxes.cuda()

    nonzeros = labels.nonzero()
    for row in range(labels.shape[0]):
        row_idx = nonzeros[:, 0] == row
        nonzero_idxes = nonzeros[row_idx, 1]
        idxes[row, : len(nonzero_idxes)] = nonzero_idxes
    loss_func = nn.MultiLabelMarginLoss(reduction="sum")
    hinge_loss = loss_func(predicted, idxes.long())

    return hinge_loss


class ImageEmbedder(nn.Module):
    """
        I need a thing to take images and spit out vector embeddings that encode semantics.
        This... this'll do, I guess.
    """

    def __init__(self, embedding_dim):
        super(ImageEmbedder, self).__init__()

        pretrained = models.resnet18(pretrained=True)
        # for param in pretrained.parameters():
        #     # freeze pretrained weights
        #     param.requires_grad = False
        pretrained.fc = nn.Linear(512, 512, bias=True)  # fine-tuned final layer, w/ gradients on

        embedder_operations = OrderedDict(
            {
                "resnet": pretrained,
                "relu1": nn.ReLU(inplace=True),
                "fc2": nn.Linear(512, embedding_dim, bias=True),
                "tanh": nn.Tanh(),
            }
        )
        self.network = nn.Sequential(embedder_operations)

    def forward(self, X):
        return self.network(X)

class SequenceClassifier(nn.Module):
    """
        Given a sequence of image vectors, intelligently weight the importance of each member
        of the sequence and use it to predict presence/absence of a class.
    """

    def __init__(self, in_dim, hidden_dim, num_layers, classes):
        super(SequenceClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_dim, num_layers=num_layers)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu1 = nn.ReLU6(inplace=True)
        self.linear2 = nn.Linear(self.hidden_dim, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, seq_len):

        hidden, _ = self.lstm(X.view(seq_len, 1, -1))
        mapped = self.relu1(self.linear1(hidden.view(seq_len, -1)))
        probabilities = self.sigmoid(self.linear2(mapped))

        return probabilities

class ImageSequenceClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, classes):
        super(ImageSequenceClassifier, self).__init__()

        network_operations = OrderedDict(
            {
                "embedder": ImageEmbedder(embedding_dim),
                "classifier": SequenceClassifier(embedding_dim, hidden_dim, num_layers, classes),
            }
        )
        self.network = nn.Sequential(network_operations)

    def forward(self, X):
        return self.network(X)
