from collections import OrderedDict
import torchvision.models as models
import torch
import torch.nn as nn


def TotalLogLoss(predicted, labels):
    """
        Take the binary log loss for each predicted label and sum them.
        This results in the equivalent of training multiple binary classifiers, a useful formulation
        for our multi-label problem.
    """

    labels = labels.bool()  # for logical indexing

    # take complement of classes that are not present
    predicted[~labels] = 1 - predicted[~labels]

    # take negative log and sum
    negative_log = -torch.log(predicted)

    return negative_log.sum()


class ImageEmbedder(nn.Module):
    """
        I need a thing to take images and spit out vector embeddings that encode semantics.
        This... this'll do, I guess.
    """

    def __init__(self, embedding_dim):
        super(ImageEmbedder, self).__init__()

        pretrained = models.resnet18(pretrained=True)
        for param in pretrained.parameters():
            # freeze pretrained weights
            param.requires_grad = False
        pretrained.fc = nn.Linear(512, 512, bias=True)  # fine-tuned final layer, w/ gradients on

        embedder_operations = OrderedDict(
            {"resnet": pretrained, "relu1": nn.ReLU(inplace=True), "fc2": nn.Linear(512, embedding_dim, bias=True)}
        )
        self.network = nn.Sequential(embedder_operations)

    def forward(self, X):
        return self.network(X)


class SequenceClassifier(nn.Module):
    """
        Given a sequence of image vectors, intelligently weight the importance of each member
        of the sequence and use it to predict presence/absence of a class.
    """

    def __init__(self, seq_len, in_dim, classes):
        super(SequenceClassifier, self).__init__()

        selector_operations = OrderedDict(
            {
                "linear1": nn.Linear(in_dim, 128, seq_len),
                "relu": nn.ReLU(inplace=True),
                "linear2": nn.Linear(128, 1, seq_len),
                "sigmoid": nn.Sigmoid(),
            }
        )
        self.selector = nn.Sequential(selector_operations)

        predictor_operations = OrderedDict(
            {
                "linear1": nn.Linear(in_dim, in_dim),
                "relu": nn.ReLU(inplace=True),
                "linear2": nn.Linear(in_dim, classes),
                "sigmoid": nn.Sigmoid(),  # not a softmax--we have a multitarget problem
            }
        )
        self.predictor = nn.Sequential(predictor_operations)

    def forward(self, X):

        # get the weighted mean vector representation of the embedding sequence

        selector_vector = self.selector(X)
        selected = X * selector_vector
        selected = selected.mean(axis=0)

        # use the weighted mean vector for final prediction step

        predicted = self.predictor(selected)

        return predicted
