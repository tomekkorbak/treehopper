from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable as Var

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def sentiment_accuracy_score(self, predictions, labels):
        labels = torch.FloatTensor(labels)
        correct = (predictions==labels).sum()
        total = labels.size(0)
        acc = float(correct)/total
        return acc