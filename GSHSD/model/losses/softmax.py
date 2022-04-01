import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self, num_class, num_feature):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_class, num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weights, gamma=1.1):
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        targets = targets.squeeze()

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.weights[targets]*(1-pt)**self.gamma * BCE_loss

        return F_loss.mean()
