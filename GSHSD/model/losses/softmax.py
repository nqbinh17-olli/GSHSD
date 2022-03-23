import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_class, num_feature):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_class, num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss
