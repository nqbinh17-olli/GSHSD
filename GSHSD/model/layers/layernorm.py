import torch
from torch import nn


class ScaleNorm(nn.Module):
    """
    1. Paper: https://arxiv.org/pdf/1910.05895.pdf
    2. scale = sqrt(d), d = dimension
    """
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm