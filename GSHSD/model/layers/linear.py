import math
import torch
from torch import nn
from torch import Tensor

def scale_xavier_uniform_(tensor: Tensor, scale: int, gain: float = 1.) -> Tensor:
    r"""
    1. Copy from: https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_
    2. Idea from: https://arxiv.org/pdf/1910.05895.pdf (Transformers without Tears:
    Improving the Normalization of Self-Attention)
    3. Intuitive: Xavier normal yields initial weights that are too large. 
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + scale * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return torch.nn.init._no_grad_uniform_(tensor, -a, a)

def init_weights(m, scale_xavier):
    if type(m) == nn.Linear:
        scale_xavier_uniform_(m.weight, scale_xavier)
        #torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          m.bias.data.fill_(0.01) # fill with value that is closer to zero is better
    return

class Linear_(nn.Module):
    def __init__(self, in_size: int, out_size: int, scale_xavier: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        init_weights(self.linear, scale_xavier)

    def forward(self, x):
        return self.linear(x)

class FixNormLinear(nn.Module):
    """
    1. Paper: https://aclanthology.org/N18-1031.pdf
    2. The problem is: weight W favors the common words or classes appear during training time.
    3. We have a result: Wx = ||W|| ||x|| * cos(W, x). And cos(W, x) is unlikely to have this problem. 
    """
    def __init__(self, in_size: int, out_size: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(in_size, out_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        if bias is True:
            torch.nn.init.xavier_uniform_(self.weights)
            self.bias.data.fill_(0.01)
        else:
            self.bias = None
        self.g_scale = math.sqrt(in_size)
        self.eps = eps

    def forward(self, x):
        w_norm = 1.0 / torch.norm(self.weights, dim=-1, keepdim=True).clamp(min=self.eps)
        x_norm = 1.0 / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = self.g_scale * torch.matmul(x, self.weights) * w_norm * x_norm
        if self.bias is not None:
            x += self.bias
        return x