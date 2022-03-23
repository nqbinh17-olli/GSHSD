import torch
from torch import nn

# %%
class FFN(nn.Module):
    def __init__(self, in_size: int = 768, excitation_factor: int = 4, activation = torch.relu) -> None:
        super(FFN, self).__init__()
        scaled_size = int(in_size*excitation_factor)
        self.scale_linear = nn.Linear(in_size, scaled_size, bias=True)
        self.scale_out_linear = nn.Linear(scaled_size, in_size)
        self.activation = activation
        
        nn.init.xavier_normal_(self.scale_linear.weight)
        nn.init.constant_(self.scale_linear.bias, 0)
        nn.init.xavier_normal_(self.scale_out_linear.weight)
        nn.init.constant_(self.scale_out_linear.bias, 0)

    def forward(self, features, weights_factor:float = 1.0):
        x = self.scale_linear(features)
        x = self.activation(x)
        x = self.scale_out_linear(x)
        return weights_factor*x + features
        