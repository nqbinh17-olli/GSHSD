import torch
from torch import nn

class ConvBlock(nn.Module):
    """
    padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default="zeros"    
    """
    def __init__(self, d_model, hidden_size, stride=1):
        super(ConvBlock, self).__init__()
        
        # Convolution layers definition
        self.conv_1 = nn.Conv1d(d_model, hidden_size, 1, stride)
        self.conv_2 = nn.Conv1d(d_model, hidden_size, 2, stride)
        self.conv_3 = nn.Conv1d(d_model, hidden_size, 3, stride)
        self.conv_4 = nn.Conv1d(d_model, hidden_size, 4, stride)
        # Activation
        self.relu = nn.functional.silu
        # Max pooling layers definition
        self.local_pool = nn.MaxPool1d(3, 3)
        
        self.__init_model_params()

    def __init_model_params(self):
        self.__init_params(self.conv_1, bias=True)
        self.__init_params(self.conv_2, bias=True)
        self.__init_params(self.conv_3, bias=True)
        self.__init_params(self.conv_4, bias=True)

    def __init_params(self, layer, bias = True):
        nn.init.xavier_normal_(layer.weight)
        if bias:
            nn.init.constant_(layer.bias, 0)
       
    def forward(self, embeddings, chanel_last=True):

        embeddings_ = embeddings.transpose(1, 2).contiguous() if chanel_last else embeddings

        conv_1_out = self.relu(self.conv_1(embeddings_))
        conv_1_out = self.local_pool(conv_1_out)
        conv_2_out = self.relu(self.conv_2(embeddings_))
        conv_2_out = self.local_pool(conv_2_out)
        conv_3_out = self.relu(self.conv_3(embeddings_))
        conv_3_out = self.local_pool(conv_3_out)
        conv_4_out = self.relu(self.conv_4(embeddings_))
        conv_4_out = self.local_pool(conv_4_out)

        conv_out = torch.cat([conv_1_out, conv_2_out, conv_3_out, conv_4_out], dim=-1)
        conv_out = conv_out.transpose(1, 2).contiguous() if chanel_last else conv_out

        return conv_out