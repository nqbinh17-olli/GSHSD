#%%
import torch
from torch import nn
from model.layers.FFN import FFN
from model.layers.attention import SelfAttention


# %%
class CrossAttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768) -> None:
        super(CrossAttentionPooling, self).__init__()
       
        # self.FFN1= FFN(in_size)
        self.Attention = SelfAttention(in_size=in_size, hidden_size=in_size)
        # self.FFN2= FFN(in_size)
        # self.layer_norm_AT = nn.LayerNorm(in_size)

    def forward(self, cls_features, seq_features):
        cls_ctx = cls_features.unsqueeze(dim=1)
        out = self.Attention(cls_ctx, seq_features, seq_features)
        return out

# %%
class SqueezeAttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768, squeeze_factor: int = 6) -> None:
        super(SqueezeAttentionPooling, self).__init__()
        scaled_size = int(in_size//squeeze_factor)
        self.squeeze = nn.Linear(in_size, scaled_size)
        self.excitation = nn.Linear(scaled_size, in_size)
        self.layer_norm_SE = nn.LayerNorm(in_size)

        self.FFN1= FFN(in_size)
        self.Attention = SelfAttention(in_size=in_size, hidden_size=in_size)
        self.FFN2= FFN(in_size)
        self.layer_norm_AT = nn.LayerNorm(in_size)
        
        nn.init.xavier_normal_(self.squeeze.weight)
        nn.init.constant_(self.squeeze.bias, 0)
        
        nn.init.xavier_normal_(self.squeeze.weight)
        nn.init.constant_(self.excitation.bias, 0)


    def forward(self, features):
        #attention by channels
        global_infor = features.mean(dim=1) #sum by seq_len dimension
        squeezed_infor = self.squeeze(global_infor)
        chanel_dim_attention = torch.sigmoid(self.excitation(squeezed_infor)) # [batch, dim]
        sq_ex_out =  torch.einsum('abc,ac->abc', features, chanel_dim_attention)# features*chanel_dim_attetion.unsqueeze(dim=1)
        sq_ex_out = self.layer_norm_SE(sq_ex_out + features)
        

        x = self.FFN1(sq_ex_out, weights_factor=0.5)
        cls_ctx = x[:,0,:].unsqueeze(dim=1)
        out = x + self.Attention(x, cls_ctx, cls_ctx)
        final_emb = self.FFN2(out, weights_factor=0.5)
        context_vector = self.layer_norm_AT(final_emb)
        return context_vector.mean(dim=1)

# %%
class SelectionPooling(nn.Module):
    def __init__(self, in_size: int = 768) -> None:
        super(SelectionPooling, self).__init__()
        self.seq_weight_agg = nn.Linear(in_size, 1, bias=True)
        nn.init.xavier_normal_(self.seq_weight_agg.weight)
        nn.init.constant_(self.seq_weight_agg.bias, 0)

    def forward(self, features):
        attention = torch.sigmoid(self.seq_weight_agg(features)) # [batch, dim]
        context_vector = features*attention
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

