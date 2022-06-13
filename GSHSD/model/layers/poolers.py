#%%
from matplotlib import scale
import torch
from torch import nn
from model.layers.FFN import FFN
from model.layers.attention import SelfAttention


# %%
class CrossAttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768) -> None:
        super(CrossAttentionPooling, self).__init__()
       
        self.FFN1= FFN(in_size)
        self.Attention = SelfAttention(in_size=in_size, hidden_size=in_size, head_num=8)
        self.FFN2= FFN(in_size)
        self.activation = nn.functional.silu
        # self.layer_norm_AT = nn.LayerNorm(in_size)

    def forward(self, cls_features, seq_features):
        cls_features = cls_features.unsqueeze(dim=1)
        seq_features = self.FFN1(seq_features, weights_factor=0.5)
        out = self.Attention(cls_features, seq_features, seq_features)
        out = self.activation(out.squeeze())
        out = self.FFN2(out, weights_factor=0.5)
        return out

# %%
# class SlotsAttentionPooling(nn.Module):
#     def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
#         super().__init__()
#         self.W = nn.Linear(in_size, hidden_size)
#         self.V = nn.Linear(hidden_size, 1)

#         nn.init.xavier_normal_(self.W.weight)
#         nn.init.constant_(self.W.bias, 0)
#         nn.init.xavier_normal_(self.V.weight)
#         nn.init.constant_(self.V.bias, 0)
        
#     def forward(self, features):
#         att = torch.tanh(self.W(features))
#         score = self.V(att) # [batch, seq_len, 1]
#         attention_weights = torch.softmax(score, dim=1)
#         context_vector = attention_weights * features
#         context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
#         return context_vector



class AttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.heads = 8
        assert in_size % self.heads == 0 
        assert hidden_size % self.heads == 0 
        self.W_Q = nn.Linear(in_size, hidden_size)
        self.W_K = nn.Linear(in_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, in_size)
        self.scale = in_size ** -0.5
        self.V = nn.Linear(hidden_size // self.heads, 1)
        self.activation_dropout = nn.Dropout(0.1)
        self.softmax_dropout = nn.Dropout(0.1)

        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_out.weight)
        nn.init.xavier_normal_(self.V.weight)
        
    # def ori_forward(self, features):
    #     residual = features
    #     features = features * self.scale

    #     att = torch.tanh(self.W(features))
    #     att = self.activation_dropout(att)
    #     score = self.V(att) # [batch, seq_len, 1]

    #     attention_weights = torch.softmax(score, dim=1)
    #     context_vector = attention_weights * residual
    #     context_vector = self.W_out(context_vector)
    #     context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
    #     return context_vector

    def forward(self, features):
        # Multi-head & Self Attention Style Implementation
        #features = features * self.scale
        features = features[:,1:,:] # remove CLS token
        batch, seq_len, _ = features.shape
        Q = self.W_Q(features).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        K = self.W_K(features).view(batch, seq_len, self.heads, -1).transpose(1, 2)

        att = torch.tanh(Q) # [batch, heads, seq_len, head_dim]
        #att = self.activation_dropout(att)
        score = self.V(att) # [batch, heads, seq_len, 1]

        attention_weights = torch.softmax(score, dim=2)
        attention_weights = self.softmax_dropout(attention_weights)
        context_vector = attention_weights * K # [batch, heads, seq_len, head_dim]
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
        context_vector = self.W_out(context_vector)
        return context_vector      

class TaskBasedPooling(nn.Module):
    def __init__(self, in_size, knowledge_kernels = 3, heads = 12):
        super(TaskBasedPooling, self).__init__()
        self.fc_in = nn.Linear(in_size, in_size)
        self.layer_norm = nn.LayerNorm(in_size)
        self.W_knowledge = nn.Parameter(torch.Tensor(knowledge_kernels, in_size))
        self.P_knowledge = nn.Parameter(torch.Tensor(knowledge_kernels))

    def xavier_init(self, layer):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(self, features):
        sent_embed = features[:,0,:] # CLS embedding as sentence embedding
        features = features[:,1:,:] # remove CLS

        features = self.layer_norm(features)
        features = self.fc_in(features) # (batch_size, seq_len, dim)

        knowledge_based = torch.matmul(self.W_knowledge.transpose(0, 1), torch.sigmoid(self.P_knowledge)) # (dim)
        knowledge_based = knowledge_based.unsqueeze(0).unsqueeze(0).expand(features.size(0), -1, -1) # (batch_size, 1, dim)
        attn_knowledge = torch.softmax(features @ knowledge_based.transpose(1, 2)) # (batch_size, seq_len, 1)
        # attention score based on Knowledge
        knowledge_based_sent_embed = torch.sum(attn_knowledge * features, dim = 1)
        return knowledge_based_sent_embed

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