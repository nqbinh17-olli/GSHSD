import torch
from torch import nn

from model.layers.FFN import FFN
from model.layers.attention import SelfAttention
from model.layers.linear import Linear_, init_weights


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


class AttentionPoolingv1(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = Linear_(in_size, hidden_size)
        self.V = Linear_(hidden_size, 1)
        
    def forward(self, features):
        # Attention Score focus on 'CLS' token more than other tokens.
        att = torch.tanh(self.W(features))
        score = self.V(att) # [batch, seq_len, 1]
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
        return context_vector

class AttentionPooling(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.heads = 8
        assert in_size % self.heads == 0 
        assert hidden_size % self.heads == 0 
        self.W_Q = Linear_(in_size, hidden_size)
        self.W_K = Linear_(in_size, hidden_size)
        self.W_out = Linear_(hidden_size, in_size)
        self.scale = in_size ** -0.5
        self.V = Linear_(hidden_size // self.heads, 1)
        self.activation_dropout = nn.Dropout(0.1)
        self.softmax_dropout = nn.Dropout(0.1)

    def forward(self, features):
        # Multi-head & Self Attention Style Implementation
        batch, seq_len, _ = features.shape
        Q = self.W_Q(features).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        K = self.W_K(features).view(batch, seq_len, self.heads, -1).transpose(1, 2)

        att = torch.tanh(Q) # [batch, heads, seq_len, head_dim]
        att = self.activation_dropout(att)
        score = self.V(att) # [batch, heads, seq_len, 1]

        attention_weights = torch.softmax(score, dim=2)
        attention_weights = self.softmax_dropout(attention_weights)
        context_vector = attention_weights * K # [batch, heads, seq_len, head_dim]
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        context_vector = torch.sum(context_vector, dim=1) # [batch, dim]
        context_vector = self.W_out(context_vector)
        return context_vector      

class TaskBasedPooling(nn.Module):
    def __init__(self, in_size, knowledge_kernels = 3, heads = 12, eps=1e9):
        super(TaskBasedPooling, self).__init__()
        self.heads = heads
        assert in_size % heads == 0
        self.head_dim = in_size // heads
        self.fc_key = Linear_(in_size, in_size)
        self.fc_value = Linear_(in_size, in_size)
        self.fc_query = Linear_(in_size, in_size)
        self.fc_out = Linear_(in_size, in_size)
        self.layer_norm = nn.LayerNorm(in_size)
        self.W_knowledge = nn.Parameter(torch.Tensor(knowledge_kernels, in_size))
        self.P_knowledge = nn.Parameter(torch.Tensor(knowledge_kernels, 1))
        self.dropout_attn = nn.Dropout(0.1)

        torch.nn.init.xavier_uniform_(self.W_knowledge)
        torch.nn.init.xavier_uniform_(self.P_knowledge)
        self.eps = eps

    def forward(self, features, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)

        features = self.layer_norm(features)
        batch_size, seq_len, _ = features.shape
        Key = self.fc_key(features).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2) 
        # (batch_size, heads, seq_len, head_dim)
        Value = self.fc_value(features).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        # (batch_size, heads, seq_len, head_dim)

        knowledge_based = torch.matmul(self.W_knowledge.transpose(0, 1), torch.sigmoid(self.P_knowledge)).squeeze() # (dim)
        knowledge_based = knowledge_based.unsqueeze(0).unsqueeze(0).expand(Key.size(0), -1, -1) # (batch_size, 1, heads, head_dim)
        Query = self.fc_query(knowledge_based).view(batch_size, 1, self.heads, self.head_dim).transpose(1, 2)

        attn_knowledge = torch.matmul(Key, Query.transpose(-1, -2)) / self.head_dim ** -0.5 # (batch_size, heads, seq_len, 1)
        attn_knowledge = attn_knowledge.masked_fill(attention_mask == 0, -self.eps)
        attn_knowledge = torch.softmax(attn_knowledge, dim=-1)
        # attention score based on Knowledge
        attn_knowledge = self.dropout_attn(attn_knowledge)
        knowledge_based_sent_embed = torch.mean(attn_knowledge * Value, dim = 2)
        knowledge_based_sent_embed = knowledge_based_sent_embed.view(batch_size, -1)
        return self.fc_out(knowledge_based_sent_embed)

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