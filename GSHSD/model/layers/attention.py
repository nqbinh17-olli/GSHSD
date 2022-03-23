from torch import nn

from model.operators import dot_attention_function


class SelfAttention(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 768*2) -> None:
        super(SelfAttention, self).__init__()
        self.K = nn.Linear(in_size , hidden_size)
        self.Q = nn.Linear(in_size , hidden_size)
        self.V = nn.Linear(in_size , hidden_size)
        
        nn.init.xavier_normal_(self.K.weight)
        nn.init.constant_(self.K.bias, 0)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.constant_(self.V.bias, 0)

    def forward(self, q, k, v):
        # attention by [CLS] query in the rest vectors
        seq_q = self.Q(q) #[:,1:,:]
        cls_k = self.K(k)
        cls_v = self.V(v)
        out = dot_attention_function(q=seq_q, k=cls_k, v=cls_v)
        return out