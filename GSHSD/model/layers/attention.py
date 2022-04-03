from torch import nn

from model.operators import dot_attention_function


class SelfAttention(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 768*2, head_num:int = 8) -> None:
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.K = nn.Linear(in_size , hidden_size)
        self.Q = nn.Linear(in_size , hidden_size)
        self.V = nn.Linear(in_size , hidden_size)
        self.O = nn.Linear(hidden_size , in_size)
        self.__init_weights()
    
    def __init_weights(self):
        nn.init.xavier_normal_(self.K.weight)
        nn.init.constant_(self.K.bias, 0)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.constant_(self.V.bias, 0)
        nn.init.xavier_normal_(self.O.weight)
        nn.init.constant_(self.O.bias, 0)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

    def forward(self, q, k, v):
        # attention by [CLS] query in the rest vectors
        seq_q = self.Q(q) #[:,1:,:]
        cls_k = self.K(k)
        cls_v = self.V(v)

        seq_q = self._reshape_to_batches(seq_q)
        cls_k = self._reshape_to_batches(cls_k)
        cls_v = self._reshape_to_batches(cls_v)

        out = dot_attention_function(q=seq_q, k=cls_k, v=cls_v)
        out = self._reshape_from_batches(out)

        return self.O(out)
