import torch
import torch.nn.functional as F

def dot_attention_function(q, k, v):
    """
     q: [bs, length, dim] or [bs, res_cnt, dim]
     k=v: [bs, length, dim] or [bs, m_params, dim]
    """
    attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, length, length] or [bs, res_cnt, m_params]
    attn_weights = F.softmax(attn_weights, -1)
    output = torch.matmul(attn_weights, v) # [bs, length, dim] or [bs, res_cnt, dim]
    return output

# distances functions
def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    ((bs,k,dim)*(bs,k,dim)).sum(dim_axis) == (bs,dim).dot((bs,dim).T) IF AND ONLY IF `k==1`.
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
    b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b
    assert len(a.shape) in [2,3] and len(b.shape) in [2,3]
    
    if len(a.shape) == 3:
        if len(b.shape) == 2:
            b = b.unsqueeze(dim=1)
        bs,_,h = b.shape #res always 1 while training
        b = b.permute(1, 0, 2) # [1, bs, dim]
        b = b.expand(bs, bs, h)
        return (a * b).sum(dim=-1)    
    elif len(a.shape) == 2:
        if len(b.shape) == 3:
            b = b.squeeze()
        return a.matmul(b.T)

    
def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def subtraction(a, b, p=2, dim=-1, pairwise=True):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    return torch.norm(a.unsqueeze(0) - b.unsqueeze(1), p=p, dim=dim) if pairwise \
           else torch.norm(a - b, p=p, dim=dim)
