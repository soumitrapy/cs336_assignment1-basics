from jaxtyping import Float, Bool
from einops import einsum

import torch
from torch import Tensor

def softmax(x: Float[Tensor, "..."],
            dim: int = -1) -> Float[Tensor, "..."]:
    y = x-x.max(dim=dim, keepdim=True).values
    exp = torch.exp(y)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

# def scaled_dot_product_attention(q: Float[Tensor, "... seq_len d_k"],
#                                  k: Float[Tensor, "... seq_len d_k"],
#                                  v: Float[Tensor, "... seq_len d_v"],
#                                  mask: Bool[Tensor, "seq_len seq_len"] = None,
#                                  ) -> Float[Tensor, "... seq_len d_v"]:

#     d_k = q.shape[-1]
#     scores =  einsum(q, k, "... seq_len d_k, ... seq_len d_k -> ... seq_len seq_len") / (d_k ** 0.5)

#     if mask is not None:
#         scores = scores.masked_fill(~mask, float('-inf'))
    
#     attn_weights = softmax(scores, dim=-1)
#     output = einsum(attn_weights, v, "... seq_len seq_len, ... seq_len d_v -> ... seq_len d_v")
#     return output

def scaled_dot_product_attention(q: Float[Tensor, "... query_seq_len d_k"],
                                 k: Float[Tensor, "... key_seq_len d_k"],
                                 v: Float[Tensor, "... key_seq_len d_v"],
                                 mask: Bool[Tensor, "query_seq_len key_seq_len"] = None,
                                 ) -> Float[Tensor, "... query_seq_len d_v"]:

    d_k = q.shape[-1]
    scores =  einsum(q, k, "... query_seq_len d_k, ... key_seq_len d_k -> ... query_seq_len key_seq_len") / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, v, "... query_seq_len key_seq_len, ... key_seq_len d_v -> ... query_seq_len d_v")
    return output