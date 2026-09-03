from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange

import torch
from torch import Tensor

def softmax(x: Float[Tensor, "..."],
            dim: int = -1) -> Float[Tensor, "..."]:
    y = x-x.max(dim=dim, keepdim=True).values
    exp = torch.exp(y)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

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

def log_softmax(x: Float[Tensor, "..."],
                dim: int = -1) -> Float[Tensor, "..."]:
    y = x-x.max(dim=dim, keepdim=True).values
    exp = torch.exp(y)
    return y - torch.log(torch.sum(exp, dim=dim, keepdim=True))

def logsumexp(x: Float[Tensor, "..."],
              dim: int = -1,
              keepdim: bool = False) -> Float[Tensor, "..."]:
    y = x-x.max(dim=dim, keepdim=True).values
    exp = torch.exp(y)
    return torch.log(torch.sum(exp, dim=dim, keepdim=keepdim)) + x.max(dim=dim, keepdim=keepdim).values

def cross_entropy_loss(logits: Float[Tensor, "... num_classes"],
                       targets: Int[Tensor, "..."],
)-> Float[Tensor, "..."]:
    targets_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # shape (...), logits of the target class
    return -targets_logits + logsumexp(logits, dim=-1) # shape (...), cross entropy loss for each sample

def perplexity(logits: Float[Tensor, "... seq_len num_classes"],
               targets: Int[Tensor, "... seq_len"],
) -> Float[Tensor, "..."]:
    loss = cross_entropy_loss(logits, targets) # shape (... , seq_len), cross entropy loss for each sample
    return torch.exp(loss.mean(dim=-1))