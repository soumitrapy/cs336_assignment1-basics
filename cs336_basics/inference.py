import torch
from torch import Tensor
from torch.nn import Module

from .bpe_tokenizer import BPETokenizer
from .nn.functional import softmax

def decoding(
        prompt: str,
        model: torch.nn.Module,
        tokenizer: BPETokenizer,
        tau: float = 1.0,
        p: float = 1.0,
        max_new_tokens: int = 100,
        device: str = "cpu",
    ) -> str:
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)
    was_training = model.training
    model.eval()
    i = 0
    with torch.no_grad():
        while i < max_new_tokens:
            logits = model(x) # (1, seq_len, vocab_size)
            logits = logits[:, -1, :] # (1, vocab_size)
            probs = softmax(logits, dim=-1, tau=tau) # (1, vocab_size)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs, p)
            cutoff_index = torch.clamp(cutoff_index, min=0, max=probs.shape[-1]-1)
            cutoff_prob = sorted_probs[0, cutoff_index]
            filtered_probs = torch.where(probs >= cutoff_prob, probs, torch.tensor(0.0, device=probs.device))
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
            next_token_id = torch.multinomial(filtered_probs, num_samples=1) # (1,)
            x = torch.cat([x, next_token_id.unsqueeze(0)], dim=-1) # (1, seq_len+1)
            i += 1
    model.train(was_training)
    output_ids = x.squeeze(0).tolist()
    output_text = tokenizer.decode(output_ids)
    return output_text

               

