# Problem (unicode1): Understanding Unicode (1 point)
- a. Null Character ['\x00', as it is non printable character Null]
- b. Printed value is acutally the value of the character which Null or '\x00'. but the representation is actuall the string literal representation of the character itself so it is as ""\\x00'"
- c. It will be a repplace by the non pritable character '\x00'. So the output of "this is a test" + chr(0) + "string" will be: "this is a test\x00string"

# Problem (unicode2): Unicode Encodings (3 points)
- a. UTF-8 is compact (space efficient), ASCII-compatible, widely used in practice.
- b. the decode function decodes each byte individually from each byte string, it will work for ASCII chartacters as these characters only converts into one byte for any Non-ASCII charatters it is transferred into multiple bytes, so for those case individual byte does not represent any valid characters so for any non-ASCII characters we will get error.
- c. any 2-byte with not of the from 110xxxxx 10xxxxxx

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
- a. Time: 182 sec (don't use tracemalloc during time computation), Memory: 28 MB (peak)
longest token in vocab: b' accomplishment'
- b. Merging is taking much time.

# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
- a. longest token: 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'
- b. a lot of new tokens are added.

# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
- a. Average compression ratio for TinyStories tokenizer: 4.11 bytes/token
     Average compression ratio for owt tokenizer: 4.43 bytes/token
- b. Average compression ratio for TinyStories tokenizer on owt data: 3.26 bytes/token. As tinystories tokenizer have not seen more general data set like owt, so the compression will be lower compared to the owt trained tokenizer.
- c. Average throughput for owt tokenizer on owt data: 414148.82 bytes/second or 0.41 MB/second
To process 852 GB data, it would take 571.45 hours
- d.as uint16 can represent total 2^16 unsigned numbers. and my vocab has only 32k id, so it is sufficient for my use case.

# Problem (transformer_accounting): Transformer LM resource accounting (5 points)
| Module | shapes | #parameters | #FLOPS | O(FLOPS) |
|:------ |:------|:------|:------|:------|
| Embedding | ... -> ... d_model | vocab_size.d_model | 0 |
| Rope | ... seq_len d_k -> ... seq_len d_k | 0 | 3.d_k.seq_len |
| Softmax | ... d -> ... d | 0 | 3.d |
| Linear | ... in_features -> ... out_features | out_features.in_features | (2.in_features-1).out_features |
| RMSNorm | ... d_model -> ... d_model | d_model | 3.d_model |
| sigmoid | ... d -> ... d | 0 | 4d |
| SiLU | ... d -> ... d | 0 | 5d |
| SwiGLU | ... d_model -> ... d_model | 3.d_model.d_ff | 6.d_model.d_ff + 6.d_ff |
|scaled_dot_prod_attention (d_k=d_v) | ... seq_len d_k, ... seq_len d_k,  ... seq_len d_k -> ... seq_len d_k | 0 | 4.seq_len<sup>2</sup>.d_k + 3.seq_len<sup>2</sup> | 4.seq_len<sup>2</sup>.d_k |
| MultiheadedAttention (with rope) | ... seq_len d_model -> ... seq_len d_model | 4.d_model.(num_heads.d_k) | **8.seq_len.d_model.(num_heads.d_k)**[qkv_proj+out_proj] + **4.seq_len<sup>2</sup>.(num_heads.d_k)** + 3.seq_len<sup>2</sup>.num_heads [scaled_dot_product] + 6.seq_len.(num_heads.d_k)[rope] [if num_heads.d_k = d_model: 8.seq_len.d_model<sup>2</sup> + 4.seq_len<sup>2</sup>.d_model + 6.seq_len.d_model + 3.seq_len<sup>2</sup>.num_heads |  8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) [ 8.seq_len.d_model<sup>2</sup> + 4.d_model.seq_len<sup>2</sup>] |
| TransformerBlock| ... seq_len d_model -> ... seq_len d_model | 4_d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model | [MultiheadAttention] + (**6.seq_len.d_model.d_ff** + 6.seq_len.d_ff)[FFN(SwiGLU)] + 2.seq_len.d_model | 8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) + 4.seq_len.d_ff [ 8.seq_len.d_model<sup>2</sup> + 4.d_model.seq_len<sup>2</sup>] + 4.seq_len.d_ff |
|TransformerLM| ... seq_len -> ... seq_len vocab_size | 2.vocab_size.d_model+d_model+num_layers.(4.d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model) | num_layers.[TransformerBlock]+ 3.seq_len.d_model + 2.seq_len.d_model.vocab_size | num_layers.(8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) + 4.seq_len.d_ff) + 2.seq_len.vocab_size |

- a. Number of Parameters = 2.vocab_size.d_model + d_model + num_layers.(4_d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model). Total parameters: 1640452800 ~ 1.6B, total parameters in GB (in fp32): 6.5618112 GB
- b.

| Matrix | FLOPS |
|:------ |:------|
| output_proj | 2.seq_len.d_model.vocab_size.(batch) |
|MultiHeadSelfAttention | **8.seq_len.d_model.(num_heads.d_k)**[qkv_proj+out_proj] + **4.seq_len<sup>2</sup>.(num_heads.d_k)** + 3.seq_len<sup>2</sup>.num_heads [scaled_dot_product] + 6.seq_len.(num_heads.d_k)[rope] |
|FFN (SwiGLu) | 6.d_model.d_ff.(batch) |

- c. In most of the cases FFN is taking most of the flops (~60%) followed by multi head attention (~30%)

- d. If I increase the context length, Boottleneck becomes the multihead attention (~73%)

# Problem (learning_rate_tuning): Tuning the learning rate (1 point)
It will depend.

# Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)
```python
def scaled_dot_product_attention(q: Float[Tensor, "... query_seq_len d_k"],
                                 k: Float[Tensor, "... key_seq_len d_k"],
                                 v: Float[Tensor, "... key_seq_len d_v"],
                                 mask: Bool[Tensor, "query_seq_len key_seq_len"] = None,
                                 ) -> Float[Tensor, "... query_seq_len d_v"]: ### Peak Memory: ... (seq_len.d_k + 2.seq_len^2) (excluding input variables)
    d_k = q.shape[-1]
    scores =  einsum(q, k, "... query_seq_len d_k, ... key_seq_len d_k -> ... query_seq_len key_seq_len") / (d_k ** 0.5) 
    # Memory: ... seq_len^2 (scores)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = softmax(scores, dim=-1) # Memory: ... seq_len^2 (scores) + ... seq_len^2 (attn_weights)
    output = einsum(attn_weights, v, "... query_seq_len key_seq_len, ... key_seq_len d_v -> ... query_seq_len d_v") 
    # Memory: ... seq_len^2 (scores) + ... seq_len^2 (attn_weights) + ... seq_len d_v (output)
    return output

class MultiheadSelfAttention(Module):
     def __init__(self, d_model: int, 
                 num_heads: int,
                 theta: float = 10000.0,
                 max_seq_len: int | None = None,
                 ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.qkv_proj = Linear(d_model, (self.d_k + self.d_k + self.d_v)*self.num_heads)
        self.out_proj = Linear(self.num_heads * self.d_v, d_model)
        self.rope = None
        if max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(base=theta, dim=self.d_k, max_seq_len=max_seq_len)

    def forward(self, 
                x: Float[Tensor, "... seq_len d_model"],
                token_positions: Int[Tensor, "... seq_len"] | None = None
                ) -> Float[Tensor, "... seq_len d_model"]: 
                # Peak Memory: ... 7.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask) + 2.seq_len^2.num_heads (or ...seq_len.d_model)
        proj = self.qkv_proj(x) # Memory: ... 3.seq_len.d_k.num_heads(proj)
        q, k, v = torch.split(proj, [self.num_heads*self.d_k, self.num_heads*self.d_k, self.num_heads*self.d_v], dim=-1) 
        # Memory: ... 6.seq_len.d_k.num_heads(proj+q,k,v)
        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        if self.rope is not None:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions) 
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0) # shape (seq_len, seq_len)
        # Memory: ... 6.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask)
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        # Memory: ... 6.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask) + ... seq_len.d_k.num_heads + 2.seq_len^2.num_heads (multihead_attention)
        out = self.out_proj(attn_out)
        # Memory: ... 6.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask) + ... seq_len.d_k.num_heads (attn_out) + ...seq_len.d_model (out)
        return out

class SiLU(Module):
    def forward(self,
                x: Float[Tensor,"... d_model"]
                ) -> Float[Tensor,"... d_model"]:
        return einsum(x, torch.sigmoid(x), "... d_model, ... d_model-> ... d_model")

class SwiGLU(Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int | None = None,
                 ) -> None:
        super().__init__()
        self.in_features = in_features
        if hidden_features is None:
            self.d_ff = max(round(in_features/(3*8)), 1)*64 # d_ff = 8/3 * d_model nearest multiple of 64
        else:
            self.d_ff = hidden_features
        self.linear1 = Linear(self.in_features, self.d_ff)
        self.linear3 = Linear(self.in_features, self.d_ff)
        self.linear2 = Linear(self.d_ff, self.in_features)
        self.silu = SiLU()

    def forward(self,
                x: Float[Tensor,"... d_model"]
                ) -> Float[Tensor,"... d_model"]:
        x1 = self.silu(self.linear1(x))
        x3 = self.linear3(x)
        x2 = self.linear2(einsum(x1, x3, "... d_ff, ... d_ff -> ... d_ff"))
        return x2

class TransformerBlock(Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 theta: float = 10000.0, 
                 max_seq_len: int | None = None,
                 ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(in_features=d_model, hidden_features=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]: # Peak Memory: ... 7.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask) + 2.seq_len^2.num_heads (or ...seq_len.d_model)
        attn_out = self.attn(self.ln1(x), token_positions) # Memory: ... 7.seq_len d_k.num_heads(proj+q,k,v) + seq_len^2 (mask) + 2.seq_len^2.num_heads (or ...seq_len.d_model)
        x = x + attn_out # Memory: ...seq_len.d_model
        ffn_out = self.ffn(self.ln2(x)) # ...seq_len.d_model
        x = x + ffn_out # Memory: ...seq_len.d_model
        return x


class TransformerLM(Module):
    def __init__(self, 
                 vocab_size: int,
                 context_len: int,
                 num_layers: int,
                 d_model: int = 2**13,
                 num_heads: int = 16,
                 d_ff: int = 2**15,
                 rope_theta: float = 10000.0,
                 ) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_len) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.embedding(x)                   # shape (..., seq_len, d_model)
        for mha in self.layers:
            x = mha(x, token_positions=None)    # shape (..., seq_len, d_model)
        x = self.final_norm(x)                  # shape (..., seq_len, d_model)
        logits = self.output_proj(x)            # shape (..., seq_len, vocab_size)
        #logits = softmax(logits, dim=-1)
        return logits
```




- a.
| Parameters | 2.vocab_size.d_model+d_model+num_layers.(12.d_model<sup>2</sup> + 2.d_model) | [d_model = num_heads.d_k, d_ff = 8/3.d_model]
| Gradients | #Parameters |
| Optimizer State | 2.#Parameters |
| Activations | batch_size.7.seq_len d_model + seq_len^2 + 2.seq_len^2.num_heads (or batch.seq_len.d_model)

