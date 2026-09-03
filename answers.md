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
| Linear(without b) | ... in_features -> ... out_features | out_features.in_features | 2.in_features.out_features |
| RMSNorm | ... d_model -> ... d_model | d_model | 3.d_model |
| sigmoid | ... d -> ... d | 0 | 4d |
| SiLU | ... d -> ... d | 0 | 5d |
| SwiGLU | ... d_model -> ... d_model | 3.d_model.d_ff | 6.d_model.d_ff + 6.d_ff |
|scaled_dot_prod_attention (d_k=d_v) | ... seq_len d_k, ... seq_len d_k,  ... seq_len d_k -> ... seq_len d_k | 0 | 4.seq_len<sup>2</sup>.d_k + 3.seq_len<sup>2</sup> | 4.seq_len<sup>2</sup>.d_k |
| MultiheadedAttention (with rope) | ... seq_len d_model -> ... seq_len d_model | 4.d_model.(num_heads.d_k) | **8.seq_len.d_model.(num_heads.d_k)**[qkv_proj+out_proj] + **4.seq_len<sup>2</sup>.(num_heads.d_k)** + 3.seq_len<sup>2</sup>.num_heads [scaled_dot_product] + 6.seq_len.(num_heads.d_k)[rope] [if num_heads.d_k = d_model: 8.seq_len.d_model<sup>2</sup> + 4.seq_len<sup>2</sup>.d_model + 6.seq_len.d_model + 3.seq_len<sup>2</sup>.num_heads |  8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) [ 8.seq_len.d_model<sup>2</sup> + 4.d_model.seq_len<sup>2</sup>] |
| TransformerBlock| ... seq_len d_model -> ... seq_len d_model | 4_d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model | [MultiheadAttention] + (**6.seq_len.d_model.d_ff** + 6.seq_len.d_ff)[FFN(SwiGLU)] + 2.seq_len.d_model | 8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) + 4.seq_len.d_ff [ 8.seq_len.d_model<sup>2</sup> + 4.d_model.seq_len<sup>2</sup>] + 4.seq_len.d_ff |
|TransformerLM| ... seq_len -> ... seq_len vocab_size | 2.vocab_size.d_model+d_model+num_layers.(4_d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model) | num_layers.[TransformerBlock]+ 3.seq_len.d_model + 2.seq_len.d_model.vocab_size | num_layers.(8.seq_len.d_model.(num_heads.d_k) + 4 seq_len<sup>2</sup>.(num_heads_d_k) + 4.seq_len.d_ff) + 2.seq_len.vocab_size |

- a. Number of Parameters = 2.vocab_size.d_model + d_model + num_layers.(4_d_model.(num_heads.d_k) + 3.d_model.d_ff + 2.d_model). Total parameters: 1640452800 ~ 1.6B, total parameters in GB (in fp32): 6.5618112 GB
- b.

| Matrix | FLOPS |
|:------ |:------|
| output_proj | 2.seq_len.d_model.vocab_size.(batch) |
|MultiHeadSelfAttention | **8.seq_len.d_model.(num_heads.d_k)**[qkv_proj+out_proj] + **4.seq_len<sup>2</sup>.(num_heads.d_k)** + 3.seq_len<sup>2</sup>.num_heads [scaled_dot_product] + 6.seq_len.(num_heads.d_k)[rope] |
|FFN (SwiGLu) | 6.d_model.d_ff.(batch) |

- c. In most of the cases FFN is taking most of the flops (~60%) followed by multi head attention (~30%)

- d. If I increase the context length, Boottleneck becomes the multihead attention (~73%)