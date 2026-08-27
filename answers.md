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