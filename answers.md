# Problem (unicode1): Understanding Unicode (1 point)
- a. Null Character ['\x00', as it is non printable character Null]
- b. Printed value is acutally the value of the character which Null or '\x00'. but the representation is actuall the string literal representation of the character itself so it is as ""\\x00'"
- c. It will be a repplace by the non pritable character '\x00'. So the output of "this is a test" + chr(0) + "string" will be: "this is a test\x00string"

# Problem (unicode2): Unicode Encodings (3 points)
- a. UTF-8 is compact (space efficient), ASCII-compatible, widely used in practice.
- b. the decode function decodes each byte individually from each byte string, it will work for ASCII chartacters as these characters only converts into one byte for any Non-ASCII charatters it is transferred into multiple bytes, so for those case individual byte does not represent any valid characters so for any non-ASCII characters we will get error.
- c. any 2-byte with not of the from 110xxxxx 10xxxxxx