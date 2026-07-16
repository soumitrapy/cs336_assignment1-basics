# UniCode Encoding
Here are the Unicode code point ranges that UTF-8 encodes with 1, 2, 3, or 4 bytes:
    * 1 byte: U+0000 to U+007F
    * 2 bytes: U+0080 to U+07FF
    * 3 bytes: U+0800 to U+FFFF
    * 4 bytes: U+10000 to U+10FFFF
    UTF-8 converts a character into 2 bytes when the Unicode code point is in the range U+0080 to U+07FF.

    The rule for 2-byte UTF-8 is:
    ```text
    110xxxxx 10xxxxxx
    ```
    The `x` bits are filled with the bits of the Unicode code point.

    Answer: So any 2-byte other than this form will obviously not be a valid utf-8 encode.
    (Correct example for explanation, not a wrong example)
    Example: `é`
    Character: `é`  
    Unicode code point: `U+00E9`
    `U+00E9` in binary is:
    ```text
    0000 0000 1110 1001
    ```

    For UTF-8 2-byte encoding, we take the bits `11101001` and place them into the pattern:

    ```text
    110xxxxx 10xxxxxx
    ```

    Split the bits into 5 bits and 6 bits:

    ```text
    11101001 -> 00011 101001
    ```

    Fill them in:

    ```text
    11000011 10101001
    ```

    In hex, that is:

    ```text
    C3 A9
    ```

    So `é` becomes:

    ```python
    "é".encode("utf-8")  # b'\xc3\xa9'
    ```
