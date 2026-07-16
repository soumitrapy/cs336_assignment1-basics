def encode_utf8(s="Hello, World! € 😀"):
    print(f"s: {s}")
    print(f'encode: {s.encode("utf-8")}, list: {list(s.encode("utf-8"))}')
    for c in s:
        print(f'c: {c}, hex: {hex(ord(c))}, encode: {c.encode("utf-8").hex()}, list(hex): {[hex(t) for t in list(c.encode("utf-8"))]}, list(dec): {list(c.encode("utf-8"))}')

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
if __name__ == "__main__":
    s = "Hello, World! € 😀"
    s = 'Hello, World!'
    print(s)
    print(decode_utf8_bytes_to_str_wrong(s.encode("utf-8")))
