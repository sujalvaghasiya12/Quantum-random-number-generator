# scripts/extractors.py
from typing import List
import hashlib

def von_neumann(bits: List[int]) -> List[int]:
    """von Neumann extractor: read bits two-by-two."""
    out = []
    it = iter(bits)
    for a, b in zip(it, it):
        if a == 0 and b == 1:
            out.append(0)
        elif a == 1 and b == 0:
            out.append(1)
        # 00 and 11 are discarded
    return out

def xor_whiten(bits: List[int], block=8) -> List[int]:
    """XOR whitening: XOR groups of `block` bits to produce one bit per block."""
    out = []
    n = len(bits)
    for i in range(0, n, block):
        chunk = bits[i:i+block]
        if len(chunk) < block: break
        x = 0
        for b in chunk:
            x ^= b
        out.append(x & 1)
    return out

def sha256_whiten(bits: List[int]) -> List[int]:
    """Cryptographic whitening: pack bits into bytes, hash with SHA-256, expand to bits."""
    # pack bits to bytes
    b = 0
    bytes_out = bytearray()
    for i, bit in enumerate(bits):
        b = (b << 1) | (bit & 1)
        if (i % 8) == 7:
            bytes_out.append(b)
            b = 0
    # if leftover bits, pad with zeros
    if len(bits) % 8 != 0:
        b <<= (8 - (len(bits) % 8))
        bytes_out.append(b)

    h = hashlib.sha256(bytes(bytes_out)).digest()
    # expand to bit list
    out_bits = []
    for byte in h:
        for i in reversed(range(8)):
            out_bits.append((byte >> i) & 1)
    return out_bits
