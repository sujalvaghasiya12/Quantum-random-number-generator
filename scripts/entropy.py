# scripts/entropy.py
import math
from collections import Counter
from typing import List

def shannon_entropy(bits: List[int]) -> float:
    """Shannon entropy (bits per symbol) for binary sequence."""
    n = len(bits)
    if n == 0: return 0.0
    cnt = Counter(bits)
    ent = 0.0
    for k in cnt:
        p = cnt[k] / n
        ent -= p * math.log2(p)
    return ent

def empirical_min_entropy(bits: List[int]) -> float:
    """Rough min-entropy estimate: -log2(max p_i)."""
    n = len(bits)
    if n == 0: return 0.0
    cnt = Counter(bits)
    pmax = max(cnt.values()) / n
    return -math.log2(pmax)
