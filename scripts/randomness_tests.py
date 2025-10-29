#!/usr/bin/env python3
"""
Simple randomness checks:
- Frequency (monobit) test
- Runs test (basic)
Not a replacement for NIST STS.
"""
import argparse
import pandas as pd
import numpy as np
from scipy import stats

def frequency_test(bits):
    n = len(bits)
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / np.sqrt(n)
    p_value = stats.norm.sf(s_obs) * 2  # two-sided
    return p_value

def runs_test(bits):
    # runs test per NIST: count runs and compare
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) > (2 / np.sqrt(n)):
        return 0.0  # fails prerequisite
    # count runs
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    expected_runs = 2*n*pi*(1-pi)
    variance = 2*n*pi*(1-pi)*(2*n*pi*(1-pi)-(1))
    z = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0.0
    p_value = stats.norm.sf(abs(z))*2
    return p_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_bits.csv", help="CSV file with 'bit' column")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bits = df['bit'].astype(int).tolist()
    n = len(bits)
    print(f"Loaded {n} bits from {args.input}")

    p_freq = frequency_test(bits)
    p_runs = runs_test(bits)

    print("Frequency (monobit) test p-value:", p_freq)
    print("Runs test p-value:", p_runs)

    alpha = 0.01
    print("Frequency test PASS" if p_freq > alpha else "Frequency test FAIL")
    print("Runs test PASS" if p_runs > alpha else "Runs test FAIL")

if __name__ == "__main__":
    main()
