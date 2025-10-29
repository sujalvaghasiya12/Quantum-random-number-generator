#!/usr/bin/env python3
"""
scripts/generate_qrbits_advanced.py

Advanced QRNG generator using single-qubit measurements.
- Backends: simulator (Aer), ibmq (requires IBMQ token in env), pseudo (numpy)
- Whitening: vonneumann, xor, sha256 (apply in order)
- Entropy estimation: Shannon and empirical min-entropy
- Outputs: CSV of bits and metadata JSON

Usage examples:
    # simulator 4096 bits with von-neumann + sha256 whitening
    python scripts/generate_qrbits_advanced.py --n_bits 4096 --backend simulator \
        --whiten vonneumann,sha256 --out data/qrng_bits.csv

    # pseudo RNG (fast)
    python scripts/generate_qrbits_advanced.py --n_bits 2048 --backend pseudo --out data/pseudo_bits.csv

    # IBMQ (requires IBMQ token exported as IBMQ_TOKEN)
    python scripts/generate_qrbits_advanced.py --n_bits 1024 --backend ibmq --ibmq_backend ibmq_quito \
        --out data/ibmq_bits.csv
"""
from __future__ import annotations
import argparse
import json
import hashlib
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# core libs
import numpy as np
import pandas as pd

# qiskit optional
try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.providers.ibmq import IBMQ
    from qiskit.tools.monitor import job_monitor
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# ---------- Extractors / whitening ----------
def von_neumann(bits: List[int]) -> List[int]:
    """von Neumann extractor: read bits two-by-two, emit unbiased bits."""
    out = []
    # iterate in pairs
    for i in range(0, len(bits) - 1, 2):
        a, b = bits[i], bits[i + 1]
        if a == 0 and b == 1:
            out.append(0)
        elif a == 1 and b == 0:
            out.append(1)
        # 00 or 11 -> discard
    return out

def xor_whiten(bits: List[int], block: int = 8) -> List[int]:
    """XOR whiten: XOR blocks of `block` bits to create 1 bit per block."""
    out = []
    for i in range(0, len(bits), block):
        chunk = bits[i:i+block]
        if len(chunk) < block:
            break
        x = 0
        for b in chunk:
            x ^= (b & 1)
        out.append(x & 1)
    return out

def sha256_whiten(bits: List[int]) -> List[int]:
    """
    SHA-256 whitening: pack bits into bytes, hash with SHA-256, expand hash to bits.
    This is a cryptographic conditioning step: output length = 256 bits per hashed block.
    """
    if len(bits) == 0:
        return []
    # pack into bytes
    byte_arr = bytearray()
    b = 0
    cnt = 0
    for bit in bits:
        b = (b << 1) | (bit & 1)
        cnt += 1
        if cnt == 8:
            byte_arr.append(b)
            b = 0
            cnt = 0
    if cnt != 0:  # pad remaining bits with zeros
        b = b << (8 - cnt)
        byte_arr.append(b)
    digest = hashlib.sha256(bytes(byte_arr)).digest()
    out_bits = []
    for byte in digest:
        for i in reversed(range(8)):
            out_bits.append((byte >> i) & 1)
    return out_bits

# ---------- Entropy estimators ----------
def shannon_entropy(bits: List[int]) -> float:
    """Shannon entropy in bits per symbol for binary sequence."""
    n = len(bits)
    if n == 0:
        return 0.0
    p1 = sum(bits) / n
    p0 = 1.0 - p1
    ent = 0.0
    for p in (p0, p1):
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def empirical_min_entropy(bits: List[int]) -> float:
    """Empirical min-entropy: -log2(max probability of symbol)."""
    n = len(bits)
    if n == 0:
        return 0.0
    p1 = sum(bits) / n
    p0 = 1.0 - p1
    pmax = max(p0, p1)
    # avoid log2(0)
    return -math.log2(pmax) if pmax > 0 else float('inf')

# ---------- Quantum bit generation ----------
def generate_simulator_bits(n_bits: int, batch_shots: int = 4096, seed: Optional[int] = None) -> List[int]:
    """Generate bits using Qiskit Aer qasm_simulator (if available)."""
    if not QISKIT_AVAILABLE:
        print("[WARN] Qiskit not available. Falling back to pseudo RNG (numpy).")
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=n_bits).tolist()

    backend = Aer.get_backend('qasm_simulator')
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    bits: List[int] = []
    batch = min(batch_shots, n_bits)
    while len(bits) < n_bits:
        shots = min(batch, n_bits - len(bits))
        job = execute(qc, backend=backend, shots=shots, seed_simulator=seed)
        result = job.result()
        counts = result.get_counts()
        # counts is dict like {'0': count0, '1': count1}
        # expand counts to bits
        # to keep deterministic order similar to qiskit, we will add zeros then ones (order not relevant for randomness)
        zeros = counts.get('0', 0)
        ones = counts.get('1', 0)
        bits.extend([0] * zeros)
        bits.extend([1] * ones)
    return bits[:n_bits]

def generate_pseudo_bits(n_bits: int, seed: Optional[int] = None) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits).tolist()

def generate_ibmq_bits(n_bits: int, backend_name: Optional[str] = None, batch_shots: int = 1024,
                       hub: Optional[str] = None, group: Optional[str] = None, project: Optional[str] = None) -> List[int]:
    """Generate bits by submitting jobs to an IBMQ backend. Requires IBMQ token in env IBMQ_TOKEN."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit not available. Install qiskit to use IBMQ backend.")
    token = os.getenv("IBMQ_TOKEN")
    if not token:
        raise RuntimeError("IBMQ_TOKEN environment variable not set. Export your token first.")
    IBMQ.enable_account(token)
    # choose provider
    provider = None
    try:
        if hub and group and project:
            provider = IBMQ.get_provider(hub=hub, group=group, project=project)
        else:
            providers = IBMQ.providers()
            provider = providers[0]
    except Exception as e:
        raise RuntimeError(f"Failed to access IBMQ provider: {e}")

    # choose backend
    if backend_name:
        backend = provider.get_backend(backend_name)
    else:
        backends = provider.backends(filters=lambda b: b.configuration().n_qubits >= 1 and (not b.configuration().simulator))
        if not backends:
            raise RuntimeError("No suitable IBMQ backends available for measurement.")
        backend = backends[0]

    # Prepare circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    bits: List[int] = []
    while len(bits) < n_bits:
        shots = min(batch_shots, n_bits - len(bits))
        job = backend.run(qc, shots=shots)
        job_monitor(job)
        result = job.result()
        counts = result.get_counts()
        bits.extend([0] * counts.get('0', 0))
        bits.extend([1] * counts.get('1', 0))
    return bits[:n_bits]

# ---------- Utilities ----------
def apply_whitening(bits: List[int], methods: List[str]) -> Tuple[List[int], dict]:
    """Apply whitening/conditioning methods in order. Return final bits and a dict of intermediate stats."""
    stats = {}
    current = bits
    stats['raw_len'] = len(current)
    stats['raw_shannon'] = shannon_entropy(current)
    stats['raw_min_entropy'] = empirical_min_entropy(current)
    step = 0
    for m in methods:
        m = m.strip().lower()
        step += 1
        if m == 'vonneumann' or m == 'von-neumann' or m == 'vn':
            current = von_neumann(current)
            stats[f'step_{step}_method'] = 'von_neumann'
            stats[f'step_{step}_len'] = len(current)
            stats[f'step_{step}_shannon'] = shannon_entropy(current)
            stats[f'step_{step}_min_entropy'] = empirical_min_entropy(current)
        elif m == 'xor':
            current = xor_whiten(current, block=8)
            stats[f'step_{step}_method'] = 'xor_whiten_block8'
            stats[f'step_{step}_len'] = len(current)
            stats[f'step_{step}_shannon'] = shannon_entropy(current)
            stats[f'step_{step}_min_entropy'] = empirical_min_entropy(current)
        elif m == 'sha256' or m == 'sha-256':
            current = sha256_whiten(current)
            stats[f'step_{step}_method'] = 'sha256_whiten'
            stats[f'step_{step}_len'] = len(current)
            stats[f'step_{step}_shannon'] = shannon_entropy(current)
            stats[f'step_{step}_min_entropy'] = empirical_min_entropy(current)
        elif m == '':
            continue
        else:
            print(f"[WARN] Unknown whitening method '{m}' - skipping.")
    stats['final_len'] = len(current)
    stats['final_shannon'] = shannon_entropy(current)
    stats['final_min_entropy'] = empirical_min_entropy(current)
    return current, stats

def write_output(bits: List[int], out_csv: str, metadata: dict) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'bit': bits})
    df.to_csv(out_csv, index=False)
    meta_path = str(Path(out_csv).with_suffix('.meta.json'))
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved bits to {out_csv}")
    print(f"Saved metadata to {meta_path}")

# ---------- Main CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Advanced QRNG: generate random bits from single-qubit measurements")
    p.add_argument("--n_bits", type=int, default=1024, help="Number of output bits requested (before/after whitening depending on method).")
    p.add_argument("--backend", choices=['simulator', 'ibmq', 'pseudo'], default='simulator', help="Backend to use")
    p.add_argument("--batch_shots", type=int, default=4096, help="Shots per batch when submitting to quantum backend")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for simulator/pseudo RNG")
    p.add_argument("--ibmq_backend", type=str, default=None, help="IBMQ backend name (optional)")
    p.add_argument("--hub", type=str, default=None)
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--whiten", type=str, default="", help="Comma-separated whitening steps: vonneumann,xor,sha256")
    p.add_argument("--out", type=str, default="data/qrng_bits.csv", help="Output CSV path")
    p.add_argument("--estimate_entropy", action='store_true', help="Print entropy estimates")
    p.add_argument("--max_raw_bits", type=int, default=262144, help="Max raw bits to generate (safety limit)")
    return p.parse_args()

def main():
    args = parse_args()
    t0 = time.time()

    methods = [m.strip() for m in args.whiten.split(',')] if args.whiten else []
    print("QRNG generator starting")
    print(f"Requested bits: {args.n_bits}, backend: {args.backend}, whitening: {methods}, batch_shots: {args.batch_shots}")

    # Estimate raw bits required depending on whitening method: vonNeumann discards some pairs
    # conservative multiplier: if using vonneumann, assume need ~4x raw bits; else 1x
    multiplier = 1
    if any(m.lower().startswith('vn') or 'von' in m.lower() for m in methods):
        multiplier = 4  # conservative
    raw_needed = min(args.n_bits * multiplier, args.max_raw_bits)

    # Generate raw bits depending on backend
    raw_bits: List[int] = []
    try:
        if args.backend == 'pseudo':
            raw_bits = generate_pseudo_bits(raw_needed, seed=args.seed)
        elif args.backend == 'simulator':
            raw_bits = generate_simulator_bits(raw_needed, batch_shots=args.batch_shots, seed=args.seed)
        elif args.backend == 'ibmq':
            raw_bits = generate_ibmq_bits(raw_needed, backend_name=args.ibmq_backend, batch_shots=args.batch_shots,
                                          hub=args.hub, group=args.group, project=args.project)
        else:
            raise ValueError(f"Unknown backend {args.backend}")
    except Exception as e:
        print(f"[ERROR] Bit generation failed: {e}")
        return

    # Trim or pad raw bits (if padding needed, warn)
    if len(raw_bits) < raw_needed:
        print(f"[WARN] generated fewer raw bits ({len(raw_bits)}) than expected ({raw_needed}).")
    else:
        raw_bits = raw_bits[:raw_needed]

    # Apply whitening steps
    final_bits, stats = apply_whitening(raw_bits, methods)

    # If final bits less than requested, warn and do not pad (user should request larger n_bits)
    if len(final_bits) < args.n_bits:
        print(f"[WARN] After whitening, only {len(final_bits)} bits available (< requested {args.n_bits}). Saving available bits.")
        out_bits = final_bits
    else:
        out_bits = final_bits[:args.n_bits]

    # Prepare metadata
    metadata = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "requested_bits": args.n_bits,
        "backend": args.backend,
        "backend_name": args.ibmq_backend if args.backend == 'ibmq' else ("Aer" if args.backend == 'simulator' else "pseudo"),
        "whitening_steps": methods,
        "raw_bits_generated": len(raw_bits),
        "final_bits_generated": len(out_bits),
        "stats": stats,
        "seed": args.seed,
        "batch_shots": args.batch_shots
    }

    # Optionally print entropy estimates
    if args.estimate_entropy:
        print("Entropy estimates:")
        print(f"  raw Shannon entropy: {stats.get('raw_shannon'):.4f}")
        print(f"  raw min-entropy: {stats.get('raw_min_entropy'):.4f}")
        print(f"  final Shannon entropy: {stats.get('final_shannon'):.4f}")
        print(f"  final min-entropy: {stats.get('final_min_entropy'):.4f}")

    # Write outputs
    write_output(out_bits, args.out, metadata)

    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
