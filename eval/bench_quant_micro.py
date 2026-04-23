#!/usr/bin/env python3
"""
Micro-benchmark for KV Cache transfer quantization.
Tests the actual optimized code from transfer_quant.py and transfer_quant_triton.py.

Usage:
    python bench_quant_micro.py --output /tmp/quant_micro_bench.json
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
from datetime import datetime

SEQUENCE_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_WARMUP = 5
NUM_ITERATIONS = 20


@dataclass
class TimingResult:
    seq_len: int
    num_layers: int
    total_kv_elements: int
    # PyTorch GPU path
    pytorch_quant_8bit_ms: float
    pytorch_dequant_8bit_ms: float
    pytorch_quant_4bit_ms: float
    pytorch_dequant_4bit_ms: float
    # Triton path (4-bit only)
    triton_quant_4bit_ms: float
    triton_dequant_4bit_ms: float
    # Sizes
    size_fp16_mb: float
    size_8bit_mb: float
    size_4bit_mb: float


def bench_pytorch_quantize(tensor: torch.Tensor, nbits: int, group_size: int) -> float:
    """Benchmark quantize_on_gpu from transfer_quant.py."""
    from sglang.srt.disaggregation.tcp.transfer_quant import quantize_on_gpu
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        quantize_on_gpu(tensor, nbits=nbits, group_size=group_size)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def bench_pytorch_dequantize(packed: bytes, device: torch.device) -> float:
    """Benchmark dequantize_on_gpu from transfer_quant.py."""
    from sglang.srt.disaggregation.tcp.transfer_quant import dequantize_on_gpu
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dequantize_on_gpu(packed, device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def bench_triton_quantize(tensor: torch.Tensor, group_size: int) -> float:
    """Benchmark Triton 4-bit quantize."""
    from sglang.srt.disaggregation.tcp.transfer_quant_triton import quantize_4bit_on_gpu
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        quantize_4bit_on_gpu(tensor, group_size=group_size)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def bench_triton_dequantize(packed, scales, num_elements: int, group_size: int, device: torch.device, dtype: torch.dtype) -> float:
    """Benchmark Triton 4-bit dequantize."""
    from sglang.srt.disaggregation.tcp.transfer_quant_triton import dequantize_4bit_on_gpu
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dequantize_4bit_on_gpu(packed, scales, num_elements, group_size=group_size, device=device, dtype=dtype)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def run_benchmark() -> List[TimingResult]:
    from sglang.srt.disaggregation.tcp.transfer_quant import (
        quantize_on_gpu, dequantize_on_gpu,
        transfer_compression_ratio,
    )
    try:
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import (
            quantize_4bit_on_gpu as triton_q4,
            dequantize_4bit_on_gpu as triton_dq4,
        )
        has_triton = True
    except ImportError:
        has_triton = False
        print("[WARN] Triton not available, skipping Triton benchmarks")

    device = torch.device("cuda")
    results = []

    print("=" * 90)
    print("KV Cache Quantization Micro-Benchmark")
    print(f"Layers: {NUM_LAYERS}, KV Heads: {NUM_KV_HEADS}, Head Dim: {HEAD_DIM}")
    print(f"Seq lengths: {SEQUENCE_LENGTHS}")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print(f"Triton: {'available' if has_triton else 'NOT available'}")
    print("=" * 90)

    # Warmup CUDA
    torch.cuda.synchronize()
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    for seq_len in SEQUENCE_LENGTHS:
        kv_shape = (NUM_LAYERS, 2, seq_len, NUM_KV_HEADS, HEAD_DIM)
        kv_tensor = torch.randn(kv_shape, dtype=torch.float16, device=device)
        flat = kv_tensor.reshape(-1)
        total_elements = flat.numel()
        total_bytes = total_elements * 2

        # Warmup
        for _ in range(NUM_WARMUP):
            quantize_on_gpu(flat, nbits=8, group_size=64)
            quantize_on_gpu(flat, nbits=4, group_size=64)
        torch.cuda.synchronize()

        # PyTorch 8-bit quantize
        t_q8 = bench_pytorch_quantize(flat, nbits=8, group_size=64)
        packed_8bit = quantize_on_gpu(flat, nbits=8, group_size=64)
        t_dq8 = bench_pytorch_dequantize(packed_8bit, device)

        # PyTorch 4-bit quantize
        t_q4 = bench_pytorch_quantize(flat, nbits=4, group_size=64)
        packed_4bit = quantize_on_gpu(flat, nbits=4, group_size=64)
        t_dq4 = bench_pytorch_dequantize(packed_4bit, device)

        # Triton 4-bit
        if has_triton:
            for _ in range(NUM_WARMUP):
                triton_q4(flat, group_size=64)
            torch.cuda.synchronize()
            t_triton_q4 = bench_triton_quantize(flat, group_size=64)
            packed_t, scales_t = triton_q4(flat, group_size=64)
            t_triton_dq4 = bench_triton_dequantize(packed_t, scales_t, flat.numel(), 64, device, torch.float16)
        else:
            t_triton_q4 = 0.0
            t_triton_dq4 = 0.0

        # Sizes
        size_fp16_mb = total_bytes / (1024 * 1024)
        size_8bit_mb = total_bytes / (1024 * 1024) / 2 * 1.02  # ~half + overhead
        size_4bit_mb = total_bytes / (1024 * 1024) / 4 * 1.04

        r = TimingResult(
            seq_len=seq_len, num_layers=NUM_LAYERS,
            total_kv_elements=total_elements,
            pytorch_quant_8bit_ms=t_q8, pytorch_dequant_8bit_ms=t_dq8,
            pytorch_quant_4bit_ms=t_q4, pytorch_dequant_4bit_ms=t_dq4,
            triton_quant_4bit_ms=t_triton_q4, triton_dequant_4bit_ms=t_triton_dq4,
            size_fp16_mb=size_fp16_mb, size_8bit_mb=size_8bit_mb, size_4bit_mb=size_4bit_mb,
        )
        results.append(r)

        # Cleanup
        del kv_tensor, flat, packed_8bit, packed_4bit
        torch.cuda.empty_cache()

        print(f"\nseq_len={seq_len:>5}  ({total_bytes/1024/1024:.1f} MB KV)")
        print(f"  PyTorch 8-bit:  quant={t_q8:.2f}ms  dequant={t_dq8:.2f}ms  total={t_q8+t_dq8:.2f}ms")
        print(f"  PyTorch 4-bit:  quant={t_q4:.2f}ms  dequant={t_dq4:.2f}ms  total={t_q4+t_dq4:.2f}ms")
        if has_triton:
            print(f"  Triton  4-bit:  quant={t_triton_q4:.2f}ms  dequant={t_triton_dq4:.2f}ms  total={t_triton_q4+t_triton_dq4:.2f}ms")
            if t_q4 > 0:
                print(f"    Speedup vs PyTorch 4-bit: quant={t_q4/t_triton_q4:.1f}x  dequant={t_dq4/t_triton_dq4:.1f}x")

    return results


def print_summary(results: List[TimingResult]):
    print("\n" + "=" * 100)
    print("Summary (ms)")
    print("=" * 100)
    print(f"{'SeqLen':>6} {'KV(MB)':>8} | {'PyTorch 8bit Q':>14} {'PyTorch 8bit DQ':>16} {'Total 8bit':>10} | "
          f"{'PyTorch 4bit Q':>14} {'PyTorch 4bit DQ':>16} {'Total 4bit':>10} | "
          f"{'Triton 4bit Q':>13} {'Triton 4bit DQ':>15} {'Total Triton':>11}")
    print("-" * 100)
    for r in results:
        print(f"{r.seq_len:>6} {r.size_fp16_mb:>8.1f} | "
              f"{r.pytorch_quant_8bit_ms:>14.2f} {r.pytorch_dequant_8bit_ms:>16.2f} {r.pytorch_quant_8bit_ms+r.pytorch_dequant_8bit_ms:>10.2f} | "
              f"{r.pytorch_quant_4bit_ms:>14.2f} {r.pytorch_dequant_4bit_ms:>16.2f} {r.pytorch_quant_4bit_ms+r.pytorch_dequant_4bit_ms:>10.2f} | "
              f"{r.triton_quant_4bit_ms:>13.2f} {r.triton_dequant_4bit_ms:>15.2f} {r.triton_quant_4bit_ms+r.triton_dequant_4bit_ms:>11.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/quant_micro_bench.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    results = run_benchmark()
    print_summary(results)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "num_warmup": NUM_WARMUP,
            "num_iterations": NUM_ITERATIONS,
        },
        "results": [asdict(r) for r in results],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
