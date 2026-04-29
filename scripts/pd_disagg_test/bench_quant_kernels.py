#!/usr/bin/env python3
"""
Micro-benchmark: Triton vs PyTorch quantize/dequantize kernel comparison.

Run on a GPU machine inside the sglang venv:
    python bench_quant_kernels.py [--num-tokens 30] [--num-layers 28] [--iterations 20]

Measures per-call latency for:
  - 4-bit quantize: PyTorch path vs Triton path
  - 8-bit quantize: PyTorch path only
  - 4-bit dequantize: PyTorch path vs Triton path
  - 8-bit dequantize: PyTorch path only

Also reports wire-format bytes for bandwidth analysis.
"""

import argparse
import json
import struct
import sys
import time

import numpy as np
import torch

# Add sglang to path
sys.path.insert(0, "/home/ubuntu/sglang-kvtuner-splitwise")

from sglang.srt.disaggregation.tcp.transfer_quant import (
    quantize_on_gpu,
    dequantize_on_gpu,
    _HEADER_FMT,
    _HEADER_SIZE,
)

HAS_TRITON = False
try:
    from sglang.srt.disaggregation.tcp.transfer_quant_triton import (
        quantize_4bit_on_gpu as triton_quant_4bit,
        dequantize_4bit_on_gpu as triton_dequant_4bit,
    )
    HAS_TRITON = True
except ImportError:
    pass

# Qwen2.5-7B KV dimensions
NUM_KV_HEADS = 8
HEAD_DIM = 128


def create_kv_tensor(num_tokens: int, device: torch.device) -> torch.Tensor:
    """Create a realistic KV tensor with typical values."""
    data = torch.randn(
        num_tokens * NUM_KV_HEADS * HEAD_DIM,
        dtype=torch.float16,
        device=device,
    )
    # Clamp to typical KV cache value range
    data = data * 0.1
    return data


def benchmark_fn(fn, warmup=5, iterations=20):
    """Benchmark a callable, return (mean_ms, std_ms, per_iter_ms)."""
    # Warmup
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return float(times.mean()), float(times.std()), times.tolist(), result


def run_benchmarks(num_tokens: int, num_layers: int, iterations: int):
    device = torch.device("cuda:0")
    results = {}

    print(f"\n{'='*70}")
    print(f"Kernel Micro-benchmark")
    print(f"{'='*70}")
    print(f"  Model: Qwen2.5-7B (kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM})")
    print(f"  Tokens: {num_tokens}, Layers: {num_layers}")
    print(f"  Triton available: {HAS_TRITON}")
    print(f"  Iterations: {iterations} (+ 5 warmup)")
    print(f"  Device: {device}")

    tensor = create_kv_tensor(num_tokens, device)
    num_elements = tensor.numel()
    tensor_bytes = num_elements * 2  # fp16
    print(f"  Tensor shape: ({num_elements},) = {tensor_bytes/1024:.1f} KB per K/V tensor")

    # ---- 4-bit Quantize: PyTorch ----
    print(f"\n--- 4-bit Quantize (PyTorch) ---")
    mean, std, iters, wire_bytes_py = benchmark_fn(
        lambda: quantize_on_gpu(tensor, nbits=4),
        iterations=iterations,
    )
    print(f"  Mean: {mean:.3f}ms, Std: {std:.3f}ms, Wire: {len(wire_bytes_py)/1024:.1f} KB")
    results["quant_4bit_pytorch"] = {
        "mean_ms": round(mean, 3), "std_ms": round(std, 3),
        "wire_bytes": len(wire_bytes_py), "iterations": iters,
    }

    # ---- 4-bit Quantize: Triton (via quantize_on_gpu which routes to _quantize_4bit_triton) ----
    if HAS_TRITON:
        print(f"\n--- 4-bit Quantize (Triton via quantize_on_gpu) ---")
        mean_triton, std_triton, iters_triton, wire_triton = benchmark_fn(
            lambda: quantize_on_gpu(tensor, nbits=4),  # routes to _quantize_4bit_triton internally
            iterations=iterations,
        )
        print(f"  Mean: {mean_triton:.3f}ms, Std: {std_triton:.3f}ms, Wire: {len(wire_triton)/1024:.1f} KB")
        results["quant_4bit_triton"] = {
            "mean_ms": round(mean_triton, 3), "std_ms": round(std_triton, 3),
            "wire_bytes": len(wire_triton), "iterations": iters_triton,
        }

        # Test _triton_quant_4bit_to_bytes (the full path used by conn.py sender)
        from sglang.srt.disaggregation.tcp.conn import _triton_quant_4bit_to_bytes
        mean_full, std_full, iters_full, wire_full = benchmark_fn(
            lambda: _triton_quant_4bit_to_bytes(tensor),
            iterations=iterations,
        )
        print(f"  Full path (_triton_quant_4bit_to_bytes): {mean_full:.3f}ms ± {std_full:.3f}ms")
        results["quant_4bit_triton_full"] = {
            "mean_ms": round(mean_full, 3), "std_ms": round(std_full, 3),
            "wire_bytes": len(wire_full), "iterations": iters_full,
        }

    # ---- 4-bit Quantize: PyTorch fallback (force via _quantize_4bit_pytorch) ----
    print(f"\n--- 4-bit Quantize (PyTorch fallback) ---")
    from sglang.srt.disaggregation.tcp.transfer_quant import _quantize_4bit_pytorch
    mean_py4, std_py4, iters_py4, wire_py4 = benchmark_fn(
        lambda: _quantize_4bit_pytorch(tensor),
        iterations=iterations,
    )
    print(f"  Mean: {mean_py4:.3f}ms, Std: {std_py4:.3f}ms, Wire: {len(wire_py4)/1024:.1f} KB")
    results["quant_4bit_pytorch_fallback"] = {
        "mean_ms": round(mean_py4, 3), "std_ms": round(std_py4, 3),
        "wire_bytes": len(wire_py4), "iterations": iters_py4,
    }

    # ---- 8-bit Quantize: PyTorch ----
    print(f"\n--- 8-bit Quantize (PyTorch) ---")
    mean_8q, std_8q, iters_8q, wire_8q = benchmark_fn(
        lambda: quantize_on_gpu(tensor, nbits=8),
        iterations=iterations,
    )
    print(f"  Mean: {mean_8q:.3f}ms, Std: {std_8q:.3f}ms, Wire: {len(wire_8q)/1024:.1f} KB")
    results["quant_8bit_pytorch"] = {
        "mean_ms": round(mean_8q, 3), "std_ms": round(std_8q, 3),
        "wire_bytes": len(wire_8q), "iterations": iters_8q,
    }

    # ---- 4-bit Dequantize: PyTorch ----
    print(f"\n--- 4-bit Dequantize (PyTorch fallback) ---")
    from sglang.srt.disaggregation.tcp.transfer_quant import _dequantize_4bit_pytorch
    mean_d4py, std_d4py, iters_d4py, _ = benchmark_fn(
        lambda: _dequantize_4bit_pytorch(wire_py4, device),
        iterations=iterations,
    )
    print(f"  Mean: {mean_d4py:.3f}ms, Std: {std_d4py:.3f}ms")
    results["dequant_4bit_pytorch"] = {
        "mean_ms": round(mean_d4py, 3), "std_ms": round(std_d4py, 3),
        "iterations": iters_d4py,
    }

    # ---- 4-bit Dequantize: Triton ----
    if HAS_TRITON:
        print(f"\n--- 4-bit Dequantize (Triton) ---")
        # Use the full path from conn.py
        from sglang.srt.disaggregation.tcp.conn import _triton_dequant_4bit_from_bytes
        mean_dt, std_dt, iters_dt, _ = benchmark_fn(
            lambda: _triton_dequant_4bit_from_bytes(wire_full, device),
            iterations=iterations,
        )
        print(f"  Mean: {mean_dt:.3f}ms, Std: {std_dt:.3f}ms")
        results["dequant_4bit_triton"] = {
            "mean_ms": round(mean_dt, 3), "std_ms": round(std_dt, 3),
            "iterations": iters_dt,
        }

    # ---- 8-bit Dequantize: PyTorch ----
    print(f"\n--- 8-bit Dequantize (PyTorch) ---")
    mean_d8, std_d8, iters_d8, _ = benchmark_fn(
        lambda: dequantize_on_gpu(wire_8q, device),
        iterations=iterations,
    )
    print(f"  Mean: {mean_d8:.3f}ms, Std: {std_d8:.3f}ms")
    results["dequant_8bit_pytorch"] = {
        "mean_ms": round(mean_d8, 3), "std_ms": round(std_d8, 3),
        "iterations": iters_d8,
    }

    # ---- Per-layer summary (K+V pair) ----
    print(f"\n{'='*70}")
    print(f"Per-layer summary (K+V pair, {num_tokens} tokens)")
    print(f"{'='*70}")
    print(f"{'Path':<35} {'Mean(ms)':>10} {'Std(ms)':>10} {'Wire(KB)':>10}")
    print(f"{'-'*65}")

    summary_rows = []
    for key, label in [
        ("quant_4bit_pytorch_fallback", "4bit Quant  [PyTorch]"),
        ("quant_4bit_triton",          "4bit Quant  [Triton]"),
        ("quant_4bit_triton_full",     "4bit Quant  [Triton+full]"),
        ("quant_8bit_pytorch",         "8bit Quant  [PyTorch]"),
        ("dequant_4bit_pytorch",       "4bit Dequant[PyTorch]"),
        ("dequant_4bit_triton",        "4bit Dequant[Triton]"),
        ("dequant_8bit_pytorch",       "8bit Dequant[PyTorch]"),
    ]:
        if key in results:
            r = results[key]
            wire = r.get("wire_bytes", 0) / 1024
            print(f"  {label:<33} {r['mean_ms']:>10.3f} {r['std_ms']:>10.3f} {wire:>10.1f}")
            summary_rows.append((label, r["mean_ms"], r["std_ms"], wire))

    # ---- 28-layer total estimates ----
    print(f"\n{'='*70}")
    print(f"28-layer total estimates (K+V × 28 layers)")
    print(f"{'='*70}")
    print(f"{'Path':<35} {'Total(ms)':>10} {'vs baseline':>12}")
    print(f"{'-'*60}")

    baseline_quant = results.get("quant_8bit_pytorch", {}).get("mean_ms", 0)
    baseline_dequant = results.get("dequant_8bit_pytorch", {}).get("mean_ms", 0)
    baseline_total = (baseline_quant + baseline_dequant) * 28

    quant_dequant_pairs = [
        ("quant_4bit_pytorch_fallback", "dequant_4bit_pytorch",  "4bit PyTorch"),
        ("quant_4bit_triton",          "dequant_4bit_triton",   "4bit Triton"),
        ("quant_8bit_pytorch",         "dequant_8bit_pytorch",  "8bit PyTorch"),
    ]

    for qkey, dkey, plabel in quant_dequant_pairs:
        if qkey not in results or dkey not in results:
            continue
        qmean = results[qkey]["mean_ms"]
        dmean = results[dkey]["mean_ms"]
        total = (qmean + dmean) * 28
        savings = baseline_total - total
        pct = (savings / baseline_total * 100) if baseline_total > 0 else 0
        sign = "+" if savings >= 0 else ""
        print(f"  {plabel:<25} {total:>10.1f} {sign}{savings:>10.1f}ms ({sign}{pct:.1f}%)")

    # Save results
    output = {
        "config": {
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "num_elements": num_elements,
            "tensor_bytes": tensor_bytes,
            "iterations": iterations,
            "has_triton": HAS_TRITON,
        },
        "results": results,
        "summary_rows": [
            {"label": l, "mean_ms": m, "std_ms": s, "wire_kb": w}
            for l, m, s, w in summary_rows
        ],
    }

    outpath = "/tmp/kernel_benchmark_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=30,
                        help="Number of tokens per KV tensor (default: 30 for medium prompt)")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    run_benchmarks(args.num_tokens, args.num_layers, args.iterations)


if __name__ == "__main__":
    main()
