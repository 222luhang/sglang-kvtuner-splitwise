#!/usr/bin/env python3
"""
Micro-benchmark for KV Cache transfer components.
Measures quantize/dequantize/TCP-transfer timing separately.

Usage:
    python bench_quant_micro.py --output /tmp/micro_bench_results.json
"""

import argparse
import json
import time
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
from datetime import datetime

# Test parameters
SEQUENCE_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_WARMUP = 3
NUM_ITERATIONS = 10


@dataclass
class TimingResult:
    seq_len: int
    num_layers: int
    total_kv_elements: int

    # Quantization timing (ms)
    quant_8bit_mean: float
    quant_8bit_std: float
    quant_4bit_mean: float
    quant_4bit_std: float

    # Dequantization timing (ms)
    dequant_8bit_mean: float
    dequant_8bit_std: float
    dequant_4bit_mean: float
    dequant_4bit_std: float

    # Estimated transfer size (bytes)
    transfer_size_baseline: int
    transfer_size_8bit: int
    transfer_size_4bit: int

    # Estimated transfer time at different bandwidths (ms)
    transfer_1gbps_baseline: float
    transfer_1gbps_8bit: float
    transfer_1gbps_4bit: float
    transfer_10gbps_baseline: float
    transfer_10gbps_8bit: float
    transfer_10gbps_4bit: float


def quantize_int8_gpu(tensor: torch.Tensor, group_size: int = 128) -> tuple:
    """Quantize tensor to int8 on GPU."""
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    # Reshape for group quantization
    if group_size > 0:
        tensor = tensor.reshape(-1, group_size)

    # Compute scale per group
    scale = tensor.abs().max(dim=-1, keepdim=True).values / 127.0
    scale = scale.clamp(min=1e-8)

    # Quantize
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale, original_shape, original_dtype


def dequantize_int8_gpu(quantized: torch.Tensor, scale: torch.Tensor,
                        original_shape: tuple, original_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize int8 tensor on GPU."""
    # Dequantize
    dequantized = quantized.to(torch.float32) * scale

    # Reshape back
    dequantized = dequantized.reshape(original_shape)

    return dequantized.to(original_dtype)


def quantize_int4_gpu(tensor: torch.Tensor, group_size: int = 128) -> tuple:
    """Quantize tensor to int4 on GPU (packed as int8)."""
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    # Reshape for group quantization
    if group_size > 0:
        tensor = tensor.reshape(-1, group_size)

    # Compute scale per group
    scale = tensor.abs().max(dim=-1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)

    # Quantize to int4 range
    quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)

    # Pack two int4 into one int8 (nibble packing)
    # Even indices in low nibble, odd indices in high nibble
    packed = torch.zeros(quantized.shape[0], quantized.shape[1] // 2,
                         dtype=torch.int8, device=quantized.device)
    packed = (quantized[:, 0::2] & 0x0F) | ((quantized[:, 1::2] & 0x0F) << 4)

    return packed, scale, original_shape, original_dtype


def dequantize_int4_gpu(packed: torch.Tensor, scale: torch.Tensor,
                        original_shape: tuple, original_dtype: torch.dtype,
                        group_size: int = 128) -> torch.Tensor:
    """Dequantize int4 tensor on GPU."""
    # Unpack nibbles
    low_nibble = packed & 0x0F
    high_nibble = (packed >> 4) & 0x0F

    # Convert to signed int4 (handle sign extension)
    low_nibble = torch.where(low_nibble > 7, low_nibble - 16, low_nibble)
    high_nibble = torch.where(high_nibble > 7, high_nibble - 16, high_nibble)

    # Interleave
    quantized = torch.zeros(packed.shape[0], packed.shape[1] * 2,
                           dtype=torch.int8, device=packed.device)
    quantized[:, 0::2] = low_nibble
    quantized[:, 1::2] = high_nibble

    # Dequantize
    dequantized = quantized.to(torch.float32) * scale

    # Reshape back
    dequantized = dequantized.reshape(original_shape)

    return dequantized.to(original_dtype)


def measure_quantize(tensor: torch.Tensor, bits: int, num_iterations: int) -> tuple:
    """Measure quantization timing."""
    times = []

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if bits == 8:
            quantized, scale, shape, dtype = quantize_int8_gpu(tensor)
        else:
            quantized, scale, shape, dtype = quantize_int4_gpu(tensor)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def measure_dequantize(quantized: torch.Tensor, scale: torch.Tensor,
                       shape: tuple, dtype: torch.dtype, bits: int,
                       num_iterations: int) -> tuple:
    """Measure dequantization timing."""
    times = []

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if bits == 8:
            dequantized = dequantize_int8_gpu(quantized, scale, shape, dtype)
        else:
            dequantized = dequantize_int4_gpu(quantized, scale, shape, dtype)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def run_benchmark(device: str = "cuda") -> List[TimingResult]:
    """Run the micro-benchmark for all sequence lengths."""
    results = []

    print("=" * 70)
    print("KV Cache Transfer Micro-Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Layers: {NUM_LAYERS}, KV Heads: {NUM_KV_HEADS}, Head Dim: {HEAD_DIM}")
    print(f"Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print("=" * 70)
    print()

    for seq_len in SEQUENCE_LENGTHS:
        print(f"Testing seq_len={seq_len}...")

        # Create KV cache tensor (simulating all layers)
        # Shape: [num_layers, 2 (K+V), seq_len, num_kv_heads, head_dim]
        kv_shape = (NUM_LAYERS, 2, seq_len, NUM_KV_HEADS, HEAD_DIM)
        kv_tensor = torch.randn(kv_shape, dtype=torch.float16, device=device)

        total_elements = kv_tensor.numel()
        total_bytes = total_elements * 2  # float16 = 2 bytes

        # Warmup
        for _ in range(NUM_WARMUP):
            quantized_8, scale_8, shape_8, dtype_8 = quantize_int8_gpu(kv_tensor)
            quantized_4, scale_4, shape_4, dtype_4 = quantize_int4_gpu(kv_tensor)

        # Measure 8-bit quantization
        quant_8bit_mean, quant_8bit_std = measure_quantize(kv_tensor, 8, NUM_ITERATIONS)

        # Measure 4-bit quantization
        quant_4bit_mean, quant_4bit_std = measure_quantize(kv_tensor, 4, NUM_ITERATIONS)

        # Create quantized tensors for dequantization test
        quantized_8, scale_8, shape_8, dtype_8 = quantize_int8_gpu(kv_tensor)
        quantized_4, scale_4, shape_4, dtype_4 = quantize_int4_gpu(kv_tensor)

        # Warmup dequantization
        for _ in range(NUM_WARMUP):
            dequantize_int8_gpu(quantized_8, scale_8, shape_8, dtype_8)
            dequantize_int4_gpu(quantized_4, scale_4, shape_4, dtype_4)

        # Measure 8-bit dequantization
        dequant_8bit_mean, dequant_8bit_std = measure_dequantize(
            quantized_8, scale_8, shape_8, dtype_8, 8, NUM_ITERATIONS)

        # Measure 4-bit dequantization
        dequant_4bit_mean, dequant_4bit_std = measure_dequantize(
            quantized_4, scale_4, shape_4, dtype_4, 4, NUM_ITERATIONS)

        # Calculate transfer sizes
        # Baseline: raw FP16
        transfer_size_baseline = total_bytes

        # 8-bit: 1 byte per element + scale (1 float16 per 128 elements for K and V separately)
        num_groups = (seq_len * NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS) // 128
        transfer_size_8bit = total_elements * 1 + num_groups * 2  # scales stored as float16

        # 4-bit: 0.5 byte per element + scale
        transfer_size_4bit = total_elements // 2 + num_groups * 2

        # Calculate transfer times at different bandwidths (theoretical)
        # 1 Gbps = 125 MB/s, 10 Gbps = 1250 MB/s
        transfer_1gbps_baseline = transfer_size_baseline / (125 * 1024 * 1024) * 1000
        transfer_1gbps_8bit = transfer_size_8bit / (125 * 1024 * 1024) * 1000
        transfer_1gbps_4bit = transfer_size_4bit / (125 * 1024 * 1024) * 1000

        transfer_10gbps_baseline = transfer_size_baseline / (1250 * 1024 * 1024) * 1000
        transfer_10gbps_8bit = transfer_size_8bit / (1250 * 1024 * 1024) * 1000
        transfer_10gbps_4bit = transfer_size_4bit / (1250 * 1024 * 1024) * 1000

        result = TimingResult(
            seq_len=seq_len,
            num_layers=NUM_LAYERS,
            total_kv_elements=total_elements,
            quant_8bit_mean=quant_8bit_mean,
            quant_8bit_std=quant_8bit_std,
            quant_4bit_mean=quant_4bit_mean,
            quant_4bit_std=quant_4bit_std,
            dequant_8bit_mean=dequant_8bit_mean,
            dequant_8bit_std=dequant_8bit_std,
            dequant_4bit_mean=dequant_4bit_mean,
            dequant_4bit_std=dequant_4bit_std,
            transfer_size_baseline=transfer_size_baseline,
            transfer_size_8bit=transfer_size_8bit,
            transfer_size_4bit=transfer_size_4bit,
            transfer_1gbps_baseline=transfer_1gbps_baseline,
            transfer_1gbps_8bit=transfer_size_8bit / (125 * 1024 * 1024) * 1000,
            transfer_1gbps_4bit=transfer_size_4bit / (125 * 1024 * 1024) * 1000,
            transfer_10gbps_baseline=transfer_10gbps_baseline,
            transfer_10gbps_8bit=transfer_10gbps_8bit,
            transfer_10gbps_4bit=transfer_10gbps_4bit,
        )

        results.append(result)

        print(f"  Quant 8-bit: {quant_8bit_mean:.3f} ms (±{quant_8bit_std:.3f})")
        print(f"  Quant 4-bit: {quant_4bit_mean:.3f} ms (±{quant_4bit_std:.3f})")
        print(f"  Dequant 8-bit: {dequant_8bit_mean:.3f} ms (±{dequant_8bit_std:.3f})")
        print(f"  Dequant 4-bit: {dequant_4bit_mean:.3f} ms (±{dequant_4bit_std:.3f})")
        print(f"  Transfer size: {transfer_size_baseline/1024/1024:.2f} MB (baseline), "
              f"{transfer_size_8bit/1024/1024:.2f} MB (8-bit), "
              f"{transfer_size_4bit/1024/1024:.2f} MB (4-bit)")
        print()

    return results


def print_summary(results: List[TimingResult]):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    print()
    print(f"{'SeqLen':>8} {'Quant8(ms)':>12} {'Quant4(ms)':>12} {'Dequant8(ms)':>14} {'Dequant4(ms)':>14} {'Size(MB)':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r.seq_len:>8} {r.quant_8bit_mean:>12.3f} {r.quant_4bit_mean:>12.3f} "
              f"{r.dequant_8bit_mean:>14.3f} {r.dequant_4bit_mean:>14.3f} "
              f"{r.transfer_size_baseline/1024/1024:>10.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="KV Cache Transfer Micro-Benchmark")
    parser.add_argument("--output", default="/tmp/quant_micro_bench.json",
                       help="Output JSON file path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Run benchmark
    results = run_benchmark(args.device)

    # Print summary
    print_summary(results)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "num_warmup": NUM_WARMUP,
            "num_iterations": NUM_ITERATIONS,
        },
        "results": [asdict(r) for r in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
