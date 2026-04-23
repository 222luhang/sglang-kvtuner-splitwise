"""Unit tests for fused Triton 4-bit quantization/dequantization kernels.

CPU-safe tests: correctness via numpy reference implementation.
GPU tests: marked with pytest.mark.cuda, run on test machines.
"""

import numpy as np
import pytest
import struct
import torch

from sglang.srt.disaggregation.tcp.transfer_quant import (
    _HEADER_FMT,
    _HEADER_SIZE,
    quantize_for_transfer,
    dequantize_from_transfer,
)


# ── Numpy reference implementations ─────────────────────────────────────────

def _numpy_quantize_4bit(fp32: np.ndarray, group_size: int = 64):
    """Reference 4-bit symmetric quantize in numpy."""
    num_elements = len(fp32)
    remainder = num_elements % group_size
    if remainder:
        fp32 = np.concatenate([fp32, np.zeros(group_size - remainder, dtype=np.float32)])
        pad = group_size - remainder
    else:
        pad = 0
    N = len(fp32)
    num_groups = N // group_size
    groups = fp32.reshape(-1, group_size)
    q_max = 7
    abs_max = np.abs(groups).max(axis=1)
    scales = (abs_max / q_max).astype(np.float16)
    safe_scales = np.where(scales == 0, np.float16(1e-5), scales)
    quantized = np.clip(
        np.round(groups / safe_scales[:, None].astype(np.float32)), -8, 7
    ).astype(np.int8)
    q_flat = quantized.ravel()[:num_elements]
    # Pack nibbles
    q_u8 = q_flat.view(np.uint8) & 0x0F
    if num_elements % 2:
        q_u8 = np.append(q_u8, np.uint8(0))
    packed = q_u8[0::2] | (q_u8[1::2] << 4)
    return packed.astype(np.uint8), scales


def _numpy_dequantize_4bit(
    packed: np.ndarray, scales: np.ndarray,
    num_elements: int, group_size: int = 64,
) -> np.ndarray:
    """Reference 4-bit dequantize in numpy."""
    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    low = (packed & 0x0F).astype(np.int8)
    high = (packed >> 4).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    packed_len = len(packed)
    q_flat = np.empty(packed_len * 2, dtype=np.int8)
    q_flat[0::2] = low
    q_flat[1::2] = high
    q_flat = q_flat[:num_elements]

    if pad:
        q_padded = np.concatenate([q_flat, np.zeros(pad, dtype=np.int8)])
    else:
        q_padded = q_flat
    groups = q_padded.reshape(-1, group_size).astype(np.float32)
    fp32 = groups * scales[:, None].astype(np.float32)
    return fp32.ravel()[:num_elements]


# ── Wire format helpers ────────────────────────────────────────────────────

def _make_raw_bytes(n: int, is_bf16: bool = False, seed: int = 42):
    """Generate random fp16/bf16 raw bytes."""
    rng = np.random.RandomState(seed)
    fp32 = rng.randn(n).astype(np.float32) * 10
    if is_bf16:
        u32 = fp32.view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return u16.tobytes(), fp32
    else:
        return fp32.astype(np.float16).tobytes(), fp32


# ── CPU-only correctness tests ─────────────────────────────────────────────

class TestNumpyReferenceRoundTrip:
    """Verify the numpy reference is correct (used as ground truth for GPU tests)."""

    @pytest.mark.parametrize("n,group_size", [(256, 64), (100, 64), (1023, 128)])
    def test_roundtrip(self, n, group_size):
        rng = np.random.RandomState(123)
        fp32 = rng.randn(n).astype(np.float32) * 5

        packed, scales = _numpy_quantize_4bit(fp32, group_size)
        restored = _numpy_dequantize_4bit(packed, scales, n, group_size)

        assert len(restored) == n
        cos_sim = np.dot(restored, fp32) / (np.linalg.norm(restored) * np.linalg.norm(fp32) + 1e-8)
        assert cos_sim > 0.95, f"cosine similarity {cos_sim:.4f} too low"


class TestWireFormatCompatibility:
    """Ensure Triton output matches the existing wire format.

    The Triton kernels should produce bytes identical to quantize_for_transfer
    (numpy CPU path).  We test this by comparing against the numpy reference.
    """

    @pytest.mark.parametrize("n", [64, 256, 100])
    @pytest.mark.parametrize("is_bf16", [False, True])
    def test_quantize_matches_numpy_ref(self, n, is_bf16):
        """Compare quantize_for_transfer against numpy reference.

        bf16 uses a coarser fp32 approximation than fp16, so we use a
        relaxed tolerance for scale comparison.
        """
        raw, fp32 = _make_raw_bytes(n, is_bf16)
        group_size = 64

        # numpy reference
        packed_np, scales_np = _numpy_quantize_4bit(fp32, group_size)

        # existing transfer_quant.py CPU path
        from_transfer = quantize_for_transfer(raw, 2, is_bf16, nbits=4, group_size=group_size)

        # Parse the wire format
        nbits_h, gs_h, ne_h, dc_h = struct.unpack(_HEADER_FMT, from_transfer[:_HEADER_SIZE])
        assert nbits_h == 4
        assert gs_h == group_size
        assert ne_h == n

        scales_offset = _HEADER_SIZE
        scales_nbytes = ((n + group_size - 1) // group_size) * 2
        scales_from_transfer = np.frombuffer(
            from_transfer[scales_offset:scales_offset + scales_nbytes], dtype=np.float16
        )

        packed_offset = scales_offset + scales_nbytes
        packed_len = (n + 1) // 2
        packed_from_transfer = np.frombuffer(
            from_transfer[packed_offset:packed_offset + packed_len], dtype=np.uint8
        )

        # Scales should match (bf16 has ~0.5% additional rounding vs fp16)
        rtol = 1e-2 if is_bf16 else 1e-3
        np.testing.assert_allclose(scales_from_transfer, scales_np, rtol=rtol)

        # Packed values should match for fp16 (bf16 has additional rounding from bf16→fp32)
        if not is_bf16:
            np.testing.assert_array_equal(packed_from_transfer, packed_np)
        else:
            # For bf16: some nibbles may differ due to bf16→fp32 truncation
            mismatches = np.sum(packed_from_transfer != packed_np)
            mismatch_rate = mismatches / len(packed_np)
            assert mismatch_rate < 0.05, f"Too many mismatches: {mismatches} / {len(packed_np)} ({mismatch_rate:.2%})"

    @pytest.mark.parametrize("n", [64, 256, 100, 1023])
    def test_dequant_roundtrip(self, n):
        """Full roundtrip: quantize_for_transfer → dequantize_from_transfer."""
        raw, fp32 = _make_raw_bytes(n, is_bf16=False)
        group_size = 64

        packed = quantize_for_transfer(raw, 2, False, nbits=4, group_size=group_size)
        restored = dequantize_from_transfer(packed, len(raw))

        assert len(restored) == len(raw)

        orig_f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        rest_f32 = np.frombuffer(restored, dtype=np.float16).astype(np.float32)
        cos_sim = np.dot(rest_f32, orig_f32) / (
            np.linalg.norm(rest_f32) * np.linalg.norm(orig_f32) + 1e-8
        )
        assert cos_sim > 0.95, f"cosine similarity {cos_sim:.4f} too low"


# ── GPU tests (run on test machines with CUDA) ─────────────────────────────

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA GPU"
)


class TestTritonKernelCorrectness:
    """Verify Triton kernel output matches the numpy reference."""

    @needs_cuda
    @pytest.mark.parametrize("n", [64, 256, 100, 1023, 4096])
    @pytest.mark.parametrize("group_size", [64, 128])
    def test_quantize_triton_matches_numpy(self, n, group_size):
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import quantize_4bit_on_gpu

        rng = np.random.RandomState(456)
        fp32 = rng.randn(n).astype(np.float32) * 5
        tensor = torch.from_numpy(fp32).half().cuda()

        packed_triton, scales_triton = quantize_4bit_on_gpu(tensor, group_size)
        packed_np, scales_np = _numpy_quantize_4bit(fp32, group_size)

        # Scales should match (up to float16 rounding)
        np.testing.assert_allclose(
            scales_triton.numpy(), scales_np, rtol=1e-3, atol=1e-5
        )

        # Packed values: trim to actual length for comparison
        packed_len = (n + 1) // 2
        # Float32 rounding can cause ≤1 nibble mismatch per ~1000 elements
        mismatched = np.sum(packed_triton.numpy()[:packed_len] != packed_np[:packed_len])
        mismatch_rate = mismatched / packed_len
        assert mismatch_rate < 0.01, f"Too many packed mismatches: {mismatched} / {packed_len} ({mismatch_rate:.2%})"

    @needs_cuda
    @pytest.mark.parametrize("n", [64, 256, 100, 1023, 4096])
    @pytest.mark.parametrize("group_size", [64, 128])
    def test_dequant_triton_matches_numpy(self, n, group_size):
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import dequantize_4bit_on_gpu

        rng = np.random.RandomState(789)
        fp32 = rng.randn(n).astype(np.float32) * 5

        packed_np, scales_np = _numpy_quantize_4bit(fp32, group_size)

        result_triton = dequantize_4bit_on_gpu(
            packed_np, scales_np, n, group_size=group_size,
            device=torch.device("cuda"), dtype=torch.float16,
        )
        result_np = _numpy_dequantize_4bit(packed_np, scales_np, n, group_size)

        np.testing.assert_allclose(
            result_triton.cpu().numpy(),
            result_np,
            rtol=1e-3, atol=1e-3,
        )

    @needs_cuda
    @pytest.mark.parametrize("n", [64, 256, 1023])
    def test_full_roundtrip_triton(self, n):
        """quantize_on_gpu → dequantize_on_gpu roundtrip with Triton kernels."""
        from sglang.srt.disaggregation.tcp.transfer_quant import (
            quantize_on_gpu, dequantize_on_gpu,
        )

        rng = np.random.RandomState(42)
        fp32 = rng.randn(n).astype(np.float32) * 10
        tensor = torch.from_numpy(fp32).half().cuda()

        wire_bytes = quantize_on_gpu(tensor, nbits=4, group_size=64)
        result = dequantize_on_gpu(wire_bytes, device=torch.device("cuda"))

        orig = tensor.float().cpu().numpy()
        restored = result.float().cpu().numpy()

        assert len(restored) == len(orig)
        cos_sim = np.dot(restored, orig) / (
            np.linalg.norm(restored) * np.linalg.norm(orig) + 1e-8
        )
        assert cos_sim > 0.95, f"cosine similarity {cos_sim:.4f} too low"


class TestTritonKernelPerformance:
    """Benchmark: Triton vs PyTorch fallback for 4-bit."""

    @needs_cuda
    @pytest.mark.parametrize("n", [1024, 4096, 16384, 65536])
    def test_quantize_speedup(self, n, group_size=64):
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import quantize_4bit_on_gpu
        from sglang.srt.disaggregation.tcp.transfer_quant import _quantize_4bit_pytorch

        tensor = torch.randn(n, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(3):
            quantize_4bit_on_gpu(tensor, group_size)
            _quantize_4bit_pytorch(tensor, group_size)
        torch.cuda.synchronize()

        import time

        # Benchmark Triton
        times_triton = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            quantize_4bit_on_gpu(tensor, group_size)
            torch.cuda.synchronize()
            times_triton.append((time.perf_counter() - t0) * 1000)

        # Benchmark PyTorch
        times_pytorch = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _quantize_4bit_pytorch(tensor, group_size)
            torch.cuda.synchronize()
            times_pytorch.append((time.perf_counter() - t0) * 1000)

        t_tri = np.mean(times_triton)
        t_py = np.mean(times_pytorch)
        print(f"\n  quantize n={n}: Triton={t_tri:.3f}ms, PyTorch={t_py:.3f}ms, "
              f"speedup={t_py/t_tri:.2f}x")
        assert t_tri < t_py, f"Triton ({t_tri:.3f}ms) should be faster than PyTorch ({t_py:.3f}ms)"

    @needs_cuda
    @pytest.mark.parametrize("n", [1024, 4096, 16384, 65536])
    def test_dequantize_speedup(self, n, group_size=64):
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import dequantize_4bit_on_gpu
        from sglang.srt.disaggregation.tcp.transfer_quant import _dequantize_4bit_pytorch

        # Create wire-format packed bytes via numpy
        rng = np.random.RandomState(42)
        fp32 = rng.randn(n).astype(np.float32) * 10
        packed_np, scales_np = _numpy_quantize_4bit(fp32, group_size)
        device = torch.device("cuda")

        # Build wire format for PyTorch path
        header = struct.pack(_HEADER_FMT, 4, group_size, n, 0)  # dtype_code=0 fp16
        wire_bytes = header + scales_np.tobytes() + packed_np.tobytes()

        # Warmup
        for _ in range(3):
            dequantize_4bit_on_gpu(packed_np, scales_np, n, group_size, device)
            _dequantize_4bit_pytorch(wire_bytes, device)
        torch.cuda.synchronize()

        import time

        # Benchmark Triton
        times_triton = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dequantize_4bit_on_gpu(packed_np, scales_np, n, group_size, device)
            torch.cuda.synchronize()
            times_triton.append((time.perf_counter() - t0) * 1000)

        # Benchmark PyTorch
        times_pytorch = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _dequantize_4bit_pytorch(wire_bytes, device)
            torch.cuda.synchronize()
            times_pytorch.append((time.perf_counter() - t0) * 1000)

        t_tri = np.mean(times_triton)
        t_py = np.mean(times_pytorch)
        print(f"\n  dequantize n={n}: Triton={t_tri:.3f}ms, PyTorch={t_py:.3f}ms, "
              f"speedup={t_py/t_tri:.2f}x")
        assert t_tri < t_py, f"Triton ({t_tri:.3f}ms) should be faster than PyTorch ({t_py:.3f}ms)"


class TestEdgeCases:
    """Edge case tests (CPU-only)."""

    def test_zero_input(self):
        raw = np.zeros(128, dtype=np.float16).tobytes()
        packed = quantize_for_transfer(raw, 2, False, nbits=4)
        restored = dequantize_from_transfer(packed, len(raw))
        rest = np.frombuffer(restored, dtype=np.float16)
        np.testing.assert_array_equal(rest, 0.0)

    def test_single_element(self):
        raw = np.array([3.14], dtype=np.float16).tobytes()
        packed = quantize_for_transfer(raw, 2, False, nbits=4, group_size=64)
        restored = dequantize_from_transfer(packed, len(raw))
        assert len(restored) == len(raw)

    def test_odd_element_count(self):
        n = 101
        rng = np.random.RandomState(42)
        raw = rng.randn(n).astype(np.float16).tobytes()
        packed = quantize_for_transfer(raw, 2, False, nbits=4, group_size=64)
        restored = dequantize_from_transfer(packed, len(raw))
        assert len(restored) == len(raw)

    def test_compression_ratio_4bit(self):
        from sglang.srt.disaggregation.tcp.transfer_quant import transfer_compression_ratio
        ratio = transfer_compression_ratio(2048, nbits=4, group_size=64)
        assert ratio > 3.5
