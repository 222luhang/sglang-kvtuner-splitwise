"""Unit tests for TCP transfer quantization (transfer_quant.py)."""

import numpy as np
import pytest

from sglang.srt.disaggregation.tcp.transfer_quant import (
    _bf16_bytes_to_fp32,
    _fp32_to_bf16_bytes,
    dequantize_from_transfer,
    quantize_for_transfer,
    transfer_compression_ratio,
)


def _make_fp16_bytes(n: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    arr = rng.randn(n).astype(np.float16)
    return arr.tobytes()


def _make_bf16_bytes(n: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    fp32 = rng.randn(n).astype(np.float32)
    return _fp32_to_bf16_bytes(fp32)


# ── bf16 round-trip ──────────────────────────────────────────────────────────


class TestBF16Helpers:
    def test_roundtrip(self):
        fp32_orig = np.array([1.0, -0.5, 3.14, 0.0, -100.0], dtype=np.float32)
        raw = _fp32_to_bf16_bytes(fp32_orig)
        fp32_back = _bf16_bytes_to_fp32(raw, len(fp32_orig))
        # bf16 truncates mantissa; allow small relative error
        np.testing.assert_allclose(fp32_back, fp32_orig, rtol=1e-2, atol=1e-3)


# ── quantize / dequantize round-trip ─────────────────────────────────────────


class TestQuantRoundTrip:
    @pytest.mark.parametrize("nbits", [4, 8])
    @pytest.mark.parametrize("is_bf16", [False, True])
    def test_basic_roundtrip(self, nbits, is_bf16):
        n = 256
        if is_bf16:
            raw = _make_bf16_bytes(n)
        else:
            raw = _make_fp16_bytes(n)

        packed = quantize_for_transfer(raw, 2, is_bf16, nbits=nbits, group_size=64)
        restored = dequantize_from_transfer(packed, len(raw))

        assert len(restored) == len(raw)

        # Decode both to fp32 for comparison
        if is_bf16:
            orig_fp32 = _bf16_bytes_to_fp32(raw, n)
            rest_fp32 = _bf16_bytes_to_fp32(restored, n)
        else:
            orig_fp32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            rest_fp32 = np.frombuffer(restored, dtype=np.float16).astype(np.float32)

        # 8-bit should be very close; 4-bit allows more error
        if nbits == 8:
            np.testing.assert_allclose(rest_fp32, orig_fp32, rtol=0.05, atol=0.05)
        else:
            np.testing.assert_allclose(rest_fp32, orig_fp32, rtol=0.3, atol=0.3)

    def test_non_aligned_length(self):
        """Elements not a multiple of group_size."""
        n = 100  # not divisible by 64
        raw = _make_fp16_bytes(n)
        packed = quantize_for_transfer(raw, 2, False, nbits=8, group_size=64)
        restored = dequantize_from_transfer(packed, len(raw))
        assert len(restored) == len(raw)

    def test_zeros(self):
        """All-zero input should not crash."""
        raw = np.zeros(128, dtype=np.float16).tobytes()
        packed = quantize_for_transfer(raw, 2, False, nbits=8)
        restored = dequantize_from_transfer(packed, len(raw))
        rest = np.frombuffer(restored, dtype=np.float16)
        np.testing.assert_array_equal(rest, 0.0)

    def test_large_values(self):
        """Large magnitude values."""
        arr = np.array([65504.0, -65504.0] * 64, dtype=np.float16)  # fp16 max
        raw = arr.tobytes()
        packed = quantize_for_transfer(raw, 2, False, nbits=8)
        restored = dequantize_from_transfer(packed, len(raw))
        rest = np.frombuffer(restored, dtype=np.float16).astype(np.float32)
        orig = arr.astype(np.float32)
        np.testing.assert_allclose(rest, orig, rtol=0.02)


# ── compression ratio ────────────────────────────────────────────────────────


class TestCompressionRatio:
    def test_8bit_ratio(self):
        ratio = transfer_compression_ratio(2048, nbits=8, group_size=64)
        assert ratio > 1.8  # should be close to 2x

    def test_4bit_ratio(self):
        # True 4-bit packing: ~3.77x compression
        ratio = transfer_compression_ratio(2048, nbits=4, group_size=64)
        assert ratio > 3.5

    def test_actual_packed_size(self):
        """Verify actual packed size matches expectation."""
        n = 1024
        raw = _make_fp16_bytes(n)
        packed_8 = quantize_for_transfer(raw, 2, False, nbits=8)
        packed_4 = quantize_for_transfer(raw, 2, False, nbits=4)
        # 8-bit: roughly half the original + small overhead
        assert len(packed_8) < len(raw)
        # 4-bit: roughly quarter the original + small overhead
        assert len(packed_4) < len(packed_8)
        assert len(packed_4) < len(raw) * 0.35

    def test_4bit_odd_elements(self):
        """4-bit packing with odd element count (nibble padding)."""
        n = 101  # odd
        raw = _make_fp16_bytes(n)
        packed = quantize_for_transfer(raw, 2, False, nbits=4, group_size=64)
        restored = dequantize_from_transfer(packed, len(raw))
        assert len(restored) == len(raw)


# ── cosine similarity metric ─────────────────────────────────────────────────


class TestQualityMetrics:
    @pytest.mark.parametrize("nbits", [4, 8])
    def test_cosine_similarity(self, nbits):
        n = 1024
        raw = _make_fp16_bytes(n, seed=123)
        packed = quantize_for_transfer(raw, 2, False, nbits=nbits)
        restored = dequantize_from_transfer(packed, len(raw))

        orig = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        rest = np.frombuffer(restored, dtype=np.float16).astype(np.float32)

        cos_sim = np.dot(orig, rest) / (
            np.linalg.norm(orig) * np.linalg.norm(rest) + 1e-8
        )
        if nbits == 8:
            assert cos_sim > 0.999
        else:
            assert cos_sim > 0.98
