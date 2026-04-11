"""
Lightweight numpy-based quantization/dequantization for KV cache TCP transfer.

Quantizes BF16/FP16 KV data on CPU before sending over TCP, reducing transfer
size by ~2x (8-bit) or ~4x (4-bit). The packed format is self-contained so the
receiver can dequantize without out-of-band metadata.

Packed wire format
------------------
[4B nbits][4B group_size][4B num_elements][4B dtype_code]
[scales: float16, num_groups values]
[quantized: int8, num_elements values]

dtype_code: 0 = float16, 1 = bfloat16
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

_HEADER_FMT = "!iiiI"  # nbits, group_size, num_elements, dtype_code
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_DTYPE_CODE_FP16 = 0
_DTYPE_CODE_BF16 = 1

# ── helpers ──────────────────────────────────────────────────────────────────


def _bf16_bytes_to_fp32(buf: bytes, count: int) -> np.ndarray:
    """Convert raw bfloat16 bytes to float32 numpy array."""
    u16 = np.frombuffer(buf, dtype=np.uint16, count=count)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _fp32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw bfloat16 bytes."""
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


# ── public API ───────────────────────────────────────────────────────────────


def quantize_for_transfer(
    data: bytes,
    bytes_per_elem: int,
    is_bf16: bool,
    nbits: int = 8,
    group_size: int = 64,
) -> bytes:
    """Quantize raw KV bytes (BF16/FP16) into a compact wire format.

    Parameters
    ----------
    data : bytes
        Raw KV cache bytes straight from GPU (BF16 or FP16).
    bytes_per_elem : int
        Bytes per element in *data* (2 for fp16/bf16).
    is_bf16 : bool
        True if the source dtype is bfloat16, False for float16.
    nbits : int
        Quantization bit-width (4 or 8).
    group_size : int
        Number of elements per quantization group.

    Returns
    -------
    bytes
        Self-contained packed buffer (header + scales + quantized values).
    """
    num_elements = len(data) // bytes_per_elem

    # Decode to float32 for arithmetic
    if is_bf16:
        fp32 = _bf16_bytes_to_fp32(data, num_elements)
    else:
        fp32 = np.frombuffer(data, dtype=np.float16, count=num_elements).astype(
            np.float32
        )

    # Pad to multiple of group_size
    remainder = num_elements % group_size
    if remainder:
        pad = group_size - remainder
        fp32 = np.concatenate([fp32, np.zeros(pad, dtype=np.float32)])
    else:
        pad = 0

    groups = fp32.reshape(-1, group_size)
    num_groups = groups.shape[0]

    # Symmetric quantization: scale = max(|group|) / q_max
    q_max = (1 << (nbits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit
    abs_max = np.abs(groups).max(axis=1)
    scales = (abs_max / q_max).astype(np.float16)
    # Avoid division by zero
    safe_scales = np.where(scales == 0, np.float16(1e-5), scales)

    quantized = np.clip(
        np.round(groups / safe_scales[:, None].astype(np.float32)),
        -q_max - 1,
        q_max,
    ).astype(np.int8)

    # Trim padding from quantized values
    total_q = num_elements  # only keep original elements
    q_flat = quantized.ravel()[:total_q]

    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    header = struct.pack(_HEADER_FMT, nbits, group_size, num_elements, dtype_code)

    return header + scales.tobytes() + q_flat.tobytes()


def dequantize_from_transfer(
    packed: bytes,
    expected_nbytes: int,
) -> bytes:
    """Dequantize a packed buffer back to the original KV dtype bytes.

    Parameters
    ----------
    packed : bytes
        Buffer produced by :func:`quantize_for_transfer`.
    expected_nbytes : int
        Expected size of the output in bytes (= num_pages * item_len).
        Used only for a sanity check; the actual size is derived from the header.

    Returns
    -------
    bytes
        Raw bytes in the original dtype (BF16 or FP16), ready for GPU write.
    """
    nbits, group_size, num_elements, dtype_code = struct.unpack(
        _HEADER_FMT, packed[:_HEADER_SIZE]
    )
    is_bf16 = dtype_code == _DTYPE_CODE_BF16

    q_max = (1 << (nbits - 1)) - 1

    # Pad count (same logic as quantize)
    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    # Parse scales
    scales_offset = _HEADER_SIZE
    scales_nbytes = num_groups * 2  # float16
    scales = np.frombuffer(
        packed[scales_offset : scales_offset + scales_nbytes], dtype=np.float16
    ).copy()

    # Parse quantized values
    q_offset = scales_offset + scales_nbytes
    q_flat = np.frombuffer(packed[q_offset : q_offset + num_elements], dtype=np.int8)

    # Re-pad for group reshape
    if pad:
        q_padded = np.concatenate([q_flat, np.zeros(pad, dtype=np.int8)])
    else:
        q_padded = q_flat

    groups = q_padded.reshape(-1, group_size).astype(np.float32)
    fp32 = groups * scales[:, None].astype(np.float32)
    fp32 = fp32.ravel()[:num_elements]

    # Convert back to original dtype bytes
    if is_bf16:
        return _fp32_to_bf16_bytes(fp32)
    else:
        return fp32.astype(np.float16).tobytes()


def transfer_compression_ratio(
    original_nbytes: int,
    nbits: int = 8,
    group_size: int = 64,
    bytes_per_elem: int = 2,
) -> float:
    """Return the expected compression ratio for given parameters."""
    num_elements = original_nbytes // bytes_per_elem
    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    packed_size = _HEADER_SIZE + num_groups * 2 + num_elements  # header + scales + q
    return original_nbytes / packed_size
