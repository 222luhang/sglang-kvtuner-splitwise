"""
Quantization/dequantization for KV cache TCP transfer.

Supports two backends:
  * **GPU (default)** – uses PyTorch CUDA ops.  Quantization runs on GPU before
    DtoH, dequantization runs on GPU after HtoD, so only the *compressed* data
    crosses the PCIe bus.
  * **CPU fallback** – original numpy path, used when no CUDA device is available.

Packed wire format (unchanged, GPU ↔ CPU interoperable)
--------------------------------------------------------
[4B nbits][4B group_size][4B num_elements][4B dtype_code]
[scales: float16, num_groups values]
[quantized: int8 or packed nibbles, num_elements values]

dtype_code: 0 = float16, 1 = bfloat16
"""

from __future__ import annotations

import logging
import struct
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

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
    q_flat = quantized.ravel()[:num_elements]

    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    header = struct.pack(_HEADER_FMT, nbits, group_size, num_elements, dtype_code)

    if nbits == 4:
        # Pack two int4 values into one uint8: low nibble first, high nibble second
        q_u8 = q_flat.view(np.uint8) & 0x0F
        if num_elements % 2:
            q_u8 = np.append(q_u8, np.uint8(0))
        packed = q_u8[0::2] | (q_u8[1::2] << 4)
        return header + scales.tobytes() + packed.astype(np.uint8).tobytes()

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
    if nbits == 4:
        # Unpack: each uint8 holds two signed 4-bit values (low nibble, high nibble)
        packed_len = (num_elements + 1) // 2
        packed_bytes = np.frombuffer(
            packed[q_offset : q_offset + packed_len], dtype=np.uint8
        )
        low = (packed_bytes & 0x0F).astype(np.int8)
        high = (packed_bytes >> 4).astype(np.int8)
        # Sign-extend: values >= 8 are negative in signed 4-bit
        low = np.where(low >= 8, low - 16, low).astype(np.int8)
        high = np.where(high >= 8, high - 16, high).astype(np.int8)
        # Interleave back: [low0, high0, low1, high1, ...]
        q_flat = np.empty(packed_len * 2, dtype=np.int8)
        q_flat[0::2] = low
        q_flat[1::2] = high
        q_flat = q_flat[:num_elements]
    else:
        q_flat = np.frombuffer(
            packed[q_offset : q_offset + num_elements], dtype=np.int8
        )

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


# ── GPU-accelerated API ─────────────────────────────────────────────────────


def quantize_on_gpu(
    tensor: torch.Tensor,
    nbits: int = 8,
    group_size: int = 64,
) -> bytes:
    """Quantize a *GPU-resident* tensor and return packed wire-format bytes.

    The tensor must be contiguous, 1-D, and reside on a CUDA device.
    dtype should be float16 or bfloat16.

    Returns CPU bytes in the same wire format as :func:`quantize_for_transfer`
    so the receiver can use either GPU or CPU dequantization.
    """
    assert tensor.is_cuda, "quantize_on_gpu requires a CUDA tensor"

    # Use fused Triton kernel for 4-bit (much faster than PyTorch fallback)
    if nbits == 4:
        return _quantize_4bit_triton(tensor, group_size)

    # --- 8-bit path (original PyTorch, already fast) ---
    is_bf16 = tensor.dtype == torch.bfloat16
    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    num_elements = tensor.numel()

    # Work in float32 on GPU
    fp32 = tensor.to(torch.float32)

    # Pad to multiple of group_size
    remainder = num_elements % group_size
    if remainder:
        pad = group_size - remainder
        fp32 = torch.nn.functional.pad(fp32, (0, pad))
    else:
        pad = 0

    groups = fp32.reshape(-1, group_size)

    # Symmetric quantization: scale = max(|group|) / q_max
    q_max = (1 << (nbits - 1)) - 1  # 127 for 8-bit
    abs_max = groups.abs().amax(dim=1)  # (num_groups,)
    scales = (abs_max / q_max).to(torch.float16)
    safe_scales = torch.where(
        scales == 0, torch.tensor(1e-5, dtype=torch.float16, device=scales.device), scales
    )

    # Quantize
    quantized = (groups / safe_scales[:, None].float()).round().clamp(-q_max - 1, q_max)
    # Free intermediate GPU tensors early to reduce memory pressure
    del fp32, groups, abs_max
    # Trim padding, flatten
    q_flat = quantized.reshape(-1)[:num_elements].to(torch.int8)
    del quantized

    # Build wire-format bytes on CPU
    header = struct.pack(_HEADER_FMT, nbits, group_size, num_elements, dtype_code)
    scales_bytes = scales.cpu().numpy().tobytes()
    del scales, safe_scales
    q_bytes = q_flat.cpu().numpy().tobytes()

    return header + scales_bytes + q_bytes


def _quantize_4bit_pytorch(tensor: torch.Tensor, group_size: int = 64) -> bytes:
    """PyTorch fallback for 4-bit quantize (slow, used only if Triton unavailable)."""
    is_bf16 = tensor.dtype == torch.bfloat16
    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    num_elements = tensor.numel()

    fp32 = tensor.to(torch.float32)
    remainder = num_elements % group_size
    if remainder:
        fp32 = torch.nn.functional.pad(fp32, (0, group_size - remainder))

    groups = fp32.reshape(-1, group_size)
    q_max = 7
    abs_max = groups.abs().amax(dim=1)
    scales = (abs_max / q_max).to(torch.float16)
    safe_scales = torch.where(
        scales == 0, torch.tensor(1e-5, dtype=torch.float16, device=scales.device), scales
    )
    quantized = (groups / safe_scales[:, None].float()).round().clamp(-8, 7).to(torch.int8)
    del fp32, groups, abs_max, safe_scales
    q_flat = quantized.reshape(-1)[:num_elements]
    del quantized
    q_u8 = q_flat.to(torch.uint8) & 0x0F
    if num_elements % 2:
        q_u8 = torch.cat([q_u8, torch.zeros(1, dtype=torch.uint8, device=q_u8.device)])
    packed = q_u8[0::2] | (q_u8[1::2] << 4)
    del q_flat, q_u8

    header = struct.pack(_HEADER_FMT, 4, group_size, num_elements, dtype_code)
    return header + scales.cpu().numpy().tobytes() + packed.cpu().numpy().tobytes()


def _quantize_4bit_triton(tensor: torch.Tensor, group_size: int = 64) -> bytes:
    """Fast 4-bit quantize path using fused Triton kernel."""
    try:
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import (
            quantize_4bit_on_gpu,
        )
    except ImportError:
        logger.warning("[quant] Triton kernel import failed, falling back to PyTorch 4-bit")
        return _quantize_4bit_pytorch(tensor, group_size)

    is_bf16 = tensor.dtype == torch.bfloat16
    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    num_elements = tensor.numel()

    packed, scales = quantize_4bit_on_gpu(tensor, group_size=group_size)

    header = struct.pack(_HEADER_FMT, 4, group_size, num_elements, dtype_code)
    return header + scales.numpy().tobytes() + packed.numpy().tobytes()


def dequantize_on_gpu(
    packed: bytes,
    device: torch.device,
) -> torch.Tensor:
    """Dequantize packed wire-format bytes into a GPU tensor.

    Returns a contiguous 1-D tensor on *device* in the original dtype
    (float16 or bfloat16), ready to be scattered into the KV pool.
    """
    nbits, group_size, num_elements, dtype_code = struct.unpack(
        _HEADER_FMT, packed[:_HEADER_SIZE]
    )

    # Use fused Triton kernel for 4-bit (much faster than PyTorch fallback)
    if nbits == 4:
        return _dequantize_4bit_triton(packed, device)

    # --- 8-bit path (original PyTorch, already fast) ---
    is_bf16 = dtype_code == _DTYPE_CODE_BF16
    target_dtype = torch.bfloat16 if is_bf16 else torch.float16

    q_max = (1 << (nbits - 1)) - 1
    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    # Parse scales → GPU
    scales_offset = _HEADER_SIZE
    scales_nbytes = num_groups * 2
    scales_np = np.frombuffer(
        packed[scales_offset : scales_offset + scales_nbytes], dtype=np.float16
    ).copy()
    scales = torch.from_numpy(scales_np).to(device=device, dtype=torch.float16)

    # Parse quantized values → GPU
    q_offset = scales_offset + scales_nbytes
    q_np = np.frombuffer(
        packed[q_offset : q_offset + num_elements], dtype=np.int8
    ).copy()
    q_flat = torch.from_numpy(q_np).to(device=device)

    # Re-pad for group reshape
    if pad:
        q_padded = torch.nn.functional.pad(q_flat, (0, pad))
    else:
        q_padded = q_flat

    groups = q_padded.reshape(-1, group_size).float()
    del q_flat, q_padded
    fp32 = groups * scales[:, None].float()
    del groups, scales
    result = fp32.reshape(-1)[:num_elements].to(target_dtype)
    del fp32
    return result.contiguous()


def _dequantize_4bit_pytorch(packed: bytes, device: torch.device) -> torch.Tensor:
    """PyTorch fallback for 4-bit dequantize (slow, used only if Triton unavailable)."""
    nbits, group_size, num_elements, dtype_code = struct.unpack(
        _HEADER_FMT, packed[:_HEADER_SIZE]
    )
    is_bf16 = dtype_code == _DTYPE_CODE_BF16
    target_dtype = torch.bfloat16 if is_bf16 else torch.float16
    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    scales_offset = _HEADER_SIZE
    scales_nbytes = num_groups * 2
    scales = torch.from_numpy(
        np.frombuffer(packed[scales_offset:scales_offset + scales_nbytes], dtype=np.float16).copy()
    ).to(device=device, dtype=torch.float16)

    q_offset = scales_offset + scales_nbytes
    packed_len = (num_elements + 1) // 2
    packed_t = torch.from_numpy(
        np.frombuffer(packed[q_offset:q_offset + packed_len], dtype=np.uint8).copy()
    ).to(device=device)
    low = (packed_t & 0x0F).to(torch.int8)
    high = (packed_t >> 4).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    q_flat = torch.empty(packed_len * 2, dtype=torch.int8, device=device)
    q_flat[0::2] = low
    q_flat[1::2] = high
    q_flat = q_flat[:num_elements]
    del low, high, packed_t

    if pad:
        q_padded = torch.nn.functional.pad(q_flat, (0, pad))
    else:
        q_padded = q_flat
    del q_flat

    groups = q_padded.reshape(-1, group_size).float()
    del q_padded
    fp32 = groups * scales[:, None].float()
    del groups, scales
    return fp32.reshape(-1)[:num_elements].to(target_dtype).contiguous()


def _dequantize_4bit_triton(packed: bytes, device: torch.device) -> torch.Tensor:
    """Fast 4-bit dequantize path using fused Triton kernel."""
    try:
        from sglang.srt.disaggregation.tcp.transfer_quant_triton import (
            dequantize_4bit_from_transfer,
        )
    except ImportError:
        logger.warning("[dequant] Triton kernel import failed, falling back to PyTorch 4-bit")
        return _dequantize_4bit_pytorch(packed, device)

    return dequantize_4bit_from_transfer(packed, device)


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

    if nbits == 4:
        q_bytes = (num_elements + 1) // 2
    else:
        q_bytes = num_elements
    packed_size = _HEADER_SIZE + num_groups * 2 + q_bytes  # header + scales + q
    return original_nbytes / packed_size
