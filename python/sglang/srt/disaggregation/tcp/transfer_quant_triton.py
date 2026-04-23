"""
Fused Triton kernels for 4-bit KV cache quantization / dequantization.

Replaces the multi-kernel PyTorch fallback path in transfer_quant.py with
single fused kernels, eliminating kernel-launch overhead and enabling
coalesced memory access patterns.

Public API
----------
quantize_4bit_on_gpu(tensor, group_size=64) -> (packed_uint8, scales)
    Fused quantize + nibble-pack on GPU; returns CPU tensors for TCP send.

dequantize_4bit_on_gpu(packed_uint8, scales, num_elements, group_size, device, dtype) -> tensor
    Fused unpack + sign-extend + interleave + multiply on GPU.
"""

from __future__ import annotations

import struct

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.tcp.transfer_quant import _HEADER_FMT, _HEADER_SIZE, _DTYPE_CODE_FP16, _DTYPE_CODE_BF16


# ---------------------------------------------------------------------------
# Triton kernel: 4-bit quantize + nibble-pack (fused)
#
# Each program instance handles one quantization group of `GS` elements.
# HALF_GS = GS // 2 (constexpr).  We do two pointer-strided loads (even/odd)
# to avoid dynamic tensor indexing.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_quantize_4bit_kernel(
    input_ptr,
    packed_ptr,
    scale_ptr,
    N,
    group_size,
    q_max,
    eps,
    GS: tl.constexpr,      # group_size (power-of-2)
    HALF_GS: tl.constexpr, # group_size // 2
):
    gid = tl.program_id(0)
    base = gid * group_size
    out_base = gid * HALF_GS

    idx = tl.arange(0, HALF_GS)

    # Load even elements: base + 0, 2, 4, ...
    even_ptrs = input_ptr + base + idx * 2
    even_mask = (base + idx * 2) < N
    even_vals = tl.load(even_ptrs, mask=even_mask, other=0.0).to(tl.float32)

    # Load odd elements: base + 1, 3, 5, ...
    odd_ptrs = input_ptr + base + idx * 2 + 1
    odd_mask = (base + idx * 2 + 1) < N
    odd_vals = tl.load(odd_ptrs, mask=odd_mask, other=0.0).to(tl.float32)

    # Compute scale from combined values
    # Even and odd vals are separate vectors; we need absmax over all GS elements.
    # We only use even_mask for scale since scale covers the full group anyway.
    all_vals = tl.where(idx < HALF_GS, even_vals, 0.0)
    amax = tl.maximum(tl.max(tl.abs(all_vals)), eps)
    # Also check odd vals
    amax = tl.maximum(amax, tl.maximum(tl.max(tl.abs(odd_vals)), eps))

    scale = amax / q_max
    scale_inv = 1.0 / scale

    # Quantize even & odd separately
    even_scaled = even_vals * scale_inv
    even_q = tl.clamp(tl.where(even_scaled >= 0, even_scaled + 0.5, even_scaled - 0.5),
                       -(q_max + 1), q_max).to(tl.int32)

    odd_scaled = odd_vals * scale_inv
    odd_q = tl.clamp(tl.where(odd_scaled >= 0, odd_scaled + 0.5, odd_scaled - 0.5),
                      -(q_max + 1), q_max).to(tl.int32)

    # Pack: low nibble = even, high nibble = odd
    packed = ((even_q & 0xF) | ((odd_q & 0xF) << 4)).to(tl.uint8)

    pair_mask = even_mask  # store one byte per pair; valid if even element exists
    tl.store(packed_ptr + out_base + idx, packed, mask=pair_mask)
    tl.store(scale_ptr + gid, scale.to(tl.float16))


# ---------------------------------------------------------------------------
# Triton kernel: 4-bit dequantize (fused unpack + sign-extend + multiply)
#
# Each program instance handles one quantization group.
# Reads packed bytes via stride-2 pointer loads, writes interleaved output.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequantize_4bit_kernel(
    packed_ptr,
    scale_ptr,
    output_ptr,
    N,               # padded number of elements
    group_size,
    packed_stride,   # = group_size // 2
    GS: tl.constexpr,
    HALF_GS: tl.constexpr,
):
    gid = tl.program_id(0)
    base = gid * group_size
    group_mask = base < N

    idx = tl.arange(0, HALF_GS)

    # Load packed bytes
    pair_ptrs = packed_ptr + gid * packed_stride + idx
    packed_vals = tl.load(pair_ptrs, mask=group_mask, other=0)

    low_nib = packed_vals & 0xF
    high_nib = (packed_vals >> 4) & 0xF

    # Sign-extend 4-bit: cast to int32 first to avoid unsigned wrap-around
    low_signed = tl.where(low_nib >= 8, low_nib.to(tl.int32) - 16, low_nib.to(tl.int32))
    high_signed = tl.where(high_nib >= 8, high_nib.to(tl.int32) - 16, high_nib.to(tl.int32))

    scale = tl.load(scale_ptr + gid, mask=group_mask, other=1.0).to(tl.float32)

    # Write even elements (low nibble * scale)
    even_ptrs = output_ptr + base + idx * 2
    even_mask = group_mask & ((base + idx * 2) < N)
    tl.store(even_ptrs, low_signed.to(tl.float32) * scale, mask=even_mask)

    # Write odd elements (high nibble * scale)
    odd_ptrs = output_ptr + base + idx * 2 + 1
    odd_mask = group_mask & ((base + idx * 2 + 1) < N)
    tl.store(odd_ptrs, high_signed.to(tl.float32) * scale, mask=odd_mask)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Public API: fused 4-bit quantize on GPU
# ---------------------------------------------------------------------------

def quantize_4bit_on_gpu(
    tensor: torch.Tensor,
    group_size: int = 64,
) -> tuple:
    assert tensor.is_cuda
    num_elements = tensor.numel()

    remainder = num_elements % group_size
    if remainder:
        fp32 = torch.nn.functional.pad(tensor.float(), (0, group_size - remainder))
    else:
        fp32 = tensor.float()

    N = fp32.numel()
    num_groups = N // group_size
    GS = _next_pow2(group_size)
    HALF_GS = GS // 2

    packed = torch.empty(num_groups * (group_size // 2), dtype=torch.uint8, device=tensor.device)
    scales = torch.empty(num_groups, dtype=torch.float16, device=tensor.device)

    grid = (num_groups,)
    _fused_quantize_4bit_kernel[grid](
        fp32, packed, scales,
        N=N, group_size=group_size,
        q_max=7, eps=1e-5,
        GS=GS, HALF_GS=HALF_GS,
    )

    packed_len = (num_elements + 1) // 2
    return packed[:packed_len].cpu(), scales.cpu()


# ---------------------------------------------------------------------------
# Public API: fused 4-bit dequantize on GPU
# ---------------------------------------------------------------------------

def dequantize_4bit_on_gpu(
    packed: torch.Tensor,
    scales: torch.Tensor,
    num_elements: int,
    group_size: int = 64,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda")

    if isinstance(packed, (bytes, bytearray)):
        packed = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy())
    elif isinstance(packed, np.ndarray):
        packed = torch.from_numpy(packed.copy())
    if isinstance(scales, (bytes, bytearray)):
        scales = torch.from_numpy(np.frombuffer(scales, dtype=np.float16).copy())
    elif isinstance(scales, np.ndarray):
        scales = torch.from_numpy(scales.copy())

    packed_gpu = packed.to(device=device, non_blocking=True)
    scales_gpu = scales.to(device=device, dtype=torch.float16, non_blocking=True)

    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    N = num_elements + pad
    num_groups = N // group_size
    packed_stride = group_size // 2
    GS = _next_pow2(group_size)
    HALF_GS = GS // 2

    output = torch.empty(N, dtype=torch.float32, device=device)

    grid = (num_groups,)
    _fused_dequantize_4bit_kernel[grid](
        packed_gpu, scales_gpu, output,
        N=N, group_size=group_size, packed_stride=packed_stride,
        GS=GS, HALF_GS=HALF_GS,
    )

    torch.cuda.synchronize()
    return output[:num_elements].to(dtype).contiguous()


# ---------------------------------------------------------------------------
# Drop-in wrappers matching transfer_quant.py wire-format API
# ---------------------------------------------------------------------------

def quantize_4bit_for_transfer(
    data: bytes,
    bytes_per_elem: int,
    is_bf16: bool,
    group_size: int = 64,
) -> bytes:
    """GPU-accelerated 4-bit quantize → wire format."""
    num_elements = len(data) // bytes_per_elem
    if is_bf16:
        fp32 = np.frombuffer(
            np.frombuffer(data, dtype=np.uint16, count=num_elements).astype(np.uint32) << 16,
            dtype=np.float32,
        )
    else:
        fp32 = np.frombuffer(data, dtype=np.float16, count=num_elements).astype(np.float32)

    tensor = torch.from_numpy(fp32).cuda()
    packed, scales = quantize_4bit_on_gpu(tensor, group_size=group_size)
    del tensor
    torch.cuda.synchronize()

    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    header = struct.pack(_HEADER_FMT, 4, group_size, num_elements, dtype_code)
    return header + scales.numpy().tobytes() + packed.numpy().tobytes()


def dequantize_4bit_from_transfer(
    packed: bytes,
    device: torch.device,
) -> torch.Tensor:
    """GPU-accelerated 4-bit dequantize from wire format."""
    nbits, group_size, num_elements, dtype_code = struct.unpack(
        _HEADER_FMT, packed[:_HEADER_SIZE]
    )
    assert nbits == 4

    is_bf16 = dtype_code == _DTYPE_CODE_BF16
    target_dtype = torch.bfloat16 if is_bf16 else torch.float16

    remainder = num_elements % group_size
    pad = (group_size - remainder) if remainder else 0
    num_groups = (num_elements + pad) // group_size

    scales_offset = _HEADER_SIZE
    scales_nbytes = num_groups * 2
    q_offset = scales_offset + scales_nbytes
    packed_len = (num_elements + 1) // 2

    scales_np = np.frombuffer(
        packed[scales_offset:scales_offset + scales_nbytes], dtype=np.float16
    ).copy()
    packed_np = np.frombuffer(
        packed[q_offset:q_offset + packed_len], dtype=np.uint8
    ).copy()

    return dequantize_4bit_on_gpu(
        packed_np, scales_np, num_elements, group_size=group_size,
        device=device, dtype=target_dtype,
    )
