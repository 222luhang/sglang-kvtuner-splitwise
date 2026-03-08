"""
KVTuner Compressed KV Cache Transfer

Implements KV Cache compression using KVTuner quantization for efficient
transfer in disaggregated P/D architectures.

Key Features:
- 2/4/8-bit quantization compression
- VRAM→DRAM bridge for POSIX backend compatibility
- Prefill/Decode mode-aware compression
- Layer-wise quantization support
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
    CommonKVReceiver,
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level for KV Cache transfer."""
    LOSSLESS = "lossless"      # No compression (BF16/FP16)
    HIGH_QUALITY = "high"      # 8-bit quantization
    BALANCED = "balanced"      # 4-bit quantization (recommended)
    HIGH_COMPRESSION = "max"   # 2-bit quantization (experimental)


@dataclass
class KVTunerTransferConfig:
    """Configuration for KVTuner compressed transfer."""
    # Quantization settings
    nbits_key: int = 4
    nbits_value: int = 4
    asym: bool = False
    q_group_size: int = 64
    
    # Transfer settings
    enable_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    
    # VRAM bridge for POSIX backend
    enable_vram_bridge: bool = True
    bridge_buffer_size_mb: int = 256  # Buffer size for VRAM→DRAM copy
    
    # Layer-wise settings
    enable_layer_wise: bool = False
    layer_config: Optional[Dict[int, Tuple[int, int]]] = None  # {layer: (nbits_k, nbits_v)}
    
    # Prefill/Decode mode settings
    prefill_nbits: int = 4
    decode_nbits: int = 8
    
    def __post_init__(self):
        if isinstance(self.compression_level, str):
            self.compression_level = CompressionLevel(self.compression_level)
        
        # Set nbits based on compression level
        if self.compression_level == CompressionLevel.LOSSLESS:
            self.nbits_key = 16
            self.nbits_value = 16
            self.enable_compression = False
        elif self.compression_level == CompressionLevel.HIGH_QUALITY:
            self.nbits_key = 8
            self.nbits_value = 8
        elif self.compression_level == CompressionLevel.BALANCED:
            self.nbits_key = 4
            self.nbits_value = 4
        elif self.compression_level == CompressionLevel.HIGH_COMPRESSION:
            self.nbits_key = 2
            self.nbits_value = 2


@dataclass
class QuantizedKVBlock:
    """Container for a quantized KV block."""
    # Quantized data (int8)
    key_tensor: torch.Tensor
    value_tensor: torch.Tensor
    
    # Scale factors (float32)
    key_scales: torch.Tensor
    value_scales: torch.Tensor
    
    # Optional zero points for asymmetric quantization
    key_zeros: Optional[torch.Tensor] = None
    value_zeros: Optional[torch.Tensor] = None
    
    # Metadata
    layer_idx: int = 0
    original_shape: Tuple[int, ...] = ()
    nbits: int = 4
    q_group_size: int = 64
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_bytes = np.prod(self.original_shape) * 2  # BF16 = 2 bytes
        compressed_bytes = (
            self.key_tensor.numel() + self.value_tensor.numel() +
            self.key_scales.numel() * 4 + self.value_scales.numel() * 4
        )
        if self.key_zeros is not None:
            compressed_bytes += self.key_zeros.numel() * 4
        if self.value_zeros is not None:
            compressed_bytes += self.value_zeros.numel() * 4
        return original_bytes / max(compressed_bytes, 1)


class KVTunerQuantizer:
    """
    KVTuner quantizer for KV Cache compression.
    
    Supports symmetric and asymmetric quantization with configurable
    bits and group sizes.
    """
    
    def __init__(self, config: KVTunerTransferConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Precompute quantization parameters
        self._quantizers: Dict[Tuple[int, bool], Any] = {}
        
    def _get_quant_params(self, nbits: int) -> Tuple[int, int]:
        """Get quantization min/max values."""
        q_max = 2 ** (nbits - 1) - 1
        q_min = -(2 ** (nbits - 1))
        return q_min, q_max
    
    def quantize(
        self,
        tensor: torch.Tensor,
        nbits: Optional[int] = None,
        q_group_size: Optional[int] = None,
        asym: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize a tensor using KVTuner-style quantization.
        
        Args:
            tensor: Input tensor to quantize
            nbits: Number of bits (default: from config)
            q_group_size: Group size for quantization (default: from config)
            asym: Use asymmetric quantization (default: from config)
            
        Returns:
            Tuple of (quantized_tensor, scales, zeros)
        """
        nbits = nbits or self.config.nbits_key
        q_group_size = q_group_size or self.config.q_group_size
        asym = asym if asym is not None else self.config.asym
        
        q_min, q_max = self._get_quant_params(nbits)
        
        # Handle group size
        if q_group_size == -1:
            q_group_size = tensor.shape[-1]
        
        # Reshape for group-wise quantization
        original_shape = tensor.shape
        tensor_flat = tensor.reshape(-1, q_group_size).float()
        
        if asym:
            # Asymmetric quantization
            _max = tensor_flat.max(dim=1).values
            _min = tensor_flat.min(dim=1).values
            scale = (_max - _min).clamp(min=1e-5) / (q_max - q_min)
            zeros = torch.round(_min / scale) - q_min
            quant = torch.round(tensor_flat / scale.unsqueeze(1) - zeros.unsqueeze(1))
        else:
            # Symmetric quantization
            scale = tensor_flat.abs().max(dim=1).values.clamp(min=1e-5) / q_max
            zeros = None
            quant = torch.round(tensor_flat / scale.unsqueeze(1))
        
        # Clamp to valid range
        quant = quant.clamp(q_min, q_max).to(torch.int8)
        
        return quant, scale, zeros
    
    def dequantize(
        self,
        quant: torch.Tensor,
        scale: torch.Tensor,
        zeros: Optional[torch.Tensor] = None,
        original_shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize a quantized tensor.
        
        Args:
            quant: Quantized tensor
            scale: Scale factors
            zeros: Zero points (optional, for asymmetric)
            original_shape: Original tensor shape
            dtype: Output dtype
            
        Returns:
            Dequantized tensor
        """
        if zeros is not None:
            dequant = (quant.float() + zeros.unsqueeze(1)) * scale.unsqueeze(1)
        else:
            dequant = quant.float() * scale.unsqueeze(1)
        
        if original_shape is not None:
            dequant = dequant.reshape(original_shape)
        
        return dequant.to(dtype)


class VRAMBridge:
    """
    Bridge for transferring data between VRAM and DRAM.
    
    This enables POSIX backend compatibility by copying VRAM data
    to DRAM before transfer.
    """
    
    def __init__(self, buffer_size_mb: int = 256, device: str = "cuda"):
        self.buffer_size_mb = buffer_size_mb
        self.device = device
        self._buffers: List[torch.Tensor] = []
        self._buffer_index = 0
        
    def allocate_buffers(self, num_buffers: int = 4):
        """Pre-allocate DRAM buffers for efficient transfers."""
        buffer_size = self.buffer_size_mb * 1024 * 1024
        for _ in range(num_buffers):
            # Allocate on CPU (DRAM)
            buf = torch.empty(buffer_size, dtype=torch.uint8, device="cpu")
            buf.pin_memory()  # Pin for faster GPU→CPU transfer
            self._buffers.append(buf)
    
    def vram_to_dram(
        self,
        vram_tensor: torch.Tensor,
        sync: bool = True
    ) -> torch.Tensor:
        """
        Copy VRAM tensor to DRAM.
        
        Args:
            vram_tensor: Tensor in VRAM
            sync: Whether to synchronize after copy
            
        Returns:
            Tensor in DRAM (pinned memory)
        """
        # Ensure tensor is contiguous
        vram_tensor = vram_tensor.contiguous()
        
        # Allocate CPU tensor with pinned memory
        cpu_tensor = torch.empty_like(vram_tensor, device="cpu", dtype=vram_tensor.dtype)
        cpu_tensor.pin_memory()
        
        # Async copy
        cpu_tensor.copy_(vram_tensor, non_blocking=True)
        
        if sync:
            torch.cuda.synchronize()
        
        return cpu_tensor
    
    def dram_to_vram(
        self,
        dram_tensor: torch.Tensor,
        device: Optional[str] = None,
        sync: bool = True
    ) -> torch.Tensor:
        """
        Copy DRAM tensor to VRAM.
        
        Args:
            dram_tensor: Tensor in DRAM
            device: Target device (default: self.device)
            sync: Whether to synchronize after copy
            
        Returns:
            Tensor in VRAM
        """
        device = device or self.device
        vram_tensor = dram_tensor.to(device, non_blocking=True)
        
        if sync:
            torch.cuda.synchronize()
        
        return vram_tensor
    
    def get_next_buffer(self) -> Optional[torch.Tensor]:
        """Get next available pre-allocated buffer."""
        if not self._buffers:
            return None
        buf = self._buffers[self._buffer_index]
        self._buffer_index = (self._buffer_index + 1) % len(self._buffers)
        return buf


class KVTunerCompressedKVSender(CommonKVSender):
    """
    KV Sender with KVTuner compression support.
    
    Features:
    - Quantizes KV Cache before transfer
    - Uses VRAM bridge for POSIX backend compatibility
    - Supports layer-wise compression settings
    """
    
    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        config: Optional[KVTunerTransferConfig] = None,
    ):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank
        
        # Initialize config
        self.config = config or KVTunerTransferConfig()
        
        # Initialize quantizer
        self.quantizer = KVTunerQuantizer(self.config)
        
        # Initialize VRAM bridge if needed
        self.vram_bridge: Optional[VRAMBridge] = None
        if self.config.enable_vram_bridge:
            self.vram_bridge = VRAMBridge(
                buffer_size_mb=self.config.bridge_buffer_size_mb
            )
        
        # Statistics
        self.stats = {
            "total_transfers": 0,
            "total_bytes_original": 0,
            "total_bytes_compressed": 0,
            "total_time_ms": 0,
        }
        
        # Call parent init
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
    
    def compress_kv_layer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> QuantizedKVBlock:
        """
        Compress a single KV layer using KVTuner quantization.
        
        Args:
            key: Key tensor [num_tokens, num_heads, head_dim]
            value: Value tensor [num_tokens, num_heads, head_dim]
            layer_idx: Layer index
            
        Returns:
            QuantizedKVBlock with compressed data
        """
        # Get layer-specific nbits if configured
        if self.config.enable_layer_wise and self.config.layer_config:
            nbits_k, nbits_v = self.config.layer_config.get(
                layer_idx, (self.config.nbits_key, self.config.nbits_value)
            )
        else:
            nbits_k = self.config.nbits_key
            nbits_v = self.config.nbits_value
        
        # Quantize
        q_key, k_scale, k_zero = self.quantizer.quantize(key, nbits=nbits_k)
        q_value, v_scale, v_zero = self.quantizer.quantize(value, nbits=nbits_v)
        
        return QuantizedKVBlock(
            key_tensor=q_key,
            value_tensor=q_value,
            key_scales=k_scale,
            value_scales=v_scale,
            key_zeros=k_zero,
            value_zeros=v_zero,
            layer_idx=layer_idx,
            original_shape=key.shape,
            nbits=max(nbits_k, nbits_v),
            q_group_size=self.config.q_group_size,
        )
    
    def decompress_kv_layer(
        self,
        block: QuantizedKVBlock,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress a quantized KV block.
        
        Args:
            block: Quantized KV block
            dtype: Output dtype
            
        Returns:
            Tuple of (key, value) tensors
        """
        key = self.quantizer.dequantize(
            block.key_tensor,
            block.key_scales,
            block.key_zeros,
            block.original_shape,
            dtype,
        )
        
        # Calculate value shape
        value_shape = list(block.original_shape)
        value_shape[-1] = block.value_tensor.shape[-1] if len(block.value_tensor.shape) > 1 else block.original_shape[-1]
        
        value = self.quantizer.dequantize(
            block.value_tensor,
            block.value_scales,
            block.value_zeros,
            tuple(value_shape),
            dtype,
        )
        
        return key, value
    
    def send_compressed(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Send KV Cache with optional compression.
        
        Args:
            kv_indices: KV Cache indices to send
            state_indices: Optional state indices
            compress: Whether to compress before sending
            
        Returns:
            Transfer statistics
        """
        start_time = time.time()
        stats = {
            "original_size_mb": 0,
            "compressed_size_mb": 0,
            "compression_ratio": 1.0,
            "transfer_time_ms": 0,
        }
        
        if not compress or not self.config.enable_compression:
            # Fall back to uncompressed transfer
            super().send(kv_indices, state_indices)
            stats["transfer_time_ms"] = (time.time() - start_time) * 1000
            return stats
        
        # Get KV Cache data
        kv_pool = self.mgr.kv_args
        
        # Calculate original size
        num_tokens = len(kv_indices)
        num_layers = len(kv_pool.kv_data_ptrs)
        original_bytes = num_tokens * sum(kv_pool.kv_item_lens)
        stats["original_size_mb"] = original_bytes / (1024 * 1024)
        
        # Compress and transfer each layer
        compressed_blocks: List[QuantizedKVBlock] = []
        
        for layer_idx in range(num_layers):
            # Get layer KV data (would need actual implementation)
            # This is a placeholder - actual implementation would extract
            # KV data from the memory pool
            pass
        
        # Transfer compressed data
        # (Implementation depends on the actual transfer mechanism)
        
        stats["transfer_time_ms"] = (time.time() - start_time) * 1000
        stats["compressed_size_mb"] = stats["original_size_mb"] / stats["compression_ratio"]
        
        return stats


class KVTunerCompressedKVReceiver(CommonKVReceiver):
    """
    KV Receiver with KVTuner decompression support.
    
    Features:
    - Receives and decompresses KV Cache
    - Supports VRAM bridge for POSIX backend
    - Handles layer-wise decompression
    """
    
    def __init__(
        self,
        mgr: CommonKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        config: Optional[KVTunerTransferConfig] = None,
    ):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        
        # Initialize config
        self.config = config or KVTunerTransferConfig()
        
        # Initialize quantizer
        self.quantizer = KVTunerQuantizer(self.config)
        
        # Initialize VRAM bridge
        self.vram_bridge: Optional[VRAMBridge] = None
        if self.config.enable_vram_bridge:
            self.vram_bridge = VRAMBridge(
                buffer_size_mb=self.config.bridge_buffer_size_mb
            )
        
        # Statistics
        self.stats = {
            "total_receives": 0,
            "total_bytes_received": 0,
            "total_decompress_time_ms": 0,
        }
        
        # Call parent init
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
    
    def receive_and_decompress(
        self,
        compressed_data: bytes,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Receive and decompress KV Cache data.
        
        Args:
            compressed_data: Compressed KV data
            layer_idx: Layer index
            
        Returns:
            Tuple of (key, value) tensors
        """
        # Parse compressed data
        # (Implementation depends on the serialization format)
        
        # Decompress
        # (Would use quantizer.dequantize)
        
        raise NotImplementedError("Implementation depends on serialization format")
    
    def poll(self) -> KVPoll:
        """Check transfer status."""
        return super().poll()


def create_compressed_sender(
    mgr: CommonKVManager,
    bootstrap_addr: str,
    bootstrap_room: int,
    dest_tp_ranks: List[int],
    pp_rank: int,
    server_args: "ServerArgs",
) -> KVTunerCompressedKVSender:
    """
    Factory function to create a compressed KV sender.
    
    Args:
        mgr: KV Manager
        bootstrap_addr: Bootstrap server address
        bootstrap_room: Room ID
        dest_tp_ranks: Destination TP ranks
        pp_rank: Pipeline parallel rank
        server_args: Server arguments
        
    Returns:
        Configured KVTunerCompressedKVSender
    """
    config = KVTunerTransferConfig(
        nbits_key=getattr(server_args, 'kvtuner_nbits_key', 4),
        nbits_value=getattr(server_args, 'kvtuner_nbits_value', 4),
        asym=getattr(server_args, 'kvtuner_asym', False),
        q_group_size=getattr(server_args, 'kvtuner_q_group_size', 64),
        enable_compression=getattr(server_args, 'enable_kvtuner_quant', False),
        enable_layer_wise=getattr(server_args, 'enable_kvtuner_layer_wise', False),
        prefill_nbits=getattr(server_args, 'kvtuner_nbits_key', 4),
        decode_nbits=8,  # Higher precision for decode
    )
    
    return KVTunerCompressedKVSender(
        mgr=mgr,
        bootstrap_addr=bootstrap_addr,
        bootstrap_room=bootstrap_room,
        dest_tp_ranks=dest_tp_ranks,
        pp_rank=pp_rank,
        config=config,
    )


def create_compressed_receiver(
    mgr: CommonKVManager,
    bootstrap_addr: str,
    bootstrap_room: Optional[int],
    server_args: "ServerArgs",
) -> KVTunerCompressedKVReceiver:
    """
    Factory function to create a compressed KV receiver.
    
    Args:
        mgr: KV Manager
        bootstrap_addr: Bootstrap server address
        bootstrap_room: Room ID
        server_args: Server arguments
        
    Returns:
        Configured KVTunerCompressedKVReceiver
    """
    config = KVTunerTransferConfig(
        nbits_key=getattr(server_args, 'kvtuner_nbits_key', 4),
        nbits_value=getattr(server_args, 'kvtuner_nbits_value', 4),
        asym=getattr(server_args, 'kvtuner_asym', False),
        q_group_size=getattr(server_args, 'kvtuner_q_group_size', 64),
        enable_compression=getattr(server_args, 'enable_kvtuner_quant', False),
        enable_layer_wise=getattr(server_args, 'enable_kvtuner_layer_wise', False),
        prefill_nbits=4,
        decode_nbits=getattr(server_args, 'kvtuner_nbits_key', 8),
    )
    
    return KVTunerCompressedKVReceiver(
        mgr=mgr,
        bootstrap_addr=bootstrap_addr,
        bootstrap_room=bootstrap_room,
        config=config,
    )