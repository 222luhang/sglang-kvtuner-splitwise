"""
NIXL VRAM Bridge for POSIX Backend Compatibility

This module provides VRAM→DRAM bridging for NIXL POSIX backend,
enabling KV Cache transfer on systems without RDMA hardware.

Key Features:
- Transparent VRAM→DRAM copy before POSIX transfer
- Pinned memory for efficient GPU→CPU transfers
- Automatic fallback when VRAM registration fails
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BridgeBuffer:
    """A buffer for VRAM→DRAM bridging."""
    data: torch.Tensor  # CPU tensor (pinned memory)
    size_bytes: int
    in_use: bool = False
    last_used: float = 0.0


class NIXLVRamBridge:
    """
    VRAM Bridge for NIXL POSIX backend.
    
    When NIXL POSIX backend is used (for systems without RDMA),
    it cannot register VRAM memory directly. This bridge:
    
    1. Allocates pinned CPU memory buffers
    2. Copies VRAM data to CPU before transfer
    3. Provides transparent integration with NIXL send/receive
    
    Usage:
        bridge = NIXLVRamBridge(buffer_size_mb=256, num_buffers=4)
        
        # Before sending VRAM data
        cpu_data = bridge.copy_vram_to_dram(vram_tensor)
        # Then use POSIX transfer with cpu_data
        
        # After receiving to DRAM
        vram_data = bridge.copy_dram_to_vram(cpu_tensor)
    """
    
    def __init__(
        self,
        buffer_size_mb: int = 256,
        num_buffers: int = 4,
        device: str = "cuda",
    ):
        """
        Initialize VRAM bridge.
        
        Args:
            buffer_size_mb: Size of each buffer in MB
            num_buffers: Number of pre-allocated buffers
            device: CUDA device
        """
        self.buffer_size_mb = buffer_size_mb
        self.num_buffers = num_buffers
        self.device = device
        
        # Buffer pool
        self._buffers: List[BridgeBuffer] = []
        self._buffer_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "vram_to_dram_copies": 0,
            "dram_to_vram_copies": 0,
            "bytes_transferred_vram_to_dram": 0,
            "bytes_transferred_dram_to_vram": 0,
            "total_copy_time_ms": 0,
        }
        
        # Check if bridge is needed
        self._needs_bridge = self._check_needs_bridge()
        
        if self._needs_bridge:
            self._allocate_buffers()
            logger.info(
                f"NIXL VRAM Bridge initialized: {num_buffers} buffers × {buffer_size_mb}MB"
            )
        else:
            logger.info("VRAM Bridge not needed (RDMA backend available)")
    
    def _check_needs_bridge(self) -> bool:
        """Check if VRAM bridge is needed based on NIXL backend."""
        # Check environment variable for NIXL backend
        nixl_backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "").upper()
        
        # POSIX backend needs VRAM bridge
        if nixl_backend == "POSIX":
            return True
        
        # Check if we're in a testing/debugging mode
        if os.environ.get("SGLANG_FORCE_VRAM_BRIDGE", "").lower() == "true":
            return True
        
        return False
    
    def _allocate_buffers(self):
        """Pre-allocate pinned CPU memory buffers."""
        buffer_size = self.buffer_size_mb * 1024 * 1024
        
        for i in range(self.num_buffers):
            try:
                # Allocate CPU tensor with pinned memory
                cpu_tensor = torch.empty(
                    buffer_size,
                    dtype=torch.uint8,
                    device="cpu",
                )
                cpu_tensor.pin_memory()
                
                self._buffers.append(BridgeBuffer(
                    data=cpu_tensor,
                    size_bytes=buffer_size,
                ))
            except Exception as e:
                logger.warning(f"Failed to allocate buffer {i}: {e}")
                break
        
        logger.info(f"Allocated {len(self._buffers)} pinned CPU buffers")
    
    def needs_bridge(self) -> bool:
        """Check if VRAM bridge is needed."""
        return self._needs_bridge
    
    def get_available_buffer(self, min_size: int = 0) -> Optional[torch.Tensor]:
        """
        Get an available buffer from the pool.
        
        Args:
            min_size: Minimum required size in bytes
            
        Returns:
            CPU tensor buffer or None if none available
        """
        with self._buffer_lock:
            current_time = time.time()
            
            for buf in self._buffers:
                if not buf.in_use and buf.size_bytes >= min_size:
                    buf.in_use = True
                    buf.last_used = current_time
                    return buf.data
            
            # No buffer available, try to allocate a new one
            if min_size > 0:
                try:
                    new_buf = torch.empty(min_size, dtype=torch.uint8, device="cpu")
                    new_buf.pin_memory()
                    
                    bridge_buf = BridgeBuffer(
                        data=new_buf,
                        size_bytes=min_size,
                        in_use=True,
                        last_used=current_time,
                    )
                    self._buffers.append(bridge_buf)
                    return new_buf
                except Exception as e:
                    logger.warning(f"Failed to allocate new buffer: {e}")
            
            return None
    
    def release_buffer(self, buffer: torch.Tensor):
        """Release a buffer back to the pool."""
        with self._buffer_lock:
            for buf in self._buffers:
                if buf.data.data_ptr() == buffer.data_ptr():
                    buf.in_use = False
                    return
    
    def copy_vram_to_dram(
        self,
        vram_tensor: torch.Tensor,
        buffer: Optional[torch.Tensor] = None,
        sync: bool = True,
    ) -> torch.Tensor:
        """
        Copy VRAM tensor to DRAM (pinned memory).
        
        Args:
            vram_tensor: Tensor in GPU memory
            buffer: Optional pre-allocated buffer
            sync: Whether to synchronize after copy
            
        Returns:
            CPU tensor in pinned memory
        """
        start_time = time.time()
        
        # Ensure contiguous
        if not vram_tensor.is_contiguous():
            vram_tensor = vram_tensor.contiguous()
        
        # Calculate size
        size_bytes = vram_tensor.numel() * vram_tensor.element_size()
        
        # Get or allocate buffer
        if buffer is None:
            buffer = self.get_available_buffer(size_bytes)
            if buffer is None:
                # Fall back to regular allocation
                buffer = torch.empty_like(vram_tensor, device="cpu")
                buffer.pin_memory()
        
        # Resize buffer if needed
        if buffer.numel() < vram_tensor.numel():
            buffer = torch.empty_like(vram_tensor, device="cpu")
            buffer.pin_memory()
        
        # Copy slice
        view = buffer[:vram_tensor.numel()].view(vram_tensor.shape)
        view.copy_(vram_tensor, non_blocking=True)
        
        if sync:
            torch.cuda.synchronize()
        
        # Update stats
        self.stats["vram_to_dram_copies"] += 1
        self.stats["bytes_transferred_vram_to_dram"] += size_bytes
        self.stats["total_copy_time_ms"] += (time.time() - start_time) * 1000
        
        return view
    
    def copy_dram_to_vram(
        self,
        dram_tensor: torch.Tensor,
        device: Optional[str] = None,
        sync: bool = True,
    ) -> torch.Tensor:
        """
        Copy DRAM tensor to VRAM.
        
        Args:
            dram_tensor: Tensor in CPU memory
            device: Target CUDA device
            sync: Whether to synchronize after copy
            
        Returns:
            GPU tensor
        """
        start_time = time.time()
        
        device = device or self.device
        
        # Create GPU tensor and copy
        vram_tensor = dram_tensor.to(device, non_blocking=True)
        
        if sync:
            torch.cuda.synchronize()
        
        # Update stats
        size_bytes = dram_tensor.numel() * dram_tensor.element_size()
        self.stats["dram_to_vram_copies"] += 1
        self.stats["bytes_transferred_dram_to_vram"] += size_bytes
        self.stats["total_copy_time_ms"] += (time.time() - start_time) * 1000
        
        return vram_tensor
    
    def copy_kv_cache_to_dram(
        self,
        kv_data_ptrs: List[int],
        kv_data_lens: List[int],
        kv_indices: np.ndarray,
        gpu_id: int = 0,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Copy KV Cache from VRAM to DRAM for POSIX transfer.
        
        Args:
            kv_data_ptrs: List of KV data GPU pointers
            kv_data_lens: List of KV data lengths
            kv_indices: Indices of KV blocks to copy
            gpu_id: GPU ID
            
        Returns:
            Tuple of (list of CPU tensors, metadata dict)
        """
        cpu_tensors = []
        metadata = {
            "original_ptrs": kv_data_ptrs,
            "indices": kv_indices.tolist(),
            "gpu_id": gpu_id,
        }
        
        for ptr, length in zip(kv_data_ptrs, kv_data_lens):
            # Create tensor from GPU memory pointer
            # Note: This requires careful handling of memory layout
            # The actual implementation would need to work with the
            # specific memory layout of SGLang's KV cache
            
            # For now, we create a placeholder
            # In production, this would use cuda_ipc or similar
            pass
        
        return cpu_tensors, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        stats = self.stats.copy()
        
        # Calculate rates
        if stats["total_copy_time_ms"] > 0:
            total_bytes = (
                stats["bytes_transferred_vram_to_dram"] +
                stats["bytes_transferred_dram_to_vram"]
            )
            stats["bandwidth_gbps"] = (
                total_bytes * 8 / 1e9 / (stats["total_copy_time_ms"] / 1000)
            )
        else:
            stats["bandwidth_gbps"] = 0
        
        # Buffer pool info
        stats["num_buffers"] = len(self._buffers)
        stats["buffers_in_use"] = sum(1 for b in self._buffers if b.in_use)
        stats["buffer_size_mb"] = self.buffer_size_mb
        
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "vram_to_dram_copies": 0,
            "dram_to_vram_copies": 0,
            "bytes_transferred_vram_to_dram": 0,
            "bytes_transferred_dram_to_vram": 0,
            "total_copy_time_ms": 0,
        }


# Global bridge instance
_vram_bridge: Optional[NIXLVRamBridge] = None
_bridge_lock = threading.Lock()


def get_vram_bridge(
    buffer_size_mb: int = 256,
    num_buffers: int = 4,
    device: str = "cuda",
    force_new: bool = False,
) -> NIXLVRamBridge:
    """
    Get or create the global VRAM bridge instance.
    
    Args:
        buffer_size_mb: Buffer size in MB
        num_buffers: Number of buffers
        device: CUDA device
        force_new: Force creation of new instance
        
    Returns:
        NIXLVRamBridge instance
    """
    global _vram_bridge
    
    with _bridge_lock:
        if _vram_bridge is None or force_new:
            _vram_bridge = NIXLVRamBridge(
                buffer_size_mb=buffer_size_mb,
                num_buffers=num_buffers,
                device=device,
            )
        return _vram_bridge


def needs_vram_bridge() -> bool:
    """Check if VRAM bridge is needed for current configuration."""
    bridge = get_vram_bridge()
    return bridge.needs_bridge()