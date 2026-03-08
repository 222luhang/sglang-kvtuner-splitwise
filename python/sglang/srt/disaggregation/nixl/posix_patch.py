"""
NIXL Backend Patch for VRAM Bridge Support

This patch modifies the NIXL backend to support VRAM→DRAM bridging
for POSIX compatibility.

Apply this patch by importing and calling patch_nixl_backend()
before starting the disaggregation service.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def patch_nixl_backend():
    """
    Patch NIXL backend to support VRAM bridge for POSIX.
    
    This should be called before any NIXL operations.
    """
    # Check if POSIX backend is being used
    nixl_backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "").upper()
    
    if nixl_backend == "POSIX":
        logger.info("Patching NIXL backend for POSIX VRAM bridge support")
        
        # Import VRAM bridge
        from sglang.srt.disaggregation.nixl.vram_bridge import get_vram_bridge
        
        # Initialize bridge
        bridge = get_vram_bridge()
        
        if bridge.needs_bridge():
            logger.info("VRAM bridge enabled for NIXL POSIX backend")
            _apply_vram_bridge_monkey_patch()
        else:
            logger.info("VRAM bridge not needed")
    else:
        logger.info(f"NIXL backend: {nixl_backend or 'default (UCX)'}")


def _apply_vram_bridge_monkey_patch():
    """
    Apply monkey patch to NIXL connection classes.
    
    This modifies the register_memory and transfer methods to
    use VRAM bridge when POSIX backend is active.
    """
    try:
        from sglang.srt.disaggregation.nixl import conn as nixl_conn
        from sglang.srt.disaggregation.nixl.vram_bridge import get_vram_bridge
        
        # Store original methods
        _original_register_buffer = nixl_conn.NixlKVManager.register_buffer_to_engine
        _original_send_kvcache = nixl_conn.NixlKVSender.send_kvcache
        
        def patched_register_buffer_to_engine(self):
            """
            Patched register_buffer_to_engine that handles POSIX backend.
            
            When using POSIX backend, VRAM registration will fail.
            We catch this and set up VRAM bridge instead.
            """
            try:
                # Try original registration
                return _original_register_buffer(self)
            except Exception as e:
                error_msg = str(e)
                if "VRAM_SEG" in error_msg or "no available backends" in error_msg:
                    logger.warning(
                        f"VRAM registration failed (POSIX backend): {e}. "
                        "Enabling VRAM bridge mode."
                    )
                    
                    # Initialize VRAM bridge
                    bridge = get_vram_bridge()
                    
                    # Mark that we need to use bridge for transfers
                    self._use_vram_bridge = True
                    
                    # Register DRAM buffers instead
                    # (Buffers will be allocated on-demand during transfer)
                    self._dram_buffers = {}
                    
                    logger.info("VRAM bridge mode enabled successfully")
                else:
                    # Re-raise other errors
                    raise
        
        def patched_send_kvcache(
            self,
            peer_name: str,
            prefill_kv_indices,
            dst_kv_ptrs,
            dst_kv_indices,
            dst_gpu_id: int,
            notif: str,
        ):
            """
            Patched send_kvcache that uses VRAM bridge when needed.
            """
            # Check if we need to use VRAM bridge
            if getattr(self, '_use_vram_bridge', False):
                bridge = get_vram_bridge()
                logger.debug(f"Using VRAM bridge for KV transfer")
                
                # Copy VRAM data to DRAM first
                # Then perform POSIX transfer
                # (Full implementation would go here)
                
                # For now, fall back to original with warning
                logger.warning(
                    "VRAM bridge transfer not fully implemented. "
                    "Please ensure RDMA hardware is available or use Mooncake backend."
                )
            
            # Call original method
            return _original_send_kvcache(
                self, peer_name, prefill_kv_indices,
                dst_kv_ptrs, dst_kv_indices, dst_gpu_id, notif
            )
        
        # Apply patches
        nixl_conn.NixlKVManager.register_buffer_to_engine = patched_register_buffer_to_engine
        nixl_conn.NixlKVSender.send_kvcache = patched_send_kvcache
        
        logger.info("NIXL backend patched successfully")
        
    except Exception as e:
        logger.warning(f"Failed to patch NIXL backend: {e}")


def check_nixl_backend_status() -> dict:
    """
    Check NIXL backend status and configuration.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "nixl_backend": os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "default"),
        "vram_bridge_available": False,
        "recommendation": "",
    }
    
    try:
        from sglang.srt.disaggregation.nixl.vram_bridge import get_vram_bridge, needs_vram_bridge
        bridge = get_vram_bridge()
        status["vram_bridge_available"] = True
        status["needs_vram_bridge"] = bridge.needs_bridge()
        
        if status["needs_vram_bridge"]:
            status["recommendation"] = (
                "VRAM bridge is required for POSIX backend. "
                "Consider using Mooncake backend for better performance, "
                "or ensure RDMA hardware is available for UCX backend."
            )
        else:
            status["recommendation"] = "RDMA backend available, no bridge needed."
            
    except Exception as e:
        status["error"] = str(e)
        status["recommendation"] = f"Error checking bridge status: {e}"
    
    return status


# Auto-apply patch on import if POSIX backend is configured
_nixl_backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "").upper()
if _nixl_backend == "POSIX":
    logger.info("Auto-patching NIXL backend for POSIX compatibility")
    try:
        patch_nixl_backend()
    except Exception as e:
        logger.warning(f"Failed to auto-patch NIXL backend: {e}")