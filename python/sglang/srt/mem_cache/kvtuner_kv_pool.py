"""
KVTuner Quantized KV Cache Pool for SGLang

This module provides a memory pool that integrates KVTuner quantization
with SGLang's existing memory management.

Key Features:
- Prefill/Decode mode-aware quantization
- Residual cache management (recent tokens in full precision)
- Efficient quantization/dequantization paths
- Memory statistics and debugging support
"""

import logging
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field

import torch

from sglang.srt.layers.quantization.kvtuner_quant import (
    KVTunerQuantConfig,
    KVTunerQuantizationMethod,
    KVTunerQuantizedTensor,
    KVTunerVanillaQuantizer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, KVCache
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


@dataclass
class KVTunerModeConfig:
    """Mode-specific quantization configuration for Prefill/Decode.
    
    Attributes:
        nbits_key: Quantization bits for keys
        nbits_value: Quantization bits for values
        residual_length: Number of recent tokens to keep in full precision
        q_group_size: Group size for quantization
        enable_quantization: Whether to enable quantization for this mode
    """
    nbits_key: int = 4
    nbits_value: int = 4
    residual_length: int = 128
    q_group_size: int = 64
    enable_quantization: bool = True


class KVTunerMHATokenToKVPool(MHATokenToKVPool):
    """
    MHA Token to KV Pool with KVTuner quantization support.
    
    This class extends MHATokenToKVPool to support flexible quantization
    of KV cache using KVTuner's quantization algorithms with Prefill/Decode
    mode awareness.
    
    Architecture:
    - Inherits all buffer management from MHATokenToKVPool
    - Adds quantized cache storage in parallel
    - Uses residual cache for recent tokens (full precision)
    - Quantizes older tokens to save memory
    - Dequantizes on-demand when reading cache
    """
    
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        # KVTuner specific args
        kvtuner_config: Optional[KVTunerQuantConfig] = None,
        # Mode-specific configs (optional, overrides kvtuner_config)
        prefill_config: Optional[KVTunerModeConfig] = None,
        decode_config: Optional[KVTunerModeConfig] = None,
    ):
        """Initialize with KVTuner quantization support.
        
        Args:
            kvtuner_config: Base KVTuner quantization configuration
            prefill_config: Mode-specific config for Prefill (EXTEND) mode
            decode_config: Mode-specific config for Decode mode
        """
        # Store KVTuner config before calling parent init
        self.kvtuner_config = kvtuner_config
        self.enable_kvtuner = kvtuner_config is not None
        
        # Mode-specific configs
        self.prefill_config = prefill_config
        self.decode_config = decode_config
        
        if self.enable_kvtuner:
            logger.info(f"Initializing KVTuner quantized KV cache with config: {kvtuner_config}")
            if prefill_config:
                logger.info(f"  Prefill config: nbits_k={prefill_config.nbits_key}, "
                           f"nbits_v={prefill_config.nbits_value}, "
                           f"residual={prefill_config.residual_length}")
            if decode_config:
                logger.info(f"  Decode config: nbits_k={decode_config.nbits_key}, "
                           f"nbits_v={decode_config.nbits_value}, "
                           f"residual={decode_config.residual_length}")
        
        # Call parent init
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )
        
        if self.enable_kvtuner:
            self._init_kvtuner_quantization()
    
    def _init_kvtuner_quantization(self):
        """Initialize KVTuner quantization structures."""
        assert self.kvtuner_config is not None
        
        # Initialize quantization methods for base and mode-specific configs
        self.kvtuner_quant_method = KVTunerQuantizationMethod(self.kvtuner_config)
        
        # Mode-specific quantizers
        self.prefill_quant_method: Optional[KVTunerQuantizationMethod] = None
        self.decode_quant_method: Optional[KVTunerQuantizationMethod] = None
        
        if self.prefill_config:
            prefill_cfg = KVTunerQuantConfig(
                nbits_key=self.prefill_config.nbits_key,
                nbits_value=self.prefill_config.nbits_value,
                asym=self.kvtuner_config.asym,
                axis_key=self.kvtuner_config.axis_key,
                axis_value=self.kvtuner_config.axis_value,
                q_group_size=self.prefill_config.q_group_size,
                residual_length=self.prefill_config.residual_length,
                compute_dtype=self.kvtuner_config.compute_dtype,
            )
            self.prefill_quant_method = KVTunerQuantizationMethod(prefill_cfg)
        
        if self.decode_config:
            decode_cfg = KVTunerQuantConfig(
                nbits_key=self.decode_config.nbits_key,
                nbits_value=self.decode_config.nbits_value,
                asym=self.kvtuner_config.asym,
                axis_key=self.kvtuner_config.axis_key,
                axis_value=self.kvtuner_config.axis_value,
                q_group_size=self.decode_config.q_group_size,
                residual_length=self.decode_config.residual_length,
                compute_dtype=self.kvtuner_config.compute_dtype,
            )
            self.decode_quant_method = KVTunerQuantizationMethod(decode_cfg)
        
        # Quantized cache storage per layer
        # Format: {layer_idx: {slot_idx: quantized_tensor}}
        self.quantized_k_cache: List[Dict[int, KVTunerQuantizedTensor]] = [
            {} for _ in range(self.layer_num)
        ]
        self.quantized_v_cache: List[Dict[int, KVTunerQuantizedTensor]] = [
            {} for _ in range(self.layer_num)
        ]
        
        # Track which slots are quantized vs full precision
        self.quantized_slots: List[Set[int]] = [set() for _ in range(self.layer_num)]
        
        # Track token positions for residual cache management
        # Maps slot_idx -> token_position (for determining age)
        self.slot_positions: List[Dict[int, int]] = [{} for _ in range(self.layer_num)]
        
        # Current sequence length per layer (for residual cache management)
        self.seq_lengths: List[int] = [0 for _ in range(self.layer_num)]
        
        # Statistics
        self.stats = {
            "total_quantize_ops": 0,
            "total_dequantize_ops": 0,
            "prefill_quantize_count": 0,
            "decode_quantize_count": 0,
        }
        
        # Get effective residual length
        self._residual_length = self.kvtuner_config.residual_length
        
        logger.info(f"KVTuner quantization initialized: "
                   f"nbits_k={self.kvtuner_config.nbits_key}, "
                   f"nbits_v={self.kvtuner_config.nbits_value}, "
                   f"asym={self.kvtuner_config.asym}, "
                   f"residual_length={self._residual_length}")
    
    def _get_quant_method_for_mode(self, forward_mode: Optional[ForwardMode]) -> KVTunerQuantizationMethod:
        """Get the appropriate quantization method for the current forward mode."""
        if forward_mode is None:
            return self.kvtuner_quant_method
        
        if forward_mode.is_decode_or_idle():
            if self.decode_quant_method:
                return self.decode_quant_method
        elif forward_mode.is_extend() or forward_mode.is_extend_or_draft_extend_or_mixed():
            if self.prefill_quant_method:
                return self.prefill_quant_method
        
        return self.kvtuner_quant_method
    
    def _get_residual_length_for_mode(self, forward_mode: Optional[ForwardMode]) -> int:
        """Get the appropriate residual length for the current forward mode."""
        if forward_mode is None:
            return self._residual_length
        
        if forward_mode.is_decode_or_idle():
            if self.decode_config:
                return self.decode_config.residual_length
        elif forward_mode.is_extend() or forward_mode.is_extend_or_draft_extend_or_mixed():
            if self.prefill_config:
                return self.prefill_config.residual_length
        
        return self._residual_length
    
    def _should_quantize_token(
        self, 
        buffer_idx: int, 
        slot_idx: int,
        forward_mode: Optional[ForwardMode] = None,
        token_position: Optional[int] = None,
    ) -> bool:
        """Determine if a token should be quantized based on residual length policy.
        
        KVTuner keeps the most recent `residual_length` tokens in full precision.
        Older tokens are quantized to save memory.
        
        Args:
            buffer_idx: Layer buffer index
            slot_idx: Slot index in the cache
            forward_mode: Current forward mode (Prefill/Decode)
            token_position: Position of the token in the sequence
            
        Returns:
            True if the token should be quantized, False to keep in full precision
        """
        if not self.enable_kvtuner:
            return False
        
        # Check if quantization is disabled for this mode
        if forward_mode is not None:
            if forward_mode.is_decode_or_idle() and self.decode_config:
                if not self.decode_config.enable_quantization:
                    return False
            elif (forward_mode.is_extend() or forward_mode.is_extend_or_draft_extend_or_mixed()) and self.prefill_config:
                if not self.prefill_config.enable_quantization:
                    return False
        
        # Get effective residual length for current mode
        residual_length = self._get_residual_length_for_mode(forward_mode)
        
        # If no position info, use simplified policy
        if token_position is None:
            # Default: quantize all tokens (simplified)
            # In production, would track token age
            return True
        
        # Calculate distance from the end of sequence
        seq_length = self.seq_lengths[buffer_idx]
        distance_from_end = seq_length - token_position
        
        # Keep recent tokens in full precision
        if distance_from_end < residual_length:
            return False
        
        return True
    
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        forward_batch: Optional[Any] = None,
    ):
        """Set KV buffer with optional quantization.
        
        This overrides the parent method to add KVTuner quantization support.
        
        Args:
            layer: RadixAttention layer
            loc: Location indices in the cache
            cache_k: Key cache tensor [num_tokens, head_num, head_dim]
            cache_v: Value cache tensor [num_tokens, head_num, v_head_dim]
            k_scale: Optional scaling factor for keys
            v_scale: Optional scaling factor for values
            layer_id_override: Optional layer ID override
            forward_batch: Optional forward batch to get mode info
        """
        if not self.enable_kvtuner:
            # Use parent's implementation if KVTuner not enabled
            super().set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override)
            return
        
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        buffer_idx = layer_id - self.start_layer
        
        # Get forward mode if available
        forward_mode = None
        if forward_batch is not None:
            forward_mode = getattr(forward_batch, 'forward_mode', None)
        
        # Get quantization method for current mode
        quant_method = self._get_quant_method_for_mode(forward_mode)
        
        # Get positions if available (for residual cache management)
        positions = None
        if forward_batch is not None:
            positions = getattr(forward_batch, 'positions', None)
        
        # Update sequence length
        num_new_tokens = loc.numel()
        self.seq_lengths[buffer_idx] += num_new_tokens
        
        # Process each token
        for i, slot_idx in enumerate(loc.tolist()):
            # Get token position for residual cache management
            token_position = None
            if positions is not None and i < len(positions):
                token_position = int(positions[i].item()) if torch.is_tensor(positions[i]) else positions[i]
            
            # Determine if we should quantize this token
            should_quantize = self._should_quantize_token(
                buffer_idx, slot_idx, forward_mode, token_position
            )
            
            if should_quantize:
                # Extract single token tensors
                k_token = cache_k[i:i+1]  # [1, heads, head_dim]
                v_token = cache_v[i:i+1]  # [1, heads, v_head_dim]
                
                # Quantize and store
                q_k = quant_method.quantize_key(k_token)
                q_v = quant_method.quantize_value(v_token)
                
                self.quantized_k_cache[buffer_idx][slot_idx] = q_k
                self.quantized_v_cache[buffer_idx][slot_idx] = q_v
                self.quantized_slots[buffer_idx].add(slot_idx)
                
                # Track position
                if token_position is not None:
                    self.slot_positions[buffer_idx][slot_idx] = token_position
                
                # Update statistics
                self.stats["total_quantize_ops"] += 1
                if forward_mode is not None:
                    if forward_mode.is_decode_or_idle():
                        self.stats["decode_quantize_count"] += 1
                    elif forward_mode.is_extend() or forward_mode.is_extend_or_draft_extend_or_mixed():
                        self.stats["prefill_quantize_count"] += 1
            else:
                # Keep in full precision - remove from quantized cache if present
                if slot_idx in self.quantized_slots[buffer_idx]:
                    del self.quantized_k_cache[buffer_idx][slot_idx]
                    del self.quantized_v_cache[buffer_idx][slot_idx]
                    self.quantized_slots[buffer_idx].discard(slot_idx)
                    if slot_idx in self.slot_positions[buffer_idx]:
                        del self.slot_positions[buffer_idx][slot_idx]
        
        # Always store in underlying buffer for compatibility
        # In a fully optimized implementation, quantized slots wouldn't be stored here
        super().set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override)
    
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV buffer with dequantization if needed.
        
        This method dequantizes quantized slots on-demand.
        
        Args:
            layer_id: Layer ID
            
        Returns:
            Tuple of (key_buffer, value_buffer)
        """
        if not self.enable_kvtuner:
            return super().get_kv_buffer(layer_id)
        
        buffer_idx = layer_id - self.start_layer
        
        # Get base buffers from parent
        k_buffer = self._get_key_buffer(layer_id)
        v_buffer = self._get_value_buffer(layer_id)
        
        # Dequantize any quantized slots
        # This is done on-demand - in production would be optimized with fused kernels
        if self.quantized_slots[buffer_idx]:
            self.stats["total_dequantize_ops"] += len(self.quantized_slots[buffer_idx])
            
            for slot_idx in self.quantized_slots[buffer_idx]:
                if slot_idx in self.quantized_k_cache[buffer_idx]:
                    # Dequantize and update buffer
                    q_k = self.quantized_k_cache[buffer_idx][slot_idx]
                    q_v = self.quantized_v_cache[buffer_idx][slot_idx]
                    
                    k_dequant = q_k.dequantize()
                    v_dequant = q_v.dequantize()
                    
                    # Update buffer (in-place modification)
                    # Handle batch dimension if present
                    if k_buffer.dim() == 3 and k_dequant.dim() == 3:
                        k_buffer[slot_idx:slot_idx+1] = k_dequant.squeeze(0)
                        v_buffer[slot_idx:slot_idx+1] = v_dequant.squeeze(0)
                    elif k_buffer.dim() == 2 and k_dequant.dim() == 3:
                        k_buffer[slot_idx:slot_idx+1] = k_dequant.squeeze(0).squeeze(0)
                        v_buffer[slot_idx:slot_idx+1] = v_dequant.squeeze(0).squeeze(0)
                    else:
                        # Fallback: try direct assignment
                        try:
                            k_buffer[slot_idx] = k_dequant.flatten()
                            v_buffer[slot_idx] = v_dequant.flatten()
                        except Exception as e:
                            logger.warning(f"Failed to dequantize slot {slot_idx}: {e}")
        
        return k_buffer, v_buffer
    
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Get key buffer with dequantization."""
        return self.get_kv_buffer(layer_id)[0]
    
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Get value buffer with dequantization."""
        return self.get_kv_buffer(layer_id)[1]
    
    def clear_quantized_cache(self, layer_id: Optional[int] = None):
        """Clear quantized cache for a layer or all layers.
        
        Args:
            layer_id: Optional layer ID to clear. If None, clears all layers.
        """
        if layer_id is not None:
            buffer_idx = layer_id - self.start_layer
            self.quantized_k_cache[buffer_idx].clear()
            self.quantized_v_cache[buffer_idx].clear()
            self.quantized_slots[buffer_idx].clear()
            self.slot_positions[buffer_idx].clear()
        else:
            for i in range(self.layer_num):
                self.quantized_k_cache[i].clear()
                self.quantized_v_cache[i].clear()
                self.quantized_slots[i].clear()
                self.slot_positions[i].clear()
    
    def get_quantized_memory_stats(self) -> dict:
        """Get memory statistics for quantized cache.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.enable_kvtuner:
            return {"enabled": False}
        
        total_quantized_slots = sum(len(s) for s in self.quantized_slots)
        total_quantized_bytes = 0
        total_scale_bytes = 0
        
        for layer_idx in range(self.layer_num):
            for slot_idx in self.quantized_slots[layer_idx]:
                if slot_idx in self.quantized_k_cache[layer_idx]:
                    q_k = self.quantized_k_cache[layer_idx][slot_idx]
                    q_v = self.quantized_v_cache[layer_idx][slot_idx]
                    total_quantized_bytes += q_k.tensor.nbytes + q_v.tensor.nbytes
                    total_scale_bytes += q_k.scale.nbytes + q_v.scale.nbytes
                    if q_k.zeros is not None:
                        total_quantized_bytes += q_k.zeros.nbytes
                    if q_v.zeros is not None:
                        total_quantized_bytes += q_v.zeros.nbytes
        
        # Calculate what full precision would be
        slot_bytes = self.head_num * self.head_dim * self.dtype.itemsize * 2  # K + V
        full_precision_bytes = total_quantized_slots * slot_bytes
        
        savings_bytes = full_precision_bytes - (total_quantized_bytes + total_scale_bytes)
        savings_mb = savings_bytes / (1024 * 1024)
        
        return {
            "enabled": True,
            "quantized_slots": total_quantized_slots,
            "quantized_data_mb": total_quantized_bytes / (1024 * 1024),
            "quantized_scale_mb": total_scale_bytes / (1024 * 1024),
            "quantized_total_mb": (total_quantized_bytes + total_scale_bytes) / (1024 * 1024),
            "full_precision_mb": full_precision_bytes / (1024 * 1024),
            "savings_mb": savings_mb,
            "compression_ratio": full_precision_bytes / max(total_quantized_bytes + total_scale_bytes, 1),
            "stats": self.stats.copy(),
        }
    
    def get_memory_usage(self) -> dict:
        """Get comprehensive memory usage statistics.
        
        Returns:
            Dictionary with detailed memory usage breakdown
        """
        base_stats = {
            "base_k_buffer_mb": sum(b.nbytes for b in self.k_buffer) / (1024 * 1024),
            "base_v_buffer_mb": sum(b.nbytes for b in self.v_buffer) / (1024 * 1024),
        }
        
        quantized_stats = self.get_quantized_memory_stats()
        
        return {
            **base_stats,
            **quantized_stats,
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_quantize_ops": 0,
            "total_dequantize_ops": 0,
            "prefill_quantize_count": 0,
            "decode_quantize_count": 0,
        }


class KVTunerMLATokenToKVPool:
    """Placeholder for MLA (Multi-Head Latent Attention) with KVTuner support.
    
    MLA is used in models like DeepSeek-v2/v3. KVTuner can be applied to the
    compressed latent representations.
    
    TODO: Implement full MLA support with:
    - KV compression for latent attention
    - Mode-aware quantization for MLA cache
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "KVTunerMLATokenToKVPool is not yet implemented. "
            "Please use KVTunerMHATokenToKVPool for standard MHA models."
        )
