"""
KVTuner KV Cache Quantization Adapter for SGLang

This module integrates KVTuner's flexible quantization into SGLang's KV cache management.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class KVTunerQuantConfig:
    """Configuration for KVTuner KV Cache quantization.
    
    Attributes:
        nbits_key: Quantization bits for keys (2, 4, or 8)
        nbits_value: Quantization bits for values (2, 4, or 8)
        asym: Whether to use asymmetric quantization
        axis_key: Axis for key quantization (0=per-token, 1=per-channel)
        axis_value: Axis for value quantization (0=per-token, 1=per-channel)
        q_group_size: Group size for quantization
        residual_length: Length of residual tokens kept in full precision
        compute_dtype: Compute dtype for the model
    """
    nbits_key: int = 4
    nbits_value: int = 4
    asym: bool = False
    axis_key: int = 0
    axis_value: int = 0
    q_group_size: int = 64
    residual_length: int = 128
    compute_dtype: torch.dtype = torch.float16
    
    def __post_init__(self):
        assert self.nbits_key in [2, 4, 8], f"nbits_key must be 2, 4, or 8, got {self.nbits_key}"
        assert self.nbits_value in [2, 4, 8], f"nbits_value must be 2, 4, or 8, got {self.nbits_value}"
        assert self.axis_key in [0, 1], f"axis_key must be 0 or 1, got {self.axis_key}"
        assert self.axis_value in [0, 1], f"axis_value must be 0 or 1, got {self.axis_value}"


class KVTunerVanillaQuantizer:
    """
    Vanilla Quantizer implementation adapted from KVTuner.
    Supports symmetric and asymmetric quantization.
    """
    
    def __init__(self, nbits: int, asym: bool, compute_dtype: torch.dtype):
        self.nbits = nbits
        self.asym = asym
        self.compute_dtype = compute_dtype
        self.q_max = 2 ** (nbits - 1) - 1
        self.q_min = -(2 ** (nbits - 1))
    
    def quantize(self, tensor: torch.Tensor, q_group_size: int, axis: int) -> "KVTunerQuantizedTensor":
        """Quantize a tensor.
        
        Args:
            tensor: Input tensor of shape [..., seq_len, head_dim]
            q_group_size: Group size for quantization
            axis: 0 for per-token, 1 for per-channel
            
        Returns:
            KVTunerQuantizedTensor containing quantized data and metadata
        """
        original_shape = tensor.shape
        
        # Handle axis
        if axis == 1:
            # Per-channel: transpose to make channel the last dimension
            max_dim = len(tensor.shape) - 1
            tensor = tensor.transpose(max_dim - 1, max_dim).contiguous()
        
        # Reshape for group-wise quantization
        if q_group_size == -1:
            q_group_size = tensor.shape[-1]
        
        rs = tensor.reshape(-1, q_group_size)
        
        # Compute scales and zeros
        if self.asym:
            _max, _min = rs.max(dim=1).values, rs.min(dim=1).values
            scale = (_max - _min).clamp(min=1e-5).div(self.q_max - self.q_min)
            zeros = (_min / scale).round() - self.q_min
            quant = (torch.round(rs / scale.unsqueeze(1) - zeros.unsqueeze(1))).clamp(
                self.q_min, self.q_max
            ).to(torch.int8)
        else:
            scale = rs.abs().max(dim=1).values.clamp(min=1e-5).div(self.q_max)
            zeros = None
            quant = torch.round(rs / scale.unsqueeze(1)).clamp(
                self.q_min, self.q_max
            ).to(torch.int8)
        
        return KVTunerQuantizedTensor(
            quant, scale, zeros, original_shape, axis, self
        )


class KVTunerQuantizedTensor:
    """Container for quantized tensor with metadata."""
    
    def __init__(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zeros: Optional[torch.Tensor],
        original_shape: Tuple[int, ...],
        axis: int,
        quantizer: KVTunerVanillaQuantizer
    ):
        self.tensor = tensor
        self.scale = scale
        self.zeros = zeros
        self.original_shape = original_shape
        self.axis = axis
        self.quantizer = quantizer
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to original dtype."""
        if self.quantizer.asym:
            dequant = (self.tensor.to(self.quantizer.compute_dtype) + self.zeros.unsqueeze(1)) * self.scale.unsqueeze(1)
        else:
            dequant = self.tensor.to(self.quantizer.compute_dtype) * self.scale.unsqueeze(1)
        
        # Reshape back to original shape
        dequant = dequant.view(self.original_shape)
        
        # Transpose back if per-channel
        if self.axis == 1:
            max_dim = len(self.original_shape) - 1
            dequant = dequant.transpose(max_dim - 1, max_dim)
        
        return dequant
    
    @property
    def nbytes(self) -> int:
        """Get memory usage in bytes."""
        total = self.tensor.nbytes + self.scale.nbytes
        if self.zeros is not None:
            total += self.zeros.nbytes
        return total


class KVTunerQuantizationMethod:
    """
    Quantization method for integrating KVTuner into SGLang's attention layers.
    """
    
    def __init__(self, quant_config: KVTunerQuantConfig):
        self.quant_config = quant_config
        self.key_quantizers: Dict[Tuple[int, int], KVTunerVanillaQuantizer] = {}
        self.value_quantizers: Dict[Tuple[int, int], KVTunerVanillaQuantizer] = {}
    
    def _get_key_quantizer(self, nbits: int, asym: bool) -> KVTunerVanillaQuantizer:
        key = (nbits, asym)
        if key not in self.key_quantizers:
            self.key_quantizers[key] = KVTunerVanillaQuantizer(
                nbits, asym, self.quant_config.compute_dtype
            )
        return self.key_quantizers[key]
    
    def _get_value_quantizer(self, nbits: int, asym: bool) -> KVTunerVanillaQuantizer:
        key = (nbits, asym)
        if key not in self.value_quantizers:
            self.value_quantizers[key] = KVTunerVanillaQuantizer(
                nbits, asym, self.quant_config.compute_dtype
            )
        return self.value_quantizers[key]
    
    def quantize_key(self, key: torch.Tensor) -> KVTunerQuantizedTensor:
        """Quantize key tensor."""
        quantizer = self._get_key_quantizer(
            self.quant_config.nbits_key, self.quant_config.asym
        )
        return quantizer.quantize(
            key, self.quant_config.q_group_size, self.quant_config.axis_key
        )
    
    def quantize_value(self, value: torch.Tensor) -> KVTunerQuantizedTensor:
        """Quantize value tensor."""
        quantizer = self._get_value_quantizer(
            self.quant_config.nbits_value, self.quant_config.asym
        )
        return quantizer.quantize(
            value, self.quant_config.q_group_size, self.quant_config.axis_value
        )
    
    def create_weights(self, layer: nn.Module):
        """Create quantization-related weights for attention layer."""
        # Store quantization configuration
        layer.kv_tuner_config = self.quant_config
    
    def apply(self, layer: nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")


class KVTunerQuantizedKVCache:
    """
    Quantized KV Cache storage with residual cache support.
    Based on KVTuner's FlexibleQuantizedCache design.
    """
    
    def __init__(
        self,
        config: KVTunerQuantConfig,
        num_layers: int,
        device: str = "cuda"
    ):
        self.config = config
        self.num_layers = num_layers
        self.device = device
        
        # Quantization method
        self.quant_method = KVTunerQuantizationMethod(config)
        
        # Residual cache (full precision)
        self.residual_keys: Dict[int, torch.Tensor] = {}
        self.residual_values: Dict[int, torch.Tensor] = {}
        
        # Quantized cache
        self.quantized_keys: Dict[int, KVTunerQuantizedTensor] = {}
        self.quantized_values: Dict[int, KVTunerQuantizedTensor] = {}
        
        # Track sequence lengths
        self.seq_lengths: Dict[int, int] = {}
    
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key/value states.
        
        Args:
            layer_idx: Layer index
            key_states: New key states [batch, heads, seq_len, head_dim]
            value_states: New value states [batch, heads, seq_len, head_dim]
            cache_positions: Positions in cache
            
        Returns:
            Tuple of (combined_keys, combined_values)
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Initialize layer cache if needed
        if layer_idx not in self.residual_keys:
            self.residual_keys[layer_idx] = torch.zeros(
                (0, num_heads, 0, head_dim),
                dtype=key_states.dtype,
                device=key_states.device
            )
            self.residual_values[layer_idx] = torch.zeros(
                (0, num_heads, 0, head_dim),
                dtype=value_states.dtype,
                device=value_states.device
            )
            self.seq_lengths[layer_idx] = 0
        
        # Get current residual length
        current_residual_len = self.residual_keys[layer_idx].shape[2]
        
        # Check if we need to quantize old residual
        if current_residual_len + seq_len > self.config.residual_length:
            # Quantize and move old residual to quantized cache
            if current_residual_len > 0:
                old_keys = self.residual_keys[layer_idx]
                old_values = self.residual_values[layer_idx]
                
                # Quantize
                if layer_idx not in self.quantized_keys:
                    self.quantized_keys[layer_idx] = self.quant_method.quantize_key(old_keys)
                    self.quantized_values[layer_idx] = self.quant_method.quantize_value(old_values)
                else:
                    # Merge with existing quantized cache
                    existing_keys = self.quantized_keys[layer_idx].dequantize()
                    existing_values = self.quantized_values[layer_idx].dequantize()
                    merged_keys = torch.cat([existing_keys, old_keys], dim=2)
                    merged_values = torch.cat([existing_values, old_values], dim=2)
                    self.quantized_keys[layer_idx] = self.quant_method.quantize_key(merged_keys)
                    self.quantized_values[layer_idx] = self.quant_method.quantize_value(merged_values)
                
                # Clear residual
                self.residual_keys[layer_idx] = torch.zeros(
                    (0, num_heads, 0, head_dim),
                    dtype=key_states.dtype,
                    device=key_states.device
                )
                self.residual_values[layer_idx] = torch.zeros(
                    (0, num_heads, 0, head_dim),
                    dtype=value_states.dtype,
                    device=value_states.device
                )
                current_residual_len = 0
        
        # Append new states to residual
        if self.residual_keys[layer_idx].shape[2] == 0:
            self.residual_keys[layer_idx] = key_states
            self.residual_values[layer_idx] = value_states
        else:
            self.residual_keys[layer_idx] = torch.cat(
                [self.residual_keys[layer_idx], key_states], dim=2
            )
            self.residual_values[layer_idx] = torch.cat(
                [self.residual_values[layer_idx], value_states], dim=2
            )
        
        self.seq_lengths[layer_idx] += seq_len
        
        # Return combined cache (dequantized + residual)
        return self.get_kv_buffer(layer_idx)
    
    def get_kv_buffer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full KV buffer for layer (dequantized + residual)."""
        keys_to_return = [self.residual_keys[layer_idx]]
        values_to_return = [self.residual_values[layer_idx]]
        
        if layer_idx in self.quantized_keys:
            dequant_keys = self.quantized_keys[layer_idx].dequantize()
            dequant_values = self.quantized_values[layer_idx].dequantize()
            keys_to_return.insert(0, dequant_keys)
            values_to_return.insert(0, dequant_values)
        
        # Filter empty tensors
        keys_to_return = [k for k in keys_to_return if k.shape[2] > 0]
        values_to_return = [v for v in values_to_return if v.shape[2] > 0]
        
        if len(keys_to_return) == 0:
            return (
                torch.zeros((0,), device=self.device),
                torch.zeros((0,), device=self.device)
            )
        
        return (
            torch.cat(keys_to_return, dim=2),
            torch.cat(values_to_return, dim=2)
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        residual_bytes = 0
        quantized_bytes = 0
        
        for layer_idx in range(self.num_layers):
            if layer_idx in self.residual_keys:
                residual_bytes += self.residual_keys[layer_idx].nbytes
                residual_bytes += self.residual_values[layer_idx].nbytes
            
            if layer_idx in self.quantized_keys:
                quantized_bytes += self.quantized_keys[layer_idx].nbytes
                quantized_bytes += self.quantized_values[layer_idx].nbytes
        
        return {
            "residual_mb": residual_bytes / (1024 * 1024),
            "quantized_mb": quantized_bytes / (1024 * 1024),
            "total_mb": (residual_bytes + quantized_bytes) / (1024 * 1024)
        }
