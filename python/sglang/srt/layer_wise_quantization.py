# SPDX-License-Identifier: Apache-2.0
"""
Layer-wise KV Cache Quantization for SGLang

Different transformer layers use different quantization precision based on:
1. Layer sensitivity analysis
2. User-defined configuration
3. Automatic sensitivity-based allocation
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.kvtuner_quant import VanillaQuantizer

logger = logging.getLogger(__name__)


class LayerWiseQuantConfig:
    """Configuration for layer-wise quantization."""
    
    def __init__(
        self,
        num_layers: int,
        layer_bits: Optional[Dict[int, int]] = None,
        default_bits: int = 4,
        config_file: Optional[str] = None,
    ):
        """
        Args:
            num_layers: Total number of transformer layers
            layer_bits: Dict mapping layer_id -> bits (2/4/8)
            default_bits: Default quantization bits
            config_file: Path to JSON config file
        """
        self.num_layers = num_layers
        self.default_bits = default_bits
        
        if config_file:
            self.layer_bits = self._load_from_file(config_file)
        else:
            self.layer_bits = layer_bits or {}
        
        # Fill unspecified layers with default
        for i in range(num_layers):
            if i not in self.layer_bits:
                self.layer_bits[i] = default_bits
    
    def _load_from_file(self, config_file: str) -> Dict[int, int]:
        """Load layer bits from JSON config file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        layer_bits = {}
        
        # Parse layer ranges
        for layer_range, bits in config.get('layer_bits', {}).items():
            if '-' in layer_range:
                start, end = map(int, layer_range.split('-'))
                for i in range(start, end + 1):
                    layer_bits[i] = bits
            else:
                layer_bits[int(layer_range)] = bits
        
        return layer_bits
    
    def get_bits(self, layer_id: int) -> int:
        """Get quantization bits for a specific layer."""
        return self.layer_bits.get(layer_id, self.default_bits)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'num_layers': self.num_layers,
            'layer_bits': self.layer_bits,
            'default_bits': self.default_bits,
        }


class SensitivityAnalyzer:
    """
    Analyze layer sensitivity to quantization.
    
    More sensitive layers (higher perplexity impact) should use higher precision.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.sensitivity_scores = {}
    
    def analyze(self, calibration_data: List[torch.Tensor]) -> Dict[int, float]:
        """
        Analyze sensitivity of each layer to quantization.
        
        Returns:
            Dict mapping layer_id -> sensitivity_score (higher = more sensitive)
        """
        logger.info("Analyzing layer sensitivity to quantization...")
        
        # Baseline: compute perplexity without quantization
        baseline_ppl = self._compute_perplexity(calibration_data)
        
        # Test each layer with quantization
        for layer_id in range(len(self.model.model.layers)):
            # Quantize only this layer
            ppl = self._compute_perplexity_with_quantization(
                calibration_data, [layer_id]
            )
            
            # Sensitivity = perplexity increase
            self.sensitivity_scores[layer_id] = ppl - baseline_ppl
            
            logger.info(f"Layer {layer_id}: sensitivity = {self.sensitivity_scores[layer_id]:.4f}")
        
        return self.sensitivity_scores
    
    def _compute_perplexity(self, data: List[torch.Tensor]) -> float:
        """Compute perplexity on calibration data."""
        # TODO: Implement actual perplexity computation
        # For now, return dummy value
        return 10.0
    
    def _compute_perplexity_with_quantization(
        self, 
        data: List[torch.Tensor], 
        quantized_layers: List[int]
    ) -> float:
        """Compute perplexity with specific layers quantized."""
        # TODO: Implement actual perplexity computation
        return 10.5
    
    def generate_layer_bits(
        self, 
        num_layers: int,
        bits_budget: float = 4.0,
        min_bits: int = 2,
        max_bits: int = 8
    ) -> Dict[int, int]:
        """
        Generate layer-wise bits based on sensitivity analysis.
        
        Args:
            num_layers: Total number of layers
            bits_budget: Average bits per layer (budget constraint)
            min_bits: Minimum quantization bits
            max_bits: Maximum quantization bits
            
        Returns:
            Dict mapping layer_id -> bits
        """
        if not self.sensitivity_scores:
            raise ValueError("Must call analyze() first")
        
        # Normalize sensitivity scores
        max_score = max(self.sensitivity_scores.values())
        min_score = min(self.sensitivity_scores.values())
        
        normalized_scores = {}
        for layer_id, score in self.sensitivity_scores.items():
            if max_score > min_score:
                normalized_scores[layer_id] = (score - min_score) / (max_score - min_score)
            else:
                normalized_scores[layer_id] = 0.5
        
        # Allocate bits based on sensitivity
        # More sensitive layers get more bits
        layer_bits = {}
        total_bits = 0
        
        for layer_id in range(num_layers):
            sensitivity = normalized_scores.get(layer_id, 0.5)
            
            # Map sensitivity [0, 1] to bits [min_bits, max_bits]
            bits = min_bits + sensitivity * (max_bits - min_bits)
            bits = int(round(bits))
            bits = max(min_bits, min(max_bits, bits))
            
            layer_bits[layer_id] = bits
            total_bits += bits
        
        # Adjust to meet budget
        avg_bits = total_bits / num_layers
        if avg_bits > bits_budget:
            # Need to reduce bits
            reduction = (avg_bits - bits_budget) * num_layers
            sorted_layers = sorted(
                layer_bits.keys(),
                key=lambda x: normalized_scores.get(x, 0.5)
            )
            
            for layer_id in sorted_layers:
                if reduction <= 0:
                    break
                if layer_bits[layer_id] > min_bits:
                    layer_bits[layer_id] -= 1
                    reduction -= 1
        
        logger.info(f"Generated layer bits (avg={sum(layer_bits.values())/num_layers:.2f}):")
        for layer_id, bits in layer_bits.items():
            logger.info(f"  Layer {layer_id}: {bits} bits")
        
        return layer_bits


class LayerWiseQuantizer:
    """
    Manage layer-wise quantization with different precision per layer.
    """
    
    def __init__(
        self,
        num_layers: int,
        config: Optional[LayerWiseQuantConfig] = None,
        default_nbits_key: int = 4,
        default_nbits_value: int = 4,
    ):
        """
        Args:
            num_layers: Total number of transformer layers
            config: Layer-wise quantization config
            default_nbits_key: Default bits for key quantization
            default_nbits_value: Default bits for value quantization
        """
        self.num_layers = num_layers
        self.config = config
        self.default_nbits_key = default_nbits_key
        self.default_nbits_value = default_nbits_value
        
        # Create quantizer for each layer
        self.quantizers = {}
        for layer_id in range(num_layers):
            bits = self._get_layer_bits(layer_id)
            self.quantizers[layer_id] = VanillaQuantizer(
                nbits_key=bits,
                nbits_value=bits,
            )
        
        logger.info(f"LayerWiseQuantizer initialized for {num_layers} layers")
    
    def _get_layer_bits(self, layer_id: int) -> int:
        """Get quantization bits for a layer."""
        if self.config:
            return self.config.get_bits(layer_id)
        return self.default_nbits_key
    
    def quantize_layer(
        self,
        layer_id: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Quantize KV cache for a specific layer.
        
        Args:
            layer_id: Layer identifier
            k_cache: Key cache tensor
            v_cache: Value cache tensor
            
        Returns:
            Tuple of (quantized_k, quantized_v, metadata)
        """
        quantizer = self.quantizers[layer_id]
        qk, qv, scales_k, scales_v = quantizer.quantize(k_cache, v_cache)
        
        metadata = {
            'layer_id': layer_id,
            'bits': self._get_layer_bits(layer_id),
            'scales_k': scales_k,
            'scales_v': scales_v,
        }
        
        return qk, qv, metadata
    
    def dequantize_layer(
        self,
        layer_id: int,
        qk: torch.Tensor,
        qv: torch.Tensor,
        metadata: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize KV cache for a specific layer.
        
        Args:
            layer_id: Layer identifier
            qk: Quantized key cache
            qv: Quantized value cache
            metadata: Quantization metadata
            
        Returns:
            Tuple of (dequantized_k, dequantized_v)
        """
        quantizer = self.quantizers[layer_id]
        scales_k = metadata.get('scales_k')
        scales_v = metadata.get('scales_v')
        
        k_cache = quantizer.dequantize(qk, scales_k, quantizer.nbits_key)
        v_cache = quantizer.dequantize(qv, scales_v, quantizer.nbits_value)
        
        return k_cache, v_cache
    
    def get_memory_savings(self) -> Dict:
        """Calculate memory savings from layer-wise quantization."""
        if not self.config:
            return {}
        
        total_bits = sum(self.config.layer_bits.values())
        avg_bits = total_bits / self.num_layers
        
        # Compare to FP16 baseline
        fp16_bits = 16
        savings_ratio = 1 - (avg_bits / fp16_bits)
        
        return {
            'avg_bits': avg_bits,
            'total_bits': total_bits,
            'savings_ratio': savings_ratio,
            'layer_distribution': self.config.layer_bits,
        }
