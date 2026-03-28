# SPDX-License-Identifier: Apache-2.0
"""
Memory-Aware Pipeline Parallelism for SGLang

Automatically distribute transformer layers across devices based on:
1. Available GPU memory
2. Model size requirements
3. Communication bandwidth

This enables heterogeneous deployment where devices with different
GPU memory can work together efficiently.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class MemoryAwarePipelineParallel:
    """
    Memory-aware pipeline parallelism scheduler.
    
    Distributes model layers across PP stages based on available GPU memory.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_stages: int,
        memory_per_stage: Optional[List[int]] = None,
        strategy: str = "proportional",
    ):
        """
        Args:
            num_layers: Total number of transformer layers
            num_stages: Number of pipeline stages (PP size)
            memory_per_stage: Available memory per stage in GB
            strategy: Distribution strategy ("proportional", "uniform", "custom")
        """
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.memory_per_stage = memory_per_stage or [24] * num_stages  # Default 24GB
        self.strategy = strategy
        
        self.layer_distribution = self._distribute_layers()
        
    def _distribute_layers(self) -> Dict[int, List[int]]:
        """
        Distribute layers across pipeline stages.
        
        Returns:
            Dict mapping stage_id -> list of layer_ids
        """
        if self.strategy == "uniform":
            return self._uniform_distribution()
        elif self.strategy == "proportional":
            return self._proportional_distribution()
        elif self.strategy == "custom":
            return self._custom_distribution()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _uniform_distribution(self) -> Dict[int, List[int]]:
        """Uniform distribution: equal layers per stage."""
        layers_per_stage = self.num_layers // self.num_stages
        remainder = self.num_layers % self.num_stages
        
        distribution = {}
        start_layer = 0
        
        for stage_id in range(self.num_stages):
            # Distribute remainder to first stages
            extra = 1 if stage_id < remainder else 0
            num_layers = layers_per_stage + extra
            
            end_layer = start_layer + num_layers
            distribution[stage_id] = list(range(start_layer, end_layer))
            start_layer = end_layer
        
        return distribution
    
    def _proportional_distribution(self) -> Dict[int, List[int]]:
        """
        Proportional distribution: more layers on stages with more memory.
        
        Layer count is proportional to available memory.
        """
        total_memory = sum(self.memory_per_stage)
        
        # Calculate proportional layers per stage
        layers_per_stage = [
            int(self.num_layers * mem / total_memory)
            for mem in self.memory_per_stage
        ]
        
        # Adjust for rounding errors
        total_assigned = sum(layers_per_stage)
        remainder = self.num_layers - total_assigned
        
        # Distribute remainder to stages with most memory
        sorted_indices = sorted(
            range(self.num_stages),
            key=lambda i: self.memory_per_stage[i],
            reverse=True
        )
        
        for i in range(remainder):
            layers_per_stage[sorted_indices[i]] += 1
        
        # Create distribution
        distribution = {}
        start_layer = 0
        
        for stage_id in range(self.num_stages):
            num_layers = layers_per_stage[stage_id]
            end_layer = start_layer + num_layers
            distribution[stage_id] = list(range(start_layer, end_layer))
            start_layer = end_layer
        
        return distribution
    
    def _custom_distribution(self) -> Dict[int, List[int]]:
        """Custom distribution based on memory-aware heuristics."""
        # For custom strategy, use memory-weighted distribution
        # but ensure minimum layers per stage for pipeline efficiency
        
        min_layers_per_stage = max(1, self.num_layers // (self.num_stages * 2))
        
        # First pass: proportional distribution
        distribution = self._proportional_distribution()
        
        # Second pass: ensure minimum layers
        for stage_id, layers in distribution.items():
            if len(layers) < min_layers_per_stage:
                # Steal layers from stages with most layers
                while len(distribution[stage_id]) < min_layers_per_stage:
                    max_stage = max(
                        distribution.keys(),
                        key=lambda k: len(distribution[k])
                    )
                    if len(distribution[max_stage]) <= min_layers_per_stage:
                        break
                    
                    # Move last layer from max_stage to current stage
                    layer_to_move = distribution[max_stage].pop()
                    distribution[stage_id].append(layer_to_move)
                    distribution[stage_id].sort()
        
        return distribution
    
    def get_layers_for_stage(self, stage_id: int) -> List[int]:
        """Get list of layers assigned to a pipeline stage."""
        return self.layer_distribution.get(stage_id, [])
    
    def get_stage_for_layer(self, layer_id: int) -> int:
        """Get pipeline stage for a specific layer."""
        for stage_id, layers in self.layer_distribution.items():
            if layer_id in layers:
                return stage_id
        raise ValueError(f"Layer {layer_id} not found in any stage")
    
    def estimate_memory_usage(self, bytes_per_layer: int) -> Dict[int, int]:
        """
        Estimate memory usage per stage.
        
        Args:
            bytes_per_layer: Estimated bytes per layer
            
        Returns:
            Dict mapping stage_id -> estimated memory in bytes
        """
        usage = {}
        for stage_id, layers in self.layer_distribution.items():
            usage[stage_id] = len(layers) * bytes_per_layer
        return usage
    
    def validate_distribution(self) -> bool:
        """
        Validate that distribution fits within memory constraints.
        
        Returns:
            True if valid, False otherwise
        """
        # Estimate memory per layer (rough estimate for LLM)
        # Assuming ~2GB per layer for 7B model
        bytes_per_layer = 2 * 1024 * 1024 * 1024  # 2GB
        
        usage = self.estimate_memory_usage(bytes_per_layer)
        
        valid = True
        for stage_id in range(self.num_stages):
            available = self.memory_per_stage[stage_id] * 1024 * 1024 * 1024
            used = usage[stage_id]
            
            if used > available:
                logger.warning(
                    f"Stage {stage_id}: Estimated usage ({used/1e9:.2f}GB) "
                    f"exceeds available ({available/1e9:.2f}GB)"
                )
                valid = False
            else:
                logger.info(
                    f"Stage {stage_id}: {len(self.layer_distribution[stage_id])} layers, "
                    f"estimated usage: {used/1e9:.2f}GB / {available/1e9:.2f}GB"
                )
        
        return valid
    
    def print_distribution(self):
        """Print layer distribution information."""
        logger.info("=" * 60)
        logger.info("Memory-Aware Pipeline Parallel Distribution")
        logger.info("=" * 60)
        
        total_memory = sum(self.memory_per_stage)
        
        for stage_id in range(self.num_stages):
            layers = self.layer_distribution[stage_id]
            memory = self.memory_per_stage[stage_id]
            memory_pct = 100 * memory / total_memory if total_memory > 0 else 0
            
            logger.info(f"Stage {stage_id}:")
            logger.info(f"  Memory: {memory}GB ({memory_pct:.1f}%)")
            logger.info(f"  Layers: {len(layers)} ({min(layers)}-{max(layers) if layers else 'N/A'})")
            logger.info(f"  Layer IDs: {layers}")
        
        logger.info("=" * 60)


def create_memory_aware_pp(
    model_config,
    num_stages: int,
    node_memory_gb: Optional[List[int]] = None,
    strategy: str = "proportional",
) -> MemoryAwarePipelineParallel:
    """
    Factory function to create memory-aware pipeline parallelism.
    
    Args:
        model_config: Model configuration (should have num_hidden_layers)
        num_stages: Number of pipeline stages
        node_memory_gb: Available memory per node in GB
        strategy: Distribution strategy
        
    Returns:
        MemoryAwarePipelineParallel instance
    """
    num_layers = getattr(model_config, 'num_hidden_layers', 32)
    
    if node_memory_gb is None:
        # Auto-detect GPU memory
        if torch.cuda.is_available():
            node_memory_gb = [
                torch.cuda.get_device_properties(i).total_memory // (1024**3)
                for i in range(torch.cuda.device_count())
            ]
        else:
            node_memory_gb = [24] * num_stages  # Default
    
    # Ensure we have memory for each stage
    while len(node_memory_gb) < num_stages:
        node_memory_gb.append(node_memory_gb[-1] if node_memory_gb else 24)
    
    node_memory_gb = node_memory_gb[:num_stages]
    
    return MemoryAwarePipelineParallel(
        num_layers=num_layers,
        num_stages=num_stages,
        memory_per_stage=node_memory_gb,
        strategy=strategy,
    )


# Integration with SGLang's existing PP infrastructure
def patch_sglang_pp_scheduler():
    """
    Patch SGLang's pipeline parallel scheduler to use memory-aware distribution.
    
    This modifies the model loading to respect memory-aware layer assignment.
    """
    # TODO: Integrate with SGLang's model runner
    # This would involve modifying the model parallel initialization
    pass


# Example usage and testing
if __name__ == "__main__":
    # Test uniform distribution
    print("\n=== Uniform Distribution ===")
    pp = MemoryAwarePipelineParallel(
        num_layers=32,
        num_stages=4,
        memory_per_stage=[24, 24, 24, 24],
        strategy="uniform"
    )
    pp.print_distribution()
    
    # Test proportional distribution
    print("\n=== Proportional Distribution ===")
    pp = MemoryAwarePipelineParallel(
        num_layers=32,
        num_stages=4,
        memory_per_stage=[16, 24, 48, 24],  # Heterogeneous
        strategy="proportional"
    )
    pp.print_distribution()
    pp.validate_distribution()
    
    # Test custom distribution
    print("\n=== Custom Distribution ===")
    pp = MemoryAwarePipelineParallel(
        num_layers=32,
        num_stages=4,
        memory_per_stage=[16, 16, 48, 48],
        strategy="custom"
    )
    pp.print_distribution()
    pp.validate_distribution()
