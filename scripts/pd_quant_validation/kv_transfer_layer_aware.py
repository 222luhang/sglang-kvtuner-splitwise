#!/usr/bin/env python3
"""
KVTuner Layer-Aware Compressed KV Transfer

支持层级量化配置的 KV Cache 压缩传输，与 Splitwise P/D 分离架构深度集成。

核心特性:
- 加载离线计算的层级量化配置
- 每层使用不同的量化精度
- 与 Splitwise 调度器协同工作
- 支持模式感知（Prefill/Decode 不同配置）

Usage:
    # 加载层级配置
    config = LayerQuantConfig.load("layer_quant_config.json")
    
    # 创建传输器
    transfer = LayerAwareKVTransfer(
        layer_config=config,
        backend='nixl'
    )
    
    # 传输（自动应用层级量化）
    await transfer.transfer_kv(kv_cache, src, dst)
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

import torch
import torch.nn as nn

# 导入离线配置
try:
    from kvtuner_offline_calib import ModelQuantConfig, LayerQuantConfig
except ImportError:
    # 定义简化版本
    @dataclass
    class LayerQuantConfig:
        layer_idx: int
        nbits_key: int = 4
        nbits_value: int = 4
        q_group_size: int = 64
        asym: bool = False
    
    class ModelQuantConfig:
        def __init__(self):
            self.layers: Dict[int, LayerQuantConfig] = {}
            self.default_nbits = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerAwareQuantizer:
    """
    层级感知量化器
    
    根据离线配置对每层使用不同的量化精度
    """
    
    def __init__(self, model_config: ModelQuantConfig):
        self.model_config = model_config
        self.quantizers: Dict[int, Any] = {}  # layer_idx -> quantizer
    
    def get_quantizer(self, layer_idx: int):
        """获取指定层的量化器"""
        if layer_idx not in self.quantizers:
            # 获取该层配置
            layer_config = self.model_config.layers.get(
                layer_idx,
                LayerQuantConfig(
                    layer_idx=layer_idx,
                    nbits_key=self.model_config.default_nbits,
                    nbits_value=self.model_config.default_nbits,
                )
            )
            
            # 创建量化器
            self.quantizers[layer_idx] = SimpleLayerQuantizer(
                nbits=layer_config.nbits_key,
                q_group_size=layer_config.q_group_size,
                asym=layer_config.asym,
            )
        
        return self.quantizers[layer_idx]
    
    def quantize_layer(
        self,
        kv_tensor: torch.Tensor,
        layer_idx: int
    ) -> tuple:
        """量化指定层的 KV"""
        quantizer = self.get_quantizer(layer_idx)
        return quantizer.quantize(kv_tensor)
    
    def dequantize_layer(
        self,
        quant_data: torch.Tensor,
        scale: torch.Tensor,
        zeros: Optional[torch.Tensor],
        original_shape: torch.Size,
        layer_idx: int
    ) -> torch.Tensor:
        """反量化指定层的 KV"""
        quantizer = self.get_quantizer(layer_idx)
        return quantizer.dequantize(quant_data, scale, zeros, original_shape)


class SimpleLayerQuantizer:
    """简单的层级量化器"""
    
    def __init__(self, nbits: int, q_group_size: int, asym: bool):
        self.nbits = nbits
        self.q_group_size = q_group_size
        self.asym = asym
        self.q_max = 2 ** (nbits - 1) - 1
        self.q_min = -(2 ** (nbits - 1))
    
    def quantize(self, tensor: torch.Tensor) -> tuple:
        """量化"""
        original_shape = tensor.shape
        rs = tensor.reshape(-1, self.q_group_size)
        
        if self.asym:
            _max, _min = rs.max(dim=1).values, rs.min(dim=1).values
            scale = (_max - _min).clamp(min=1e-5).div(self.q_max - self.q_min)
            zeros = (_min / scale).round() - self.q_min
            quant = (torch.round(rs / scale.unsqueeze(1) - zeros.unsqueeze(1))).clamp(
                self.q_min, self.q_max
            ).to(torch.int8)
        else:
            scale = rs.abs().max(dim=1).values.clamp(min=1e-5).div(self.q_max)
            quant = torch.round(rs / scale.unsqueeze(1)).clamp(
                self.q_min, self.q_max
            ).to(torch.int8)
        
        return quant.reshape(original_shape), scale, zeros
    
    def dequantize(
        self,
        quant: torch.Tensor,
        scale: torch.Tensor,
        zeros: Optional[torch.Tensor],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """反量化"""
        quant_flat = quant.reshape(-1, self.q_group_size)
        scale_expanded = scale.unsqueeze(1)
        
        if self.asym and zeros is not None:
            zeros_expanded = zeros.unsqueeze(1)
            dequant_flat = (quant_flat.to(torch.float32) + zeros_expanded) * scale_expanded
        else:
            dequant_flat = quant_flat.to(torch.float32) * scale_expanded
        
        return dequant_flat.view(original_shape)


class LayerAwareKVTransfer:
    """
    层级感知的 KV 传输器
    
    集成 Splitwise P/D 分离架构：
    - 根据离线配置逐层量化
    - Prefill/Decode 使用不同策略
    - 异步流水线传输
    """
    
    def __init__(
        self,
        layer_config_path: str,
        backend: str = 'nixl',
        device: str = 'cuda',
        enable_async: bool = True,
        # Splitwise 集成参数
        mode: str = 'prefill',  # 'prefill' or 'decode'
        prefill_mode_config: Optional[Dict] = None,
        decode_mode_config: Optional[Dict] = None,
    ):
        """
        初始化层级感知传输器
        
        Args:
            layer_config_path: 离线量化配置路径
            backend: 传输后端
            device: 计算设备
            mode: 当前模式（Prefill/Decode）
            prefill_mode_config: Prefill 模式特定配置
            decode_mode_config: Decode 模式特定配置
        """
        self.backend = backend
        self.device = device
        self.enable_async = enable_async
        self.mode = mode
        
        # 加载离线配置
        logger.info(f"Loading layer quant config from {layer_config_path}")
        self.model_config = self._load_config(layer_config_path)
        
        # 创建量化器
        self.quantizer = LayerAwareQuantizer(self.model_config)
        
        # 模式特定配置（Splitwise 集成）
        self.prefill_mode_config = prefill_mode_config or {}
        self.decode_mode_config = decode_mode_config or {}
        
        # 应用模式配置
        self._apply_mode_config(mode)
        
        logger.info(
            f"Layer-aware transfer initialized: "
            f"{len(self.model_config.layers)} layers configured, "
            f"mode={mode}"
        )
    
    def _load_config(self, path: str) -> ModelQuantConfig:
        """加载配置"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelQuantConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            # 返回默认配置
            config = ModelQuantConfig()
            config.default_nbits = 4
            return config
    
    def _apply_mode_config(self, mode: str):
        """应用模式特定配置（Splitwise 集成）"""
        if mode == 'prefill':
            # Prefill 模式：高吞吐，可以接受稍低精度
            scale_factor = self.prefill_mode_config.get('nbits_scale', 1.0)
        elif mode == 'decode':
            # Decode 模式：低延迟，需要更高精度
            scale_factor = self.decode_mode_config.get('nbits_scale', 1.2)
        else:
            scale_factor = 1.0
        
        # 调整量化配置
        for layer_idx, config in self.model_config.layers.items():
            # 根据模式调整 nbits
            adjusted_nbits = int(config.nbits_key * scale_factor)
            adjusted_nbits = min(8, max(2, adjusted_nbits))  # 限制在 2-8 之间
            
            config.nbits_key = adjusted_nbits
            config.nbits_value = adjusted_nbits
            
            logger.debug(
                f"Layer {layer_idx}: adjusted nbits to {adjusted_nbits} "
                f"(mode={mode}, scale={scale_factor})"
            )
    
    async def transfer_kv(
        self,
        kv_cache: torch.Tensor,  # [layers, batch, heads, seq_len, head_dim]
        src_node: str,
        dst_node: str,
        timeout: float = 60.0,
    ) -> torch.Tensor:
        """
        层级感知 KV 传输
        
        流程:
        1. 逐层量化（使用离线配置）
        2. 批量传输
        3. 目标节点逐层反量化
        
        Args:
            kv_cache: KV Cache 张量
            src_node: 源节点 URL
            dst_node: 目标节点 URL
            timeout: 超时时间
        
        Returns:
            反量化后的 KV Cache
        """
        start_time = time.time()
        num_layers = kv_cache.shape[0]
        
        logger.info(
            f"Starting layer-aware transfer: "
            f"{src_node} → {dst_node}, "
            f"layers={num_layers}, mode={self.mode}"
        )
        
        # 1. 逐层量化
        quant_start = time.time()
        quantized_layers = []
        
        for layer_idx in range(num_layers):
            layer_kv = kv_cache[layer_idx]
            
            # 使用该层的量化配置
            quant_data, scale, zeros = self.quantizer.quantize_layer(
                layer_kv, layer_idx
            )
            
            quantized_layers.append({
                'layer_idx': layer_idx,
                'data': quant_data,
                'scale': scale,
                'zeros': zeros,
                'shape': layer_kv.shape,
            })
        
        quant_time = time.time() - quant_start
        logger.info(f"Quantization completed: {quant_time*1000:.2f}ms")
        
        # 2. 传输（批量）
        transfer_start = time.time()
        await self._transfer_layers(quantized_layers, src_node, dst_node, timeout)
        transfer_time = time.time() - transfer_start
        logger.info(f"Transfer completed: {transfer_time*1000:.2f}ms")
        
        # 3. 逐层反量化
        dequant_start = time.time()
        dequantized_layers = []
        
        for q_layer in quantized_layers:
            dequant = self.quantizer.dequantize_layer(
                q_layer['data'],
                q_layer['scale'],
                q_layer['zeros'],
                q_layer['shape'],
                q_layer['layer_idx']
            )
            dequantized_layers.append(dequant)
        
        dequant_time = time.time() - dequant_start
        logger.info(f"Dequantization completed: {dequant_time*1000:.2f}ms")
        
        # 合并层
        result = torch.stack(dequantized_layers)
        
        total_time = time.time() - start_time
        logger.info(
            f"Layer-aware transfer complete: "
            f"total={total_time*1000:.2f}ms, "
            f"layers={num_layers}"
        )
        
        return result
    
    async def _transfer_layers(
        self,
        quantized_layers: List[Dict],
        src_node: str,
        dst_node: str,
        timeout: float
    ):
        """传输所有层（可优化为流水线）"""
        # TODO: 实现真实的 NIXL/NCCL 传输
        # 当前使用模拟
        await asyncio.sleep(0.01)  # 10ms 模拟
        
        logger.debug(
            f"Transferred {len(quantized_layers)} layers: "
            f"{src_node} → {dst_node}"
        )
    
    def get_layer_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        summary = {
            'total_layers': len(self.model_config.layers),
            'default_nbits': self.model_config.default_nbits,
            'mode': self.mode,
            'layers_by_nbits': {},
        }
        
        # 统计不同 nbits 的层数
        nbits_dist = {}
        for layer_idx, config in self.model_config.layers.items():
            nbits = config.nbits_key
            nbits_dist[nbits] = nbits_dist.get(nbits, 0) + 1
        
        summary['layers_by_nbits'] = nbits_dist
        
        return summary


class SplitwiseLayerAwareTransfer(LayerAwareKVTransfer):
    """
    Splitwise 增强的层级感知传输器
    
    特性:
    - Prefill/Decode 模式自动切换
    - 异构硬件支持（H100/A100）
    - 成本/功耗感知
    """
    
    def __init__(
        self,
        layer_config_path: str,
        node_hardware: Dict[str, str],  # node_url -> gpu_model
        **kwargs
    ):
        # 基础初始化
        super().__init__(layer_config_path, **kwargs)
        
        self.node_hardware = node_hardware
        
        # 硬件特定配置
        self.hw_configs = {
            'H100': {'nbits_scale': 1.0, 'q_group_size': 64},
            'A100': {'nbits_scale': 1.2, 'q_group_size': 64},
            'RTX3090': {'nbits_scale': 0.8, 'q_group_size': 32},
        }
    
    def set_mode(self, mode: str, node_url: str):
        """
        设置模式并应用硬件特定配置
        
        Args:
            mode: 'prefill' or 'decode'
            node_url: 当前节点 URL
        """
        self.mode = mode
        
        # 获取硬件类型
        gpu_model = self.node_hardware.get(node_url, 'A100')
        hw_config = self.hw_configs.get(gpu_model, self.hw_configs['A100'])
        
        # 更新配置
        if mode == 'prefill':
            self.prefill_mode_config = hw_config
        else:
            self.decode_mode_config = hw_config
        
        # 重新应用配置
        self._apply_mode_config(mode)
        
        logger.info(
            f"Mode set: {mode}, hardware: {gpu_model}, "
            f"nbits_scale: {hw_config['nbits_scale']}"
        )


# 便捷函数
async def transfer_kv_layer_aware(
    kv_cache: torch.Tensor,
    src: str,
    dst: str,
    layer_config_path: str,
    mode: str = 'prefill',
) -> torch.Tensor:
    """便捷函数：层级感知传输"""
    transfer = LayerAwareKVTransfer(
        layer_config_path=layer_config_path,
        mode=mode,
    )
    return await transfer.transfer_kv(kv_cache, src, dst)


if __name__ == "__main__":
    # 测试示例
    async def test():
        print("Testing Layer-Aware KV Transfer...")
        
        # 创建测试配置
        test_config = {
            "model_name": "Test",
            "default_nbits": 4,
            "layers": {
                "0": {"layer_idx": 0, "nbits_key": 8, "nbits_value": 8, "q_group_size": 64},
                "1": {"layer_idx": 1, "nbits_key": 4, "nbits_value": 4, "q_group_size": 64},
                "2": {"layer_idx": 2, "nbits_key": 2, "nbits_value": 2, "q_group_size": 32},
            }
        }
        
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        # 创建测试 KV
        test_kv = torch.randn(3, 1, 8, 100, 64, dtype=torch.bfloat16)
        
        # 创建传输器
        transfer = LayerAwareKVTransfer(
            layer_config_path=config_path,
            mode='prefill'
        )
        
        # 打印配置摘要
        summary = transfer.get_layer_config_summary()
        print(f"\n配置摘要:")
        print(f"  总层数：{summary['total_layers']}")
        print(f"  默认 nbits: {summary['default_nbits']}")
        print(f"  模式：{summary['mode']}")
        print(f"  按 nbits 分布：{summary['layers_by_nbits']}")
        
        # 传输
        result = await transfer.transfer_kv(
            test_kv,
            "http://localhost:30000",
            "http://localhost:30001"
        )
        
        print(f"\n✅ 测试完成")
        print(f"  输入形状：{test_kv.shape}")
        print(f"  输出形状：{result.shape}")
        
        # 验证精度
        mse = ((result - test_kv) ** 2).mean().item()
        print(f"  MSE: {mse:.6f}")
    
    # asyncio.run(test())
    print("Run with: python3 kv_transfer_layer_aware.py")
