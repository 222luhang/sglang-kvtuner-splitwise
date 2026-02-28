#!/usr/bin/env python3
"""
KVTuner KV Cache Compressed Transfer

利用 KVTuner 量化技术实现 KV Cache 压缩传输，显著降低 P/D 分离架构中的
网络传输延迟和带宽使用。

核心功能:
- KVTuner 4-bit/8-bit 量化压缩
- 增量传输（仅传输变化的 KV）
- 异步传输流水线
- 精度/压缩比权衡配置

Usage:
    from kv_transfer_compressed import CompressedKVTransfer
    
    transfer = CompressedKVTransfer(
        nbits=4,  # 4-bit 压缩
        backend='nixl'
    )
    
    # 压缩传输
    await transfer.transfer_kv(
        kv_cache,
        src_node="http://10.60.6.75:30000",
        dst_node="http://10.60.19.152:30001"
    )
"""

import asyncio
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np

# 导入 KVTuner 量化器
try:
    from sglang.srt.layers.quantization.kvtuner_quant import (
        KVTunerQuantConfig,
        KVTunerVanillaQuantizer,
        KVTunerQuantizedTensor,
    )
    KVTUNER_AVAILABLE = True
except ImportError:
    KVTUNER_AVAILABLE = False
    print("Warning: KVTuner not available, falling back to basic quantization")

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """压缩级别"""
    LOSSLESS = "lossless"      # 无损（BF16）
    HIGH_QUALITY = "high"      # 高质量（8-bit）
    BALANCED = "balanced"      # 平衡（4-bit）
    HIGH_COMPRESSION = "max"   # 高压缩（2-bit）


@dataclass
class TransferStats:
    """传输统计信息"""
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0
    transfer_time_ms: float = 0.0
    bandwidth_gbps: float = 0.0
    quantization_time_ms: float = 0.0
    dequantization_time_ms: float = 0.0
    total_time_ms: float = 0.0
    success: bool = False
    error_message: str = ""
    
    @property
    def compression_savings(self) -> float:
        """节省的带宽百分比"""
        if self.original_size_mb == 0:
            return 0.0
        return (1 - self.compressed_size_mb / self.original_size_mb) * 100


class CompressedKVTransfer:
    """
    KVTuner KV Cache 压缩传输器
    
    特性:
    - 支持 2/4/8-bit 量化压缩
    - 自动选择最优压缩级别
    - 异步流水线传输
    - 详细的统计和监控
    """
    
    def __init__(
        self,
        nbits: int = 4,
        asym: bool = False,
        q_group_size: int = 64,
        backend: str = 'nixl',
        device: str = 'cuda',
        enable_async: bool = True,
        max_concurrent_transfers: int = 4,
    ):
        """
        初始化压缩传输器
        
        Args:
            nbits: 量化位数 (2/4/8)
            asym: 非对称量化
            q_group_size: 量化组大小
            backend: 传输后端 ('nixl', 'nccl', 'tcp')
            device: 计算设备
            enable_async: 启用异步传输
            max_concurrent_transfers: 最大并发传输数
        """
        self.nbits = nbits
        self.asym = asym
        self.q_group_size = q_group_size
        self.backend = backend
        self.device = device
        self.enable_async = enable_async
        self.max_concurrent_transfers = max_concurrent_transfers
        
        # 初始化量化器
        if KVTUNER_AVAILABLE:
            self.quantizer = KVTunerVanillaQuantizer(
                nbits=nbits,
                asym=asym,
                compute_dtype=torch.bfloat16
            )
            logger.info(f"KVTuner quantizer initialized: {nbits}-bit, group_size={q_group_size}")
        else:
            self.quantizer = None
            logger.warning("KVTuner not available, using basic quantization")
        
        # 传输信号量（控制并发）
        self._semaphore = asyncio.Semaphore(max_concurrent_transfers)
        
        # 统计信息
        self.stats_history: List[TransferStats] = []
    
    async def transfer_kv(
        self,
        kv_cache: torch.Tensor,
        src_node: str,
        dst_node: str,
        compression_level: Optional[CompressionLevel] = None,
        timeout: float = 60.0,
    ) -> Tuple[torch.Tensor, TransferStats]:
        """
        压缩传输 KV Cache
        
        Args:
            kv_cache: KV Cache 张量 [layers, batch, heads, seq_len, head_dim]
            src_node: 源节点 URL
            dst_node: 目标节点 URL
            compression_level: 压缩级别（None 则自动选择）
            timeout: 超时时间（秒）
        
        Returns:
            (dequantized_kv, stats): 反量化后的 KV Cache 和统计信息
        """
        stats = TransferStats()
        start_time = time.time()
        
        try:
            async with self._semaphore:
                # 记录原始大小
                stats.original_size_mb = kv_cache.element_size() * kv_cache.numel() / (1024 * 1024)
                
                # 选择压缩级别
                if compression_level is None:
                    compression_level = self._auto_select_compression(
                        kv_cache, src_node, dst_node
                    )
                
                logger.info(
                    f"Starting KV transfer: {src_node} → {dst_node}, "
                    f"size={stats.original_size_mb:.2f}MB, compression={compression_level.value}"
                )
                
                # 量化
                quant_start = time.time()
                quantized = await self._quantize(kv_cache, compression_level)
                stats.quantization_time_ms = (time.time() - quant_start) * 1000
                
                # 记录压缩后大小
                if isinstance(quantized, KVTunerQuantizedTensor):
                    stats.compressed_size_mb = quantized.nbytes / (1024 * 1024)
                else:
                    stats.compressed_size_mb = stats.original_size_mb
                
                stats.compression_ratio = stats.original_size_mb / max(stats.compressed_size_mb, 0.001)
                
                # 传输
                transfer_start = time.time()
                await self._transfer_tensor(quantized, src_node, dst_node, timeout)
                stats.transfer_time_ms = (time.time() - transfer_start) * 1000
                
                # 计算带宽
                if stats.transfer_time_ms > 0:
                    stats.bandwidth_gbps = (stats.compressed_size_mb * 8) / (stats.transfer_time_ms / 1000) / 1000
                
                # 反量化
                dequant_start = time.time()
                dequantized = await self._dequantize(quantized, compression_level)
                stats.dequantization_time_ms = (time.time() - dequant_start) * 1000
                
                stats.total_time_ms = (time.time() - start_time) * 1000
                stats.success = True
                
                logger.info(
                    f"Transfer complete: "
                    f"compression={stats.compression_ratio:.2f}x, "
                    f"time={stats.total_time_ms:.2f}ms, "
                    f"bandwidth={stats.bandwidth_gbps:.2f}Gbps"
                )
                
                # 保存统计
                self.stats_history.append(stats)
                
                return dequantized, stats
        
        except Exception as e:
            stats.total_time_ms = (time.time() - start_time) * 1000
            stats.success = False
            stats.error_message = str(e)
            logger.error(f"Transfer failed: {e}")
            self.stats_history.append(stats)
            raise
    
    async def _quantize(
        self,
        kv_cache: torch.Tensor,
        compression_level: CompressionLevel
    ) -> Any:
        """量化 KV Cache"""
        if compression_level == CompressionLevel.LOSSLESS:
            return kv_cache
        
        if KVTUNER_AVAILABLE and self.quantizer:
            # 使用 KVTuner 量化
            nbits = {
                CompressionLevel.HIGH_QUALITY: 8,
                CompressionLevel.BALANCED: 4,
                CompressionLevel.HIGH_COMPRESSION: 2,
            }.get(compression_level, 4)
            
            # 临时修改量化器配置
            original_nbits = self.quantizer.nbits
            self.quantizer.nbits = nbits
            
            # 逐层量化（避免显存溢出）
            if kv_cache.dim() == 5:  # [layers, batch, heads, seq_len, head_dim]
                quantized_layers = []
                for i in range(kv_cache.shape[0]):
                    layer_kv = kv_cache[i]
                    q_layer = self.quantizer.quantize(
                        layer_kv.contiguous(),
                        q_group_size=self.q_group_size,
                        axis=0
                    )
                    quantized_layers.append(q_layer)
                
                self.quantizer.nbits = original_nbits
                return quantized_layers
            else:
                self.quantizer.nbits = original_nbits
                return self.quantizer.quantize(
                    kv_cache.contiguous(),
                    q_group_size=self.q_group_size,
                    axis=0
                )
        else:
            # 基本量化（fallback）
            if compression_level == CompressionLevel.HIGH_COMPRESSION:
                return kv_cache.to(torch.int8)
            else:
                return kv_cache.to(torch.int16)
    
    async def _dequantize(
        self,
        quantized: Any,
        compression_level: CompressionLevel
    ) -> torch.Tensor:
        """反量化 KV Cache"""
        if compression_level == CompressionLevel.LOSSLESS:
            return quantized
        
        if KVTUNER_AVAILABLE and isinstance(quantized, KVTunerQuantizedTensor):
            return quantized.dequantize()
        elif isinstance(quantized, list):
            # 逐层反量化
            dequantized_layers = []
            for q_layer in quantized:
                if isinstance(q_layer, KVTunerQuantizedTensor):
                    dequantized_layers.append(q_layer.dequantize())
                else:
                    dequantized_layers.append(q_layer.to(torch.bfloat16))
            return torch.stack(dequantized_layers)
        else:
            # 基本反量化
            return quantized.to(torch.bfloat16)
    
    async def _transfer_tensor(
        self,
        tensor: Any,
        src_node: str,
        dst_node: str,
        timeout: float
    ):
        """底层传输实现"""
        if self.backend == 'nixl':
            await self._transfer_nixl(tensor, src_node, dst_node, timeout)
        elif self.backend == 'nccl':
            await self._transfer_nccl(tensor, src_node, dst_node, timeout)
        else:
            await self._transfer_tcp(tensor, src_node, dst_node, timeout)
    
    async def _transfer_nixl(
        self,
        tensor: Any,
        src_node: str,
        dst_node: str,
        timeout: float
    ):
        """使用 NIXL 后端传输"""
        # TODO: 集成 NIXL UCX 后端
        # 当前使用模拟延迟
        await asyncio.sleep(0.01)  # 10ms 模拟
        logger.debug(f"NIXL transfer: {src_node} → {dst_node}")
    
    async def _transfer_nccl(
        self,
        tensor: Any,
        src_node: str,
        dst_node: str,
        timeout: float
    ):
        """使用 NCCL GPUDirect 传输"""
        # TODO: 集成 NCCL
        await asyncio.sleep(0.01)
        logger.debug(f"NCCL transfer: {src_node} → {dst_node}")
    
    async def _transfer_tcp(
        self,
        tensor: Any,
        src_node: str,
        dst_node: str,
        timeout: float
    ):
        """使用 TCP 传输（fallback）"""
        # TODO: 实现 TCP 传输
        await asyncio.sleep(0.01)
        logger.debug(f"TCP transfer: {src_node} → {dst_node}")
    
    def _auto_select_compression(
        self,
        kv_cache: torch.Tensor,
        src_node: str,
        dst_node: str
    ) -> CompressionLevel:
        """自动选择压缩级别"""
        # 基于网络距离和 KV 大小选择
        size_mb = kv_cache.element_size() * kv_cache.numel() / (1024 * 1024)
        
        if size_mb > 500:  # 大 KV，优先压缩
            return CompressionLevel.BALANCED
        elif size_mb > 100:  # 中等 KV，平衡
            return CompressionLevel.BALANCED
        else:  # 小 KV，高质量
            return CompressionLevel.HIGH_QUALITY
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if not self.stats_history:
            return {}
        
        successful = [s for s in self.stats_history if s.success]
        if not successful:
            return {"total_transfers": 0, "success_rate": 0.0}
        
        return {
            "total_transfers": len(self.stats_history),
            "success_rate": len(successful) / len(self.stats_history) * 100,
            "avg_compression_ratio": np.mean([s.compression_ratio for s in successful]),
            "avg_transfer_time_ms": np.mean([s.transfer_time_ms for s in successful]),
            "avg_total_time_ms": np.mean([s.total_time_ms for s in successful]),
            "avg_bandwidth_gbps": np.mean([s.bandwidth_gbps for s in successful]),
            "total_data_transferred_mb": sum([s.compressed_size_mb for s in successful]),
            "total_time_saved_ms": sum([
                s.original_size_mb / max(s.bandwidth_gbps * 125, 0.001) - s.total_time_ms
                for s in successful
            ]),
        }


class IncrementalKVTransfer(CompressedKVTransfer):
    """
    增量 KV Cache 传输器
    
    仅传输变化的 KV 部分，进一步减少带宽使用。
    适用于长序列的连续生成场景。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_history: Dict[str, torch.Tensor] = {}
    
    async def transfer_kv_incremental(
        self,
        kv_cache: torch.Tensor,
        src_node: str,
        dst_node: str,
        request_id: str,
    ) -> Tuple[torch.Tensor, TransferStats]:
        """
        增量传输 KV Cache
        
        Args:
            kv_cache: 当前 KV Cache
            src_node: 源节点
            dst_node: 目标节点
            request_id: 请求 ID（用于跟踪历史）
        
        Returns:
            (merged_kv, stats): 合并后的 KV Cache 和统计
        """
        # 获取历史 KV
        prev_kv = self._cache_history.get(request_id)
        
        if prev_kv is None:
            # 首次传输，完整传输
            result, stats = await self.transfer_kv(kv_cache, src_node, dst_node)
            self._cache_history[request_id] = kv_cache.cpu()
            return result, stats
        
        # 计算增量
        if kv_cache.shape == prev_kv.shape:
            delta = kv_cache - prev_kv.to(kv_cache.device)
        else:
            # 形状不同，传输完整 KV
            delta = kv_cache
        
        # 传输增量
        result, stats = await self.transfer_kv(delta, src_node, dst_node)
        
        # 合并
        if prev_kv.shape == kv_cache.shape:
            merged = prev_kv.to(kv_cache.device) + result
        else:
            merged = result
        
        # 更新历史
        self._cache_history[request_id] = kv_cache.cpu()
        
        return merged, stats
    
    def clear_history(self, request_id: Optional[str] = None):
        """清除历史缓存"""
        if request_id:
            self._cache_history.pop(request_id, None)
        else:
            self._cache_history.clear()


# 便捷函数
async def transfer_kv_compressed(
    kv_cache: torch.Tensor,
    src: str,
    dst: str,
    nbits: int = 4,
    backend: str = 'nixl',
) -> Tuple[torch.Tensor, TransferStats]:
    """
    便捷函数：压缩传输 KV Cache
    
    Usage:
        kv_dequant, stats = await transfer_kv_compressed(
            kv_cache,
            "http://10.60.6.75:30000",
            "http://10.60.19.152:30001",
            nbits=4
        )
    """
    transfer = CompressedKVTransfer(nbits=nbits, backend=backend)
    return await transfer.transfer_kv(kv_cache, src, dst)


if __name__ == "__main__":
    # 测试示例
    async def test():
        print("Testing KVTuner Compressed KV Transfer...")
        
        # 创建测试 KV Cache
        test_kv = torch.randn(32, 1, 32, 1000, 128, dtype=torch.bfloat16, device='cuda')
        
        # 创建传输器
        transfer = CompressedKVTransfer(nbits=4, backend='nixl')
        
        # 传输
        result, stats = await transfer.transfer_kv(
            test_kv,
            "http://localhost:30000",
            "http://localhost:30001"
        )
        
        # 打印统计
        print(f"\nTransfer Statistics:")
        print(f"  Original Size: {stats.original_size_mb:.2f} MB")
        print(f"  Compressed Size: {stats.compressed_size_mb:.2f} MB")
        print(f"  Compression Ratio: {stats.compression_ratio:.2f}x")
        print(f"  Transfer Time: {stats.transfer_time_ms:.2f} ms")
        print(f"  Total Time: {stats.total_time_ms:.2f} ms")
        print(f"  Bandwidth: {stats.bandwidth_gbps:.2f} Gbps")
        print(f"  Success: {stats.success}")
        
        # 验证精度
        mse = ((result - test_kv) ** 2).mean().item()
        print(f"\nMSE (精度损失): {mse:.6f}")
    
    # asyncio.run(test())
    print("Run with: python3 kv_transfer_compressed.py")
