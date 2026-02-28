# KVTuner 集成使用说明

## 概述

本目录包含 KVTuner 与 SGLang 的完整集成实现，支持：

- ✅ **Prefill/Decode 模式感知的量化策略**
- ✅ **残差缓存管理**（最近的 token 保持全精度）
- ✅ **逐层量化支持**（不同层使用不同精度）
- ✅ **内存统计和调试功能**

## 核心文件

### 1. `python/sglang/srt/mem_cache/kvtuner_kv_pool.py`

**主要类**: `KVTunerMHATokenToKVPool`

这是 KVTuner 集成的核心，扩展了 SGLang 的 `MHATokenToKVPool`，添加了：

- 量化缓存存储（与父类 buffer 并行）
- 残差缓存管理（最近 token 全精度）
- Prefill/Decode 模式感知量化
- 按需反量化

**关键方法**:

```python
# 初始化
__init__(
    size, page_size, dtype, head_num, head_dim, layer_num, device,
    enable_memory_saver,
    kvtuner_config: KVTunerQuantConfig,
    prefill_config: KVTunerModeConfig = None,
    decode_config: KVTunerModeConfig = None,
)

# 设置 KV buffer（支持模式感知）
set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale, forward_batch)

# 获取 KV buffer（自动反量化）
get_kv_buffer(layer_id)

# 获取内存统计
get_quantized_memory_stats()
```

### 2. `python/sglang/srt/layers/quantization/kvtuner_quant.py`

**主要类**:

- `KVTunerQuantConfig`: 量化配置
- `KVTunerVanillaQuantizer`: 量化器实现
- `KVTunerQuantizedTensor`: 量化张量容器
- `KVTunerQuantizationMethod`: 量化方法

### 3. `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

修改了 `token_to_kv_pool` 的初始化逻辑，当 `enable_kvtuner_quant=True` 时使用 `KVTunerMHATokenToKVPool`。

### 4. `python/sglang/srt/layers/attention/flashinfer_backend.py`

修改了 `set_kv_buffer` 调用，传递 `forward_batch` 参数以支持模式感知量化。

### 5. `python/sglang/srt/layers/attention/triton_backend.py`

同上，传递 `forward_batch` 参数。

## 使用方法

### 基本用法

启动 SGLang 服务器时添加 KVTuner 参数：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --enable-kvtuner-quant \
    --kvtuner-nbits-key 4 \
    --kvtuner-nbits-value 4 \
    --kvtuner-axis-key 0 \
    --kvtuner-axis-value 0 \
    --kvtuner-q-group-size 64 \
    --kvtuner-residual-length 128
```

### 启用逐层量化

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --enable-kvtuner-quant \
    --enable-kvtuner-layer-wise \
    --kvtuner-nbits-key 4 \
    --kvtuner-nbits-value 4 \
    --kvtuner-residual-length 128
```

### 使用配置文件

创建层量化配置文件 `layer_config.json`:

```json
{
    "layer_bits": {
        "0-5": 8,
        "6-20": 4,
        "21-31": 8
    },
    "default_bits": 4
}
```

启动时指定：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --enable-kvtuner-quant \
    --enable-kvtuner-layer-wise \
    --kvtuner-layer-config-file layer_config.json
```

## 配置参数说明

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-kvtuner-quant` | bool | False | 启用 KVTuner 量化 |
| `--kvtuner-nbits-key` | int | 4 | Key 量化位数 (2/4/8) |
| `--kvtuner-nbits-value` | int | 4 | Value 量化位数 (2/4/8) |
| `--kvtuner-asym` | bool | False | 使用非对称量化 |
| `--kvtuner-axis-key` | int | 0 | Key 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-axis-value` | int | 0 | Value 量化轴 (0=per-token, 1=per-channel) |
| `--kvtuner-q-group-size` | int | 64 | 量化组大小 |
| `--kvtuner-residual-length` | int | 128 | 残差长度（最近 token 保持全精度） |

### 逐层量化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-kvtuner-layer-wise` | bool | False | 启用逐层量化 |
| `--kvtuner-layer-bits` | str | None | 层量化位数配置（JSON 字符串） |
| `--kvtuner-layer-config-file` | str | None | 层量化配置文件路径 |

## Prefill/Decode 模式差异化配置

### 默认行为

- **Prefill 模式**: 使用基本配置（`--kvtuner-nbits-key`, `--kvtuner-nbits-value`）
- **Decode 模式**: 自动调整为更高精度（至少 4-bit），残差长度更短

### 自定义模式配置

在代码中直接指定：

```python
from sglang.srt.mem_cache.kvtuner_kv_pool import (
    KVTunerMHATokenToKVPool,
    KVTunerModeConfig,
)
from sglang.srt.layers.quantization.kvtuner_quant import KVTunerQuantConfig

# 基础配置
base_config = KVTunerQuantConfig(
    nbits_key=4,
    nbits_value=4,
    residual_length=128,
)

# Prefill 模式配置
prefill_config = KVTunerModeConfig(
    nbits_key=4,
    nbits_value=4,
    residual_length=256,  # 更长的残差
    enable_quantization=True,
)

# Decode 模式配置
decode_config = KVTunerModeConfig(
    nbits_key=8,  # 更高精度，减少反量化开销
    nbits_value=8,
    residual_length=32,  # 更短的残差
    enable_quantization=True,
)

# 创建量化池
pool = KVTunerMHATokenToKVPool(
    size=...,
    kvtuner_config=base_config,
    prefill_config=prefill_config,
    decode_config=decode_config,
)
```

## 内存统计

启用 KVTuner 后，可以通过以下方法获取内存使用统计：

```python
# 获取量化缓存统计
stats = token_to_kv_pool.get_quantized_memory_stats()
print(stats)

# 输出示例:
# {
#     "enabled": True,
#     "quantized_slots": 1024,
#     "quantized_data_mb": 128.5,
#     "quantized_scale_mb": 16.2,
#     "quantized_total_mb": 144.7,
#     "full_precision_mb": 512.0,
#     "savings_mb": 367.3,
#     "compression_ratio": 3.54,
#     "stats": {
#         "total_quantize_ops": 10240,
#         "total_dequantize_ops": 5120,
#         "prefill_quantize_count": 8192,
#         "decode_quantize_count": 2048,
#     }
# }
```

## 性能优化建议

### 1. 量化位数选择

| 场景 | 推荐配置 | 压缩比 | 精度损失 |
|------|----------|--------|----------|
| 高吞吐推理 | 4-bit K/V | ~4x | <1% perplexity |
| 低延迟推理 | 8-bit K/V | ~2x | <0.1% perplexity |
| 极限内存节省 | 2-bit K/V | ~8x | 1-3% perplexity |

### 2. 残差长度调优

| 场景 | 推荐残差长度 | 说明 |
|------|--------------|------|
| 长上下文 Prefill | 256-512 | 保持更多 token 全精度 |
| 短上下文 Decode | 16-32 | 减少内存占用 |
| 平衡模式 | 128 | 默认值 |

### 3. 逐层量化策略

```json
{
    "layer_bits": {
        "0-2": 8,      // Embedding 层用高精度
        "3-28": 4,     // 中间层用低精度
        "29-31": 8     // Output 层用高精度
    }
}
```

## 调试功能

### 启用详细日志

```bash
python -m sglang.launch_server \
    ... \
    --log-level debug
```

### 检查量化状态

```python
# 检查是否启用了 KVTuner
print(f"KVTuner enabled: {pool.enable_kvtuner}")

# 检查量化槽数量
print(f"Quantized slots: {sum(len(s) for s in pool.quantized_slots)}")

# 检查内存统计
stats = pool.get_memory_usage()
print(f"Memory savings: {stats['savings_mb']:.2f} MB")
```

## 已知限制

1. **仅支持 MHA 模型**: MLA 模型（如 DeepSeek-v2/v3）的 KVTuner 支持尚未实现
2. **FP8 量化不兼容**: KVTuner 量化与 FP8 KV cache 不能同时启用
3. **CPU Offloading**: 量化缓存的 CPU offloading 尚未优化

## 故障排除

### 问题：量化后精度下降明显

**解决方案**:
1. 增加量化位数（4-bit → 8-bit）
2. 增加残差长度（128 → 256）
3. 对敏感层使用更高精度（逐层量化）

### 问题：Decode 延迟增加

**解决方案**:
1. 使用更高的 Decode 模式量化精度（减少反量化开销）
2. 减少残差长度
3. 禁用逐层量化（简化路径）

### 问题：内存节省不如预期

**解决方案**:
1. 检查是否启用了量化（`enable_kvtuner=True`）
2. 检查量化槽数量（`get_quantized_memory_stats()`）
3. 确保残差长度设置合理（过大会减少量化比例）

## 参考资料

- [KVTuner 原论文](https://arxiv.org/abs/xxxx.xxxxx)
- [SGLang 文档](https://docs.sglang.ai)
- [集成分析报告](../sglang-kvtuner-analysis.md)
