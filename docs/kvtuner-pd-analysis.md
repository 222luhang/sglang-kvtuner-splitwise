# KVTuner P/D分离架构分析文档

> 整理时间：2026-03-28
> 基于分支：kvtuner-splitwise-realign
> 测试环境：10.60.23.70 (Prefill) + 10.60.30.66 (Decode), RTX 4090, NIXL/TCP

---

## 一、整体推理路径

```
客户端
  │
  ▼
Router (sglang-router, port 8000)
  │  分发请求到Prefill节点
  ▼
Prefill节点 (port 30000)
  │  1. Tokenize输入
  │  2. 模型Forward (EXTEND模式)
  │  3. KVTuner量化KV Cache (逐层不同精度)
  │  4. 通过NIXL/UCX传输量化后的KV Cache
  ▼
Decode节点 (port 30001)
  │  5. 接收量化KV Cache
  │  6. KVTuner反量化 (逐层不同精度)
  │  7. 模型Forward (DECODE模式)
  │  8. 逐token生成
  │  9. 通过NIXL传输生成结果
  ▼
Router
  │
  ▼
客户端
```

## 二、KVTuner量化工作原理

### 2.1 量化阶段（Prefill节点）

发生在 `KVTunerMHATokenToKVPool.set_kv_buffer()` 中，每层独立处理：

```
输入: cache_k [num_tokens, head_num, head_dim], cache_v [num_tokens, head_num, v_head_dim]
  │
  ├─ 获取当前层的量化配置: _get_quant_method_for_layer(layer_id, forward_mode)
  │    优先级: per-layer+mode > per-layer > mode > global base
  │
  ├─ 逐token判断是否需要量化: _should_quantize_token(buffer_idx, slot_idx, forward_mode, token_position, layer_id)
  │    - 计算token距序列末尾的距离: distance_from_end = seq_length - token_position
  │    - 如果 distance_from_end < residual_length → 保留全精度(残差)
  │    - 否则 → 执行量化
  │
  ├─ 量化执行 (KVTunerVanillaQuantizer):
  │    1. 确定量化参数: nbits, axis, group_size (均由layer配置决定)
  │    2. 对称/非对称量化
  │    3. 打包为KVTunerQuantizedTensor存储
  │
  └─ 存储:
       - 量化数据 → Python dict: self.quantized_k_cache[buffer_idx][slot_idx]
       - 残差数据 → 底层buffer (MHATokenToKVPool, 全精度)
       - 位置信息 → Python dict: self.slot_positions[buffer_idx][slot_idx]
```

### 2.2 反量化阶段（Decode节点）

发生在 `KVTunerMHATokenToKVPool.get_kv_buffer()` 中：

```
输入: layer_id
  │
  ├─ 获取底层buffer: k_buffer, v_buffer (可能包含全精度残差)
  │
  ├─ 遍历该层所有已量化slot: self.quantized_slots[buffer_idx]
  │    for slot_idx in quantized_slots:
  │      q_k = self.quantized_k_cache[buffer_idx][slot_idx]  # KVTunerQuantizedTensor
  │      q_v = self.quantized_v_cache[buffer_idx][slot_idx]
  │
  │    k_dequant = q_k.dequantize()  # 反量化
  │    v_dequant = q_v.dequantize()
  │
  │    k_buffer[slot_idx] = k_dequant  # 写回底层buffer
  │    v_buffer[slot_idx] = v_dequant
  │
  └─ 返回 (k_buffer, v_buffer)
```

### 2.3 逐层量化配置优先级

```
_get_quant_method_for_layer(layer_id, forward_mode):
  1. (layer_id, mode_key) → per-layer + mode-specific量化器 (最高)
  2. (layer_id, "base")   → per-layer base量化器
  3. mode_config          → 全局mode-specific量化器
  4. global base          → 全局基础量化器 (最低)
```

### 2.4 配置示例 (Qwen2.5-7B, 32层)

```json
{
  "layer_configs": [
    {"layer_id": 0,  "nbits_key": 8, "nbits_value": 8, "residual_length": 256},
    {"layer_id": 1,  "nbits_key": 8, "nbits_value": 8, "residual_length": 256},
    {"layer_id": 2,  "nbits_key": 4, "nbits_value": 4, "residual_length": 128},
    ...
    {"layer_id": 29, "nbits_key": 8, "nbits_value": 8, "residual_length": 256},
    {"layer_id": 30, "nbits_key": 8, "nbits_value": 8, "residual_length": 256},
    {"layer_id": 31, "nbits_key": 8, "nbits_value": 8, "residual_length": 256}
  ]
}
```

## 三、CUDA Graph兼容性问题

### 3.1 问题描述

Decode节点启用CUDA Graph时，在capture阶段崩溃：
```
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
```

### 3.2 根因分析

CUDA Graph要求capture期间所有操作是**确定性的、可重放的**，不允许：
- Python级别的动态数据结构操作
- CPU-GPU同步（如`.item()`）
- 动态内存分配
- CPU条件分支（基于GPU数据）

当前KVTuner实现中，**反量化路径**（`get_kv_buffer`）和**量化路径**（`set_kv_buffer`）都包含这些不兼容操作：

#### ❌ 不兼容操作清单

| 操作 | 位置 | 原因 |
|------|------|------|
| Python dict迭代 | `get_kv_buffer()` L512 | `for slot_idx in self.quantized_slots[buffer_idx]` |
| Python dict查找 | `get_kv_buffer()` L513 | `self.quantized_k_cache[buffer_idx][slot_idx]` |
| 动态张量创建 | `get_kv_buffer()` L516 | `q_k.dequantize()` 可能触发动态分配 |
| 动态条件分支 | `get_kv_buffer()` L519-527 | `if k_buffer.dim() == 3 ...` |
| Python set操作 | `set_kv_buffer()` L477 | `self.quantized_slots[buffer_idx].discard(slot_idx)` |
| CPU-GPU同步 | `set_kv_buffer()` L444 | `int(positions[i].item())` |
| Python str格式化 | `get_kv_buffer()` L531 | `logger.warning(f"...")` |

#### CUDA Graph Capture流程

```
cuda_graph_runner.py: capture_one_batch_size()
  → 创建 ForwardBatch (forward_mode=DECODE)
  → 调用 _capture_graph(graph, pool, stream, run_once_fn)
    → 调用 model forward (DECODE模式)
      → attention backend: forward()
        → 调用 get_kv_buffer(layer_id)  ← 这里触发KVTuner反量化
          → Python dict操作 ← ❌ CUDA Graph不允许
```

### 3.3 Prefill节点为什么不受影响？

Prefill节点在`ForwardMode.EXTEND`模式下工作。关键区别：
- CUDA Graph capture使用`ForwardMode.DECODE`模式
- Decode模式下`get_kv_buffer`在每次forward时被调用
- 如果capture时有预填充的KV cache，quantized_slots非空，就触发反量化
- Prefill的CUDA Graph可能不经过`get_kv_buffer`（extend模式使用不同路径）

## 四、CUDA Graph兼容性改造方案

### 方案核心思路

将所有Python动态操作替换为**固定大小的GPU张量操作**：

### 4.1 量化存储改造

```python
# 当前 (Python dict, 不兼容CUDA Graph)
self.quantized_k_cache = {0: {slot0: KVTunerQuantizedTensor, ...}, ...}

# 目标 (固定大小GPU张量)
self.quantized_k_cache = torch.zeros(max_tokens, head_num, head_dim // pack_factor, dtype=torch.int32, device=device)
self.quant_scale_k = torch.zeros(max_tokens, head_num, dtype=torch.float16, device=device)  # 量化scale
self.quant_mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)  # 标记哪些slot已量化
```

### 4.2 反量化改造

```python
# 当前 (Python循环, 不兼容CUDA Graph)
for slot_idx in self.quantized_slots[buffer_idx]:
    q_k = self.quantized_k_cache[buffer_idx][slot_idx]
    k_buffer[slot_idx] = q_k.dequantize()

# 目标 (fused CUDA kernel)
k_dequant = fused_dequantize_kernel(
    quantized_k,           # [max_tokens, head_num, packed_dim]
    quant_scale_k,         # [max_tokens, head_num]
    quant_mask,            # [max_tokens] bool
    out=k_buffer,          # 直接写入output buffer
    nbits=4,               # 可per-layer配置
    axis=0,                # 量化axis
    group_size=64,         # 量化group
)
```

### 4.3 量化条件判断改造

```python
# 当前 (CPU条件, 不兼容CUDA Graph)
distance_from_end = seq_length - token_position
if distance_from_end < residual_length:
    keep_full_precision = True

# 目标 (GPU mask)
# residual_length作为常量（启动时固定）
residual_mask = (seq_length - positions) < residual_length  # GPU上计算
# 使用torch.where替代Python if/else
```

### 4.4 残差管理改造

```python
# 当前 (Python dict跟踪)
self.slot_positions[buffer_idx][slot_idx] = token_position

# 目标 (固定大小GPU张量)
self.slot_positions = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
self.slot_positions[loc] = positions  # 直接tensor操作
```

### 4.5 实施优先级

1. **P0**: 反量化路径CUDA Graph兼容（Decode节点性能关键）
   - 用fused kernel替代Python循环
   - 用GPU tensor替代Python dict

2. **P1**: 量化路径CUDA Graph兼容（Prefill节点）
   - 移除`.item()`调用
   - 用GPU mask替代Python条件判断

3. **P2**: 逐层配置的CUDA Graph兼容
   - 每层的nbits/group_size在启动时确定，可以作为常量
   - 使用per-layer的fused kernel或参数化的通用kernel

## 五、当前测试结果汇总

### 5.1 基础功能验证

| 测试项 | 状态 | 备注 |
|--------|------|------|
| 官方sglang P/D分离 | ✅ | NIXL/TCP, 无RDMA |
| 适配版本P/D分离 | ✅ | kvtuner-splitwise-realign分支 |
| KVTuner统一量化 + P/D分离 | ✅ | 4-bit, --disable-cuda-graph |
| KVTuner逐层量化 + P/D分离 | ✅ | 32层不同精度, --disable-cuda-graph |
| KVTuner + CUDA Graph (Decode) | ❌ | 需要改造 |

### 5.2 性能数据 (Qwen2.5-7B, RTX 4090)

| 配置 | 英文20tok | 中文50tok | 中文300tok |
|------|-----------|-----------|------------|
| 无量化 | 0.45s | 0.89s | ~5.5s |
| 统一4-bit量化 | 0.76s | 0.39s | 4.44s |
| 逐层量化(4/8-bit) | 0.78s | 0.47s | 6.67s |

### 5.3 已知限制

- KVTuner仅支持MHA模型，MLA（DeepSeek-v2/v3）不支持
- Decode节点需要`--disable-cuda-graph`
- 与FP8 KV cache不兼容
- 反量化使用Python循环，性能未优化
- 6-bit量化不支持（仅2/4/8-bit）

## 六、下一步工作建议

1. **CUDA Graph兼容改造**（最重要）
   - 实现fused dequantize CUDA kernel
   - 用GPU tensor替代Python dict存储量化数据
   - 预计可提升Decode节点2-3x吞吐量

2. **性能优化**
   - 反量化与attention计算融合（避免额外内存读写）
   - Prefill端量化与KV cache写入融合

3. **功能完善**
   - 支持prefill/decode模式的差异化量化参数
   - 量化精度自动搜索/调优
   - 量化感知的KV Cache eviction策略

4. **评估验证**
   - 量化对模型精度的影响（perplexity benchmark）
   - 内存节省实际测量
   - 端到端吞吐量对比
