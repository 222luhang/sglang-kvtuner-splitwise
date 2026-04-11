# 基于逐层自适应量化的 KV Cache 传输加速方案

## 1. 背景与问题

在 PD（Prefill-Decode）分离架构中，prefill 节点计算完 KV Cache 后需通过 TCP 传输至 decode 节点。
当前实现中：

- **KVTuner 量化**：在本地 GPU 上对 KV Cache 做 int8 量化以节省显存，但 TCP 传输时仍发送原始 BF16 数据
- **TCP 逐层管道**：支持逐层发送（layer-wise pipeline），与下一层计算重叠，但传输的是未压缩的 BF16 字节

两个模块各自独立，未产生加速效果。TCP 传输数据量大，是 TTFT（Time To First Token）的主要瓶颈。

## 2. 方案目标

将量化集成到 TCP 传输管道中，实现：

1. **减少传输数据量**：int8 量化 = 2x 压缩，4-bit 量化 = 4x 压缩
2. **逐层流水线重叠**：量化/传输与下一层计算并行
3. **逐层可变精度**：敏感层高精度、非敏感层低精度，平衡质量与速度

## 3. 数据流对比

### 当前流程（无量化传输）

```
Prefill GPU ──[compute layer_i KV]──> GPU Pool (BF16)
                                         │
                                    GPU→CPU copy (BF16, 2 bytes/elem)
                                         │
                                    TCP send (BF16 原始字节)
                                         │
                                    TCP recv (BF16)
                                         │
                                    CPU→GPU copy (BF16)
                                         │
                                    Decode GPU Pool (BF16)
```

### 改进流程（量化传输）

```
Prefill GPU ──[compute layer_i KV]──> GPU Pool (BF16)
                                         │
                                    GPU→CPU copy (BF16)
                                         │
                                    CPU 量化 (BF16 → int8 + scales)  ← 新增
                                         │
                                    TCP send (int8 + scales, ~50% 数据量)
                                         │
                                    TCP recv (int8 + scales)
                                         │
                                    CPU 反量化 (int8 + scales → BF16)  ← 新增
                                         │
                                    CPU→GPU copy (BF16)
                                         │
                                    Decode GPU Pool (BF16)
```

## 4. 压缩比分析

以 BF16（2 bytes/element）为基准，group_size=64：

| 量化位宽 | 数据字节 | scales 开销 | 总大小 | 压缩比 |
|----------|---------|------------|--------|--------|
| 无量化   | 2N      | 0          | 2N     | 1.0x   |
| 8-bit    | N       | N/64 × 2   | ~1.03N | ~1.94x |
| 4-bit    | N/2     | N/64 × 2   | ~0.53N | ~3.77x |

其中 N 为元素数量，scales 使用 float16 存储。

## 5. 详细设计

### 5.1 传输量化工具函数（新建文件）

**文件**: `python/sglang/srt/disaggregation/tcp/transfer_quant.py`

实现轻量级 numpy 量化/反量化函数，专为 CPU 端传输场景优化：

- `quantize_for_transfer(data, dtype, nbits, group_size)` → 压缩后的 bytes
- `dequantize_from_transfer(data, original_nbytes, dtype, nbits, group_size)` → 原始格式 bytes

打包格式（自包含，接收端无需额外元数据）：

```
[4 bytes: nbits (int32)]
[4 bytes: group_size (int32)]
[4 bytes: num_elements (int32)]
[4 bytes: dtype_code (int32)]
[scales: float16 array, num_groups 个]
[quantized: int8 array, num_elements 个]
```

### 5.2 协议扩展

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py`

利用 layer_id 的高位标记量化消息，向后兼容：

```python
_MSG_QUANT_FLAG = 0x40000000  # layer_id 高位标记

# 发送端：layer_id |= _MSG_QUANT_FLAG
# 接收端：if layer_id & _MSG_QUANT_FLAG → 反量化
```

### 5.3 发送端修改（send_layer）

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py`，`send_layer()` 方法

在 GPU→CPU copy 之后、TCP send 之前插入量化步骤：

```python
k_data = _read_pages_from_gpu(...)
v_data = _read_pages_from_gpu(...)

if self._transfer_quant_bits is not None:
    nbits = self._get_layer_quant_bits(layer_id)
    k_data = quantize_for_transfer(k_data, kv_dtype, nbits)
    v_data = quantize_for_transfer(v_data, kv_dtype, nbits)
    k_lid = (layer_id * 2) | _MSG_QUANT_FLAG
    v_lid = (layer_id * 2 + 1) | _MSG_QUANT_FLAG
else:
    k_lid = layer_id * 2
    v_lid = layer_id * 2 + 1

_send_layer_data(conn, k_lid, k_data)
_send_layer_data(conn, v_lid, v_data)
```

### 5.4 接收端修改（_recv_kv）

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py`，`_recv_kv()` 方法

接收时检查量化标记，按需反量化：

```python
layer_id, data = _recv_layer_data(conn)

is_quantized = bool(layer_id & _MSG_QUANT_FLAG)
if is_quantized:
    layer_id &= ~_MSG_QUANT_FLAG

# ... 现有 EOF/aux 处理 ...

if is_quantized:
    data = dequantize_from_transfer(data, item_len * len(dst_kv_indices))

_write_pages_to_gpu(base_ptr, item_len, dst_kv_indices, page_size, data)
```

### 5.5 逐层可变精度

复用现有 `--kvtuner-layer-bits` 配置：

```python
def _get_layer_quant_bits(self, layer_id: int) -> int:
    if self._layer_bits_map and layer_id in self._layer_bits_map:
        return self._layer_bits_map[layer_id]
    return self._transfer_quant_bits  # 全局默认
```

典型配置（32 层模型）：
- 前 5 层（0-4）：8-bit（高敏感度，保质量）
- 中间 22 层（5-26）：4-bit（中敏感度，主要压缩收益）
- 后 5 层（27-31）：8-bit（输出层敏感度较高）

### 5.6 新增启动参数

**文件**: `python/sglang/srt/server_args.py`

```python
enable_transfer_quant: bool = False       # 启用传输量化
transfer_quant_bits: int = 8              # 全局默认量化位宽
```

与现有 `--kvtuner-layer-bits` 配合使用实现逐层精度。

### 5.7 Batch mode 同步修改

`_stream_kv()`（batch 模式发送）也需要同样的量化逻辑，确保 pipeline 和 batch 两种模式行为一致。

## 6. 实现顺序（按复杂度递增）

| 步骤 | 内容 | 复杂度 | 预计改动量 |
|------|------|--------|-----------|
| 1    | 新建 `transfer_quant.py`：numpy 量化/反量化函数 | 低 | ~120 行新文件 |
| 2    | 单元测试：验证量化/反量化正确性和压缩比 | 低 | ~80 行测试 |
| 3    | `server_args.py`：新增传输量化参数 | 低 | ~5 行 |
| 4    | `conn.py`：协议常量 + send_layer 集成量化 | 中 | ~30 行修改 |
| 5    | `conn.py`：_recv_kv 集成反量化 | 中 | ~20 行修改 |
| 6    | `conn.py`：_stream_kv batch mode 同步修改 | 中 | ~25 行修改 |
| 7    | TCPKVSender 初始化：读取量化配置 | 中 | ~20 行修改 |
| 8    | 端到端测试 + 性能基准测试 | 中 | 使用现有脚本 |

## 7. 验证方案

### 正确性验证
- 单元测试：量化→反量化的 MSE、cosine similarity
- 端到端：对比量化传输 vs BF16 传输的模型输出

### 性能测试
- **TTFT**：主要加速指标，对比不同量化位宽
- **每层传输时间**：量化开销 vs 传输节省
- **吞吐量**：tokens/s

### 测试矩阵
| 配置 | 说明 |
|------|------|
| baseline | 无量化传输（BF16） |
| uniform-8bit | 全部层 8-bit 传输 |
| uniform-4bit | 全部层 4-bit 传输 |
| mixed | 前/后层 8-bit + 中间层 4-bit |

## 8. 预期效果

- 8-bit 统一传输：TTFT 降低 ~30-40%
- 4-bit 统一传输：TTFT 降低 ~50-60%
- 混合精度传输：在质量损失可控的前提下接近 4-bit 的加速效果
