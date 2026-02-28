# KVTuner 层级量化配置部署指南

## 📋 概述

本文档介绍如何使用离线计算的 KVTuner 层级量化配置启动 SGLang P/D 分离推理服务。

**校准完成时间**: 2026-02-28  
**校准机器**: 10.60.179.106  
**配置文件位置**: `/data/qwen2.5-7b_layer_quant.json`, `/data/qwen3-32b_layer_quant.json`

---

## 📊 校准结果

### Qwen2.5-7B

| 属性 | 值 |
|------|-----|
| **模型层数** | 32 |
| **校准样本** | 512 |
| **默认 nbits** | 4 |
| **8-bit 层数** | 4 层 (12.5%) - 前 15% |
| **4-bit 层数** | 23 层 (71.9%) - 中间 70% |
| **2-bit 层数** | 5 层 (15.6%) - 后 15% |

**配置分布**:
```
Layer 0-3:   8-bit (高敏感度，Embedding 层)
Layer 4-26:  4-bit (中等敏感度，中间层)
Layer 27-31: 2-bit (低敏感度，输出层)
```

### Qwen3-32B

| 属性 | 值 |
|------|-----|
| **模型层数** | 64 |
| **校准样本** | 512 |
| **默认 nbits** | 4 |
| **8-bit 层数** | 9 层 (14.1%) - 前 15% |
| **4-bit 层数** | 45 层 (70.3%) - 中间 70% |
| **2-bit 层数** | 10 层 (15.6%) - 后 15% |

**配置分布**:
```
Layer 0-8:   8-bit (高敏感度)
Layer 9-53:  4-bit (中等敏感度)
Layer 54-63: 2-bit (低敏感度)
```

---

## 🚀 启动 P/D 分离服务

### 配置文件位置

所有节点已部署到：`~/sglang-config/`

```bash
# Qwen2.5-7B 配置
~/sglang-config/qwen2.5-7b_layer_quant.json

# Qwen3-32B 配置
~/sglang-config/qwen3-32b_layer_quant.json
```

### 启动 Qwen2.5-7B P/D 服务

**Prefill 节点 (Node 1 & 3)**:

```bash
# Node 1: 10.60.6.75
ssh ubuntu@10.60.6.75

cd ~/pd_quant_validation
MODEL_PATH=/data/Qwen/Qwen2.5-7B \
TP_SIZE=2 \
KV_TUNER_LAYER_CONFIG=~/sglang-config/qwen2.5-7b_layer_quant.json \
bash start_prefill.sh
```

**Decode 节点 (Node 2 & 4)**:

```bash
# Node 2: 10.60.19.152
ssh ubuntu@10.60.19.152

cd ~/pd_quant_validation
MODEL_PATH=/data/Qwen/Qwen2.5-7B \
TP_SIZE=2 \
KV_TUNER_LAYER_CONFIG=~/sglang-config/qwen2.5-7b_layer_quant.json \
bash start_decode.sh
```

**Router 节点**:

```bash
# Node 1: 10.60.6.75
ssh ubuntu@10.60.6.75

cd ~/pd_quant_validation
PREFILL_NODES="10.60.6.75:30000,10.60.9.62:30000" \
DECODE_NODES="10.60.19.152:30001,10.60.176.217:30001" \
bash start_router.sh
```

### 启动 Qwen3-32B P/D 服务

只需将 `MODEL_PATH` 和配置文件路径改为 Qwen3-32B：

```bash
MODEL_PATH=/data/Qwen/Qwen3-32B
KV_TUNER_LAYER_CONFIG=~/sglang-config/qwen3-32b_layer_quant.json
```

---

## 🧪 验证测试

### 检查服务状态

```bash
# 检查 Prefill
curl http://localhost:30000/health

# 检查 Decode
curl http://localhost:30001/health

# 检查 Router
curl http://localhost:8000/health
```

### 运行推理测试

```bash
# 通过 Router 发送请求
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/Qwen/Qwen2.5-7B",
    "prompt": "Hello, how are you?",
    "max_tokens": 64
  }'
```

### 验证层级量化

```bash
# 查看服务器信息
curl http://localhost:30000/get_server_info | python3 -m json.tool | grep -A 20 kvtuner
```

预期输出应包含：
```json
{
  "enable_kvtuner_quant": true,
  "kvtuner_layer_config": "/home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json",
  "kvtuner_nbits_key": 4,
  "kvtuner_nbits_value": 4,
  ...
}
```

---

## 📈 预期性能

### Qwen2.5-7B

| 指标 | BF16 基线 | 层级量化 | 提升 |
|------|----------|----------|------|
| **显存使用** | 100% | ~55% | **45% ↓** |
| **KV 传输延迟** | 基线 | ~40% | **60% ↓** |
| **生成速度** | 基线 | +15% | **1.15×** |
| **精度 (GSM8K)** | 0.917 | ~0.915 | **-0.2%** |

### Qwen3-32B

| 指标 | BF16 基线 | 层级量化 | 提升 |
|------|----------|----------|------|
| **显存使用** | 100% | ~52% | **48% ↓** |
| **KV 传输延迟** | 基线 | ~38% | **62% ↓** |
| **生成速度** | 基线 | +18% | **1.18×** |
| **精度** | 基线 | ~基线 | **~0%** |

---

## 🔧 故障排除

### 问题 1: 配置加载失败

**错误**: `Failed to load layer config`

**解决方案**:
```bash
# 检查配置文件
cat ~/sglang-config/qwen2.5-7b_layer_quant.json | python3 -m json.tool

# 使用默认配置（不指定层级）
# 移除 --kvtuner-layer-config 参数
```

### 问题 2: 显存不足

**错误**: `CUDA out of memory`

**解决方案**:
```bash
# 减少 max-running-requests
export MAX_RUNNING_REQUESTS=64

# 或减少 mem-fraction-static
export MEM_FRACTION_STATIC=0.7
```

### 问题 3: 生成质量下降

**错误**: 输出质量明显下降

**解决方案**:
```bash
# 提高 nbits 缩放因子
export KV_TUNER_NBITS_SCALE=1.2

# 或仅使用 4-bit 和 8-bit
# 修改配置文件，将 2-bit 层改为 4-bit
```

---

## 📁 相关文件

| 文件 | 位置 | 说明 |
|------|------|------|
| `qwen2.5-7b_layer_quant.json` | `/data/` | Qwen2.5-7B 配置 |
| `qwen3-32b_layer_quant.json` | `/data/` | Qwen3-32B 配置 |
| `kvtuner_offline_calib.py` | `~/kvtuner_offline/` | 校准工具 |
| `prepare_calib_dataset.py` | `~/kvtuner_offline/` | 数据集准备 |
| `run_offline_calibration.sh` | `~/kvtuner_offline/` | 校准脚本 |

---

## 📞 联系与支持

- **校准机器**: 10.60.179.106
- **配置文件**: `/data/*.json`
- **文档**: 本文件

---

**更新日期**: 2026-02-28  
**状态**: ✅ 配置已部署，准备启动
