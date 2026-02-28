# P/D 分离与量化结合验证 - 部署指南

## 1. 环境准备

### 1.1 硬件要求

| 机器 | IP | 角色 | GPU | 内存 |
|------|-----|------|-----|------|
| Node 1 | 10.60.6.75 | Prefill | ≥1×A100/H100 | ≥64GB |
| Node 2 | 10.60.19.152 | Decode | ≥1×A100/H100 | ≥64GB |
| Node 3 | 10.60.9.62 | Prefill | ≥1×A100/H100 | ≥64GB |
| Node 4 | 10.60.176.217 | Decode | ≥1×A100/H100 | ≥64GB |

### 1.2 网络要求

- 所有节点之间需要高速网络连接（推荐 InfiniBand 或 RoCE）
- 节点间延迟 <1ms
- 带宽 ≥100Gbps

### 1.3 软件要求

- Ubuntu 20.04/22.04
- NVIDIA Driver ≥535.00
- CUDA ≥12.1
- Python ≥3.10
- uv (推荐) 或 pip

## 2. 快速部署

### 2.1 在控制节点执行

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation

# 步骤 1: 部署到所有节点
./deploy_to_nodes.sh
```

### 2.2 在各节点分别执行

#### Node 1 & Node 3 (Prefill 服务)

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation

# 设置环境变量（可选）
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export KV_TUNER_NBITS_KEY=4
export KV_TUNER_NBITS_VALUE=4
export KV_TUNER_RESIDUAL_LENGTH=256

# 启动 Prefill 服务
./start_prefill.sh
```

#### Node 2 & Node 4 (Decode 服务)

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation

# 设置环境变量（可选）
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export KV_TUNER_NBITS_KEY=8  # Decode 使用更高精度
export KV_TUNER_NBITS_VALUE=8
export KV_TUNER_RESIDUAL_LENGTH=32  # Decode 使用更短残差

# 启动 Decode 服务
./start_decode.sh
```

#### 任一节点 (Router)

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation

# 设置 Prefill 和 Decode 节点地址
export PREFILL_NODES="10.60.6.75:30000,10.60.9.62:30000"
export DECODE_NODES="10.60.19.152:30001,10.60.176.217:30001"

# 启动 Router
./start_router.sh
```

## 3. 验证测试

### 3.1 检查服务状态

在任意节点运行：

```bash
./check_status.sh
```

预期输出：
```
服务状态:
----------------------------------------
Prefill (:30000)     ● 运行中
  模型：meta-llama/Llama-3.1-8B-Instruct...
  模式：prefill
  KV 量化：fp8_e5m2
Decode (:30001)      ● 运行中
  模型：meta-llama/Llama-3.1-8B-Instruct...
  模式：decode
  KV 量化：fp8_e5m2
Router (:8000)       ● 运行中
```

### 3.2 运行完整验证

```bash
./run_validation.sh
```

验证测试包括：
1. ✅ 服务健康检查
2. ✅ 服务器信息验证
3. ✅ 基本推理测试
4. ✅ 量化功能验证
5. ✅ P/D 分离模式验证
6. ✅ 性能基准测试
7. ✅ 内存使用检查

## 4. 配置选项

### 4.1 Prefill 服务配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_PATH` | meta-llama/Llama-3.1-8B-Instruct | 模型路径 |
| `PORT` | 30000 | 服务端口 |
| `TP_SIZE` | 1 | 张量并行大小 |
| `KV_CACHE_DTYPE` | fp8_e5m2 | KV Cache 量化类型 |
| `KV_TUNER_NBITS_KEY` | 4 | KVTuner Key 量化位数 |
| `KV_TUNER_NBITS_VALUE` | 4 | KVTuner Value 量化位数 |
| `KV_TUNER_RESIDUAL_LENGTH` | 256 | Prefill 残差长度 |
| `DISAGGREGATION_TRANSFER_BACKEND` | mooncake | 传输后端 |

### 4.2 Decode 服务配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_PATH` | meta-llama/Llama-3.1-8B-Instruct | 模型路径 |
| `PORT` | 30001 | 服务端口 |
| `TP_SIZE` | 1 | 张量并行大小 |
| `KV_CACHE_DTYPE` | fp8_e5m2 | KV Cache 量化类型 |
| `KV_TUNER_NBITS_KEY` | 8 | KVTuner Key 量化位数 |
| `KV_TUNER_NBITS_VALUE` | 8 | KVTuner Value 量化位数 |
| `KV_TUNER_RESIDUAL_LENGTH` | 32 | Decode 残差长度 |
| `DISAGGREGATION_TRANSFER_BACKEND` | mooncake | 传输后端 |

### 4.3 量化配置推荐

| 场景 | Prefill 配置 | Decode 配置 | 说明 |
|------|-------------|-------------|------|
| 高吞吐 | K4/V4, 残差 256 | K8/V8, 残差 32 | 平衡性能和精度 |
| 低延迟 | K8/V8, 残差 128 | K8/V8, 残差 16 | 最小化反量化开销 |
| 极限内存 | K4/V4, 残差 512 | K4/V4, 残差 64 | 最大压缩比 |

## 5. 监控与调试

### 5.1 查看日志

```bash
# 查看最新日志
tail -f /tmp/sglang_logs/prefill_*.log
tail -f /tmp/sglang_logs/decode_*.log
tail -f /tmp/sglang_logs/router_*.log
```

### 5.2 检查量化统计

```python
import requests

# 获取 Prefill 服务信息
response = requests.get("http://10.60.6.75:30000/get_server_info")
info = response.json()
print(f"KV Cache 量化：{info.get('kv_cache_dtype')}")
print(f"KVTuner 启用：{info.get('enable_kvtuner_quant')}")
```

### 5.3 性能监控

```bash
# 使用 nvidia-smi 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控网络吞吐
iftop -P -n
```

## 6. 故障排除

### 6.1 常见问题

#### 问题：服务启动失败

**解决方案:**
1. 检查 GPU 可用性：`nvidia-smi`
2. 检查端口占用：`ss -tlnp | grep 30000`
3. 查看日志：`cat /tmp/sglang_logs/*.log`

#### 问题：KV Cache 传输失败

**解决方案:**
1. 检查网络连接：`ping <peer_node>`
2. 检查 InfiniBand/RoCE：`ibstat`
3. 验证传输后端配置

#### 问题：量化后精度下降

**解决方案:**
1. 增加量化位数（4-bit → 8-bit）
2. 增加残差长度
3. 检查模型是否适合量化

### 6.2 获取帮助

```bash
# 查看脚本帮助
./start_prefill.sh --help
./start_decode.sh --help
./run_validation.sh --help
```

## 7. 清理

### 7.1 停止服务

```bash
# 停止所有 SGLang 进程
pkill -f "sglang.*launch_server"
pkill -f "sglang_router"

# 清理日志
rm -rf /tmp/sglang_logs/*
```

### 7.2 卸载（可选）

```bash
# 卸载 SGLang（如果需要）
pip uninstall sglang
```

## 8. 性能基准

### 8.1 预期性能指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| Prefill 吞吐 | ≥1000 tokens/s | 8B 模型，4-bit 量化 |
| Decode 延迟 | ≤50ms/token | 8B 模型，8-bit 量化 |
| 内存节省 | ≥50% | 相比 BF16 |
| 精度损失 | <1% | 相比 BF16 基线 |

### 8.2 基准测试

```bash
# 运行详细基准测试
python3 benchmark/benchmark_serving.py \
    --backend sglang \
    --base-url http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 1000 \
    --request-rate inf
```

## 9. 参考资料

- [SGLang 文档](https://docs.sglang.io)
- [KVTuner 使用说明](../../KVTUNER_USAGE.md)
- [P/D 分离文档](../../docs/advanced_features/pd_disaggregation.md)
- [量化 KV Cache 文档](../../docs/advanced_features/quantized_kv_cache.md)
