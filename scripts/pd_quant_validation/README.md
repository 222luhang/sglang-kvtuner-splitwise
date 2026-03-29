# KVTuner + P/D Disaggregation 部署与测试指南

> **最后更新**: 2026-03-29
> **状态**: ✅ 已验证通过

## 概述

本项目将 KVTuner (ICML2025) 层级量化适配到 sglang 框架，并验证其在 P/D (Prefill/Decode) 分离架构下的推理能力。

**核心组件**:
1. **KVTuner**: 感知层级的混合精度 KV Cache 量化
2. **P/D Disaggregation**: sglang 原生的 Prefill/Decode 分离推理
3. **sglang_router**: 请求路由网关（P/D 分离必需）

---

## 1. 架构

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │ HTTP
                    ┌──────▼──────┐
                    │   Router    │  ← sglang_router (端口 8000)
                    │  :8000      │
                    └──┬───────┬──┘
                       │       │
            ┌──────────▼─┐   ┌▼──────────┐
            │  Prefill   │   │  Decode   │
            │  :30000    │──▶│  :30000   │
            │  Bootstrap │NIXL│           │
            │  :8998     │   │           │
            └────────────┘   └───────────┘
```

**关键点**:
- **必须通过 Router 发送请求**，不能直接向 Prefill 发送（会报 "without bootstrap room id"）
- Router 内部自动处理 bootstrap_room 的分配和路由
- Bootstrap 端口 8998 由 Prefill 服务内建管理，不需要额外启动

---

## 2. 环境要求

### 硬件
- 2台以上服务器，每台至少 1 张 GPU（≥24GB 显存）
- 节点间网络互通（TCP 即可，无需 InfiniBand）

### 软件
- Python 3.12 + venv 虚拟环境 (`~/.venv`)
- CUDA 工具链
- ninja-build（编译内核需要）

### 文件位置
| 项目 | 路径 |
|------|------|
| 代码仓库 | `~/sglang-kvtuner-splitwise` |
| 分支 | `kvtuner-splitwise` |
| 模型 | `/data/Qwen/Qwen2.5-7B` |
| 量化配置 | `/data/qwen2.5-7b_layer_quant.json` |
| 虚拟环境 | `~/.venv` |

---

## 3. 部署步骤

### 3.1 所有节点：拉取代码

```bash
cd ~/sglang-kvtuner-splitwise
git fetch origin
git checkout kvtuner-splitwise
git pull origin kvtuner-splitwise
```

### 3.2 Prefill 节点：启动 Prefill 服务

```bash
cd ~/sglang-kvtuner-splitwise
source ~/.venv/bin/activate

python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disable-custom-all-reduce \
  --enable-kvtuner-quant \
  --kvtuner-layer-config /data/qwen2.5-7b_layer_quant.json \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --watchdog-timeout 9999
```

**参数说明**:
- `--disaggregation-mode prefill`: 启用 prefill 模式
- `--disaggregation-transfer-backend nixl`: KV Cache 传输后端
- `--enable-kvtuner-quant`: 启用 KVTuner 量化
- `--kvtuner-layer-config`: 层级量化配置文件（每层不同的量化位数）
- `--disable-cuda-graph`: ⚠️ **必须禁用**，否则 CUDA graph 捕获时会因 KVTuner 的 `tolist()` 操作失败
- `--watchdog-timeout 9999`: 延长超时，避免空闲时被杀

等待日志出现 `"The server is fired up and ready to roll!"`

### 3.3 Decode 节点：启动 Decode 服务

```bash
cd ~/sglang-kvtuner-splitwise
source ~/.venv/bin/activate

python -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disable-custom-all-reduce \
  --enable-kvtuner-quant \
  --kvtuner-layer-config /data/qwen2.5-7b_layer_quant.json \
  --mem-fraction-static 0.6 \
  --disable-cuda-graph \
  --watchdog-timeout 9999
```

**与 Prefill 的区别**:
- `--disaggregation-mode decode`
- `--mem-fraction-static 0.6`: Decode 节点需要更多预留显存给 KV Cache，建议用 0.6
- **不要** 加 `--dist-init-addr`（那是 TP/PP 的 TCPStore，不是 disaggregation 的）

### 3.4 Router 节点（通常在 Prefill 同一台机器）：启动 Router

```bash
cd ~/sglang-kvtuner-splitwise
source ~/.venv/bin/activate

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<PREFILL_IP>:30000 8998 \
  --decode http://<DECODE_IP>:30000 \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path /data/Qwen/Qwen2.5-7B
```

**参数说明**:
- `--prefill URL BOOTSTRAP_PORT`: Prefill 地址和 Bootstrap 端口（8998 是 sglang 默认）
- `--decode URL`: Decode 地址
- `--model-path`: Router 需要加载 tokenizer，必须指定

等待日志出现 `"Workflow completed"` 和 `"Activated N worker(s)"`

### 3.5 健康检查

```bash
# Prefill
curl -s http://<PREFILL_IP>:30000/health

# Decode
curl -s http://<DECODE_IP>:30000/health

# Router
curl -s http://<ROUTER_IP>:8000/health
```

---

## 4. 测试

### 4.1 基本推理测试

```bash
curl -s http://<ROUTER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/Qwen/Qwen2.5-7B",
    "messages": [{"role": "user", "content": "What is machine learning? Explain in 2 sentences."}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**预期**: HTTP 200，返回正常的 chat completion 响应。

### 4.2 多轮稳定性测试

```bash
for i in $(seq 1 5); do
  echo "=== Request $i ==="
  curl -s -w "\nHTTP:%{http_code} TIME:%{time_total}s\n" \
    http://<ROUTER_IP>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/data/Qwen/Qwen2.5-7B",
      "messages": [{"role": "user", "content": "Explain transformers briefly."}],
      "max_tokens": 80,
      "temperature": 0.7
    }' | grep -E "HTTP:|finish_reason"
done
```

---

## 5. 层级量化配置

配置文件格式 (`/data/qwen2.5-7b_layer_quant.json`):

```json
{
  "layer_configs": [
    {"layer_id": 0, "nbits_key": 8, "nbits_value": 8},
    {"layer_id": 1, "nbits_key": 4, "nbits_value": 4},
    ...
    {"layer_id": 31, "nbits_key": 8, "nbits_value": 8}
  ]
}
```

**⚠️ 注意事项**:
- `nbits_key` 和 `nbits_value` 只支持 **2、4、8**，不支持其他值（如 6）
- 首层和末层通常用 8-bit 保证精度
- 中间层用 4-bit 节省显存
- 量化位数不影响 Prefill/Decode 的 KV Cache 传输，两节点必须使用相同的配置文件

---

## 6. 常见问题排查

### "Disaggregated request received without bootstrap room id"
**原因**: 直接向 Prefill 发送请求，没有通过 Router
**解决**: 所有请求必须发送到 Router (端口 8000)

### "CUDA error: operation not permitted when stream is capturing"
**原因**: CUDA graph 捕获期间 KVTuner 的 `tolist()` 操作不被允许
**解决**: 添加 `--disable-cuda-graph` 参数

### "nbits_key must be 2, 4, or 8, got 6"
**原因**: 层级量化配置文件中使用了不支持的位数
**解决**: 修改配置文件，只使用 2/4/8

### "Capture cuda graph failed: CUDA error: operation failed due to a previous error during capture"
**原因**: Decode 节点 CUDA graph 捕获失败，通常由显存不足引起
**解决**: 
1. 添加 `--disable-cuda-graph`
2. 降低 `--mem-fraction-static`（推荐 0.6）
3. 降低 `--cuda-graph-max-bs`（如 16）

### "Address already in use" (端口 8998)
**原因**: Prefill 服务启动后会自动占用 8998 端口作为 bootstrap
**解决**: 不要尝试在同一台机器上手动启动 bootstrap 服务

### "TCP client failed to connect to host X:8998 - Ping failed"
**原因**: 错误地使用了 `--dist-init-addr` 参数
**解决**: Decode 节点不需要 `--dist-init-addr`，这是 TP/PP 的通信参数

### 服务启动后空闲几分钟就挂掉
**原因**: sglang 默认 watchdog 超时 300 秒
**解决**: 添加 `--watchdog-timeout 9999`

### server_args.py 参数重复报错
**原因**: 分支合并冲突导致 server_args.py 中 KVTuner 参数重复定义
**解决**: 检查并删除重复的字段定义和 CLI 参数

---

## 7. 多节点扩展

如果有多台 Prefill/Decode 节点，Router 支持多个节点：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<prefill1_ip>:30000 8998 \
  --prefill http://<prefill2_ip>:30000 8998 \
  --decode http://<decode1_ip>:30000 \
  --decode http://<decode2_ip>:30000 \
  --host 0.0.0.0 --port 8000 \
  --model-path /data/Qwen/Qwen2.5-7B
```

每个 Prefill 节点需要指定其 Bootstrap 端口（默认 8998）。

---

## 8. 历史参考

过期的文档已归档到 `_archive/` 目录，包含：
- 早期部署脚本（已废弃的 bootstrap_service_simple.py 方案）
- 各阶段的测试报告和验证总结
- NIXL 修复记录、Mooncake 配置等历史信息

如需查阅历史记录，参考 `_archive/` 目录。
