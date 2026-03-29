# 快速开始 - P/D 分离与量化验证

## 概述

本目录包含在 4 台机器上验证 SGLang KVTuner 量化与 Prefill/Decode 分离模式结合的完整工具链。

## 📁 模型路径

**已确认的本地模型** (`/data/Qwen/`):

| 模型 | 路径 | 推荐用途 |
|------|------|----------|
| Qwen2.5-7B | `/data/Qwen/Qwen2.5-7B` | ✅ 推荐测试 (显存友好) |
| Qwen3-32B | `/data/Qwen/Qwen3-32B` | 大规模测试 (需要更多显存) |

**使用方式**:
```bash
# 使用 Qwen2.5-7B (默认)
export MODEL_PATH="/data/Qwen/Qwen2.5-7B"

# 使用 Qwen3-32B
export MODEL_PATH="/data/Qwen/Qwen3-32B"
```

## 机器配置

| 角色 | 机器 | IP | 端口 | GPU |
|------|------|-----|------|-----|
| Prefill 1 | Node 1 | 10.60.6.75 | 30000 | 2×RTX 3090 |
| Decode 1 | Node 2 | 10.60.19.152 | 30001 | 2×RTX 3090 |
| Prefill 2 | Node 3 | 10.60.9.62 | 30000 | 2×RTX 3090 |
| Decode 2 | Node 4 | 10.60.176.217 | 30001 | 2×RTX 3090 |
| Router | Node 1 | - | 8000 | - |

**网络**: TCP (无 IB/RoCE)  
**传输后端**: NIXL (UCX over TCP)

## 凭据

- **用户名**: ubuntu
- **密码**: luhang222
- **Python 环境**: .venv (uv)

## 部署方式

### 方式 A: 一键部署（推荐）

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation
chmod +x deploy_and_run.sh
./deploy_and_run.sh
```

此脚本会自动：
1. 测试所有节点连接
2. 部署脚本到所有节点
3. 启动 Prefill 和 Decode 服务
4. 启动 Router
5. 运行验证测试

### 方式 B: 手动部署

**步骤 1: 部署脚本**
```bash
./deploy_to_nodes.sh
```

**步骤 2: 在各节点启动服务**

Node 1 & 3 (Prefill):
```bash
ssh ubuntu@10.60.6.75 "cd ~/pd_quant_validation && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_prefill.sh"
ssh ubuntu@10.60.9.62 "cd ~/pd_quant_validation && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_prefill.sh"
```

Node 2 & 4 (Decode):
```bash
ssh ubuntu@10.60.19.152 "cd ~/pd_quant_validation && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_decode.sh"
ssh ubuntu@10.60.176.217 "cd ~/pd_quant_validation && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_decode.sh"
```

Router (Node 1):
```bash
ssh ubuntu@10.60.6.75 "cd ~/pd_quant_validation && PREFILL_NODES='10.60.6.75:30000,10.60.9.62:30000' DECODE_NODES='10.60.19.152:30001,10.60.176.217:30001' ./start_router.sh"
```

**步骤 3: 运行验证**
```bash
ssh ubuntu@10.60.6.75 "cd ~/pd_quant_validation && python3 validate_pd_quant.py --output validation_report.json"
```

## 验证检查清单

- [ ] 所有节点 SSH 可访问
- [ ] GPU 驱动和 CUDA 已安装
- [ ] SGLang 环境已准备（.venv）
- [ ] 节点间网络连通
- [ ] 防火墙端口已开放（30000, 30001, 8000）

## 常用命令

```bash
# 检查服务状态
./check_status.sh

# 查看日志
tail -f /tmp/sglang_logs/*.log

# 停止服务
pkill -f "sglang.*launch_server"
pkill -f "sglang_router"

# 测试单个服务
curl http://localhost:30000/health
curl http://localhost:30001/health
curl http://localhost:8000/health
```

## 配置示例

### 使用不同量化配置

**Prefill (高吞吐):**
```bash
export KV_TUNER_NBITS_KEY=4
export KV_TUNER_NBITS_VALUE=4
export KV_TUNER_RESIDUAL_LENGTH=256
./start_prefill.sh
```

**Decode (低延迟):**
```bash
export KV_TUNER_NBITS_KEY=8
export KV_TUNER_NBITS_VALUE=8
export KV_TUNER_RESIDUAL_LENGTH=32
./start_decode.sh
```

### 使用不同 KV Cache 量化

```bash
# FP8 E5M2 (推荐)
export KV_CACHE_DTYPE=fp8_e5m2

# FP8 E4M3 (更高精度)
export KV_CACHE_DTYPE=fp8_e4m3

# FP4 (实验性，最大压缩)
export KV_CACHE_DTYPE=fp4_e2m1

# 禁用 KV Cache 量化
export KV_CACHE_DTYPE=none
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `deploy_to_nodes.sh` | 部署脚本到所有节点 |
| `start_prefill.sh` | 启动 Prefill 服务 |
| `start_decode.sh` | 启动 Decode 服务 |
| `start_router.sh` | 启动 Router |
| `run_validation.sh` | 运行 Bash 验证测试 |
| `validate_pd_quant.py` | Python 验证脚本（更详细） |
| `check_status.sh` | 检查服务状态 |
| `README.md` | 完整文档 |
| `DEPLOYMENT_GUIDE.md` | 部署指南 |
| `QUICK_START.md` | 本文件 |

## 故障排除

**问题**: SSH 连接失败
```bash
# 测试连接
sshpass -p luhang222 ssh ubuntu@10.60.6.75 "echo test"
```

**问题**: 服务启动超时
```bash
# 检查 GPU
nvidia-smi

# 检查端口
ss -tlnp | grep 30000

# 查看日志
cat /tmp/sglang_logs/prefill_*.log
```

**问题**: P/D 通信失败
```bash
# 测试节点间网络
ping 10.60.19.152

# 检查防火墙
sudo ufw status
```

## 预期结果

成功的验证应该显示：
- ✓ 所有服务健康
- ✓ P/D 模式配置正确
- ✓ 量化已启用
- ✓ 推理测试通过
- ✓ 性能指标合理

## 下一步

1. 阅读 `DEPLOYMENT_GUIDE.md` 了解详细配置
2. 阅读 `README.md` 了解完整验证流程
3. 运行验证并查看报告
4. 根据结果调整量化配置

## 支持

遇到问题？查看日志文件：
```bash
ls -la /tmp/sglang_logs/
```

或查阅：
- [SGLang 文档](https://docs.sglang.io)
- [KVTuner 使用说明](../../KVTUNER_USAGE.md)
