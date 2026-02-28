# P/D 分离与量化结合验证指南

## 概述

本验证方案在 4 台机器上测试 SGLang KVTuner 量化与 Prefill/Decode 分离模式的结合使用。

## 🖥️ 硬件环境

| 机器 | IP | 角色 | GPU | TP 大小 |
|------|-----|------|-----|--------|
| Node 1 | 10.60.6.75 | Prefill + Router | 2×RTX 3090 | 2 |
| Node 2 | 10.60.19.152 | Decode | 2×RTX 3090 | 2 |
| Node 3 | 10.60.9.62 | Prefill | 2×RTX 3090 | 2 |
| Node 4 | 10.60.176.217 | Decode | 2×RTX 3090 | 2 |

**网络**: TCP (无 IB/RoCE)  
**传输后端**: NIXL (UCX over TCP)

## 📁 模型

| 模型 | 路径 | 状态 |
|------|------|------|
| Qwen2.5-7B | `/data/Qwen/Qwen2.5-7B` | ✅ 已确认 |
| Qwen3-32B | `/data/Qwen/Qwen3-32B` | ✅ 已确认 |

**默认使用**: Qwen2.5-7B (显存友好，适合测试)

## 验证目标

1. ✅ 验证 KVTuner 量化在 P/D 分离模式下的功能正确性
2. ✅ 验证 Prefill 和 Decode 可以使用不同的量化配置
3. ✅ 验证 KV Cache 传输在量化模式下的正确性
4. ✅ 性能基准测试（吞吐、延迟、内存使用）

## 🚀 快速开始

### 方式 A: 一键部署（推荐）

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation
./deploy_and_run.sh
```

### 方式 B: 手动部署

详见 `QUICK_START.md`

## 验证测试项

### 测试 1: 基本连通性测试
- 验证 Prefill 和 Decode 服务可以正常通信
- 验证 KV Cache 传输正常

### 测试 2: 量化功能验证
- 验证 KVTuner 量化在 Prefill 模式下正常工作
- 验证 KVTuner 量化在 Decode 模式下正常工作
- 验证不同量化配置（4-bit, 8-bit）的正确性

### 测试 3: 模式感知量化验证
- 验证 Prefill 模式使用预配置的量化参数
- 验证 Decode 模式使用预配置的量化参数
- 验证残差缓存管理正确

### 测试 4: 性能基准测试
- 测量 Prefill 吞吐量
- 测量 Decode 延迟
- 测量内存使用率
- 与未量化基线对比

## 预期结果

| 指标 | 预期值 |
|------|--------|
| 功能正确性 | 所有测试通过 |
| 量化压缩比 | 4-bit: ~4x, 8-bit: ~2x |
| 精度损失 | <1% perplexity |
| Prefill 吞吐 | 与基线相当或更好 |
| Decode 延迟 | 增加 <20% |

## 故障排除

详见 `TROUBLESHOOTING.md`
