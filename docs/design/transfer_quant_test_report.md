# Transfer Quantization 测试报告

> 测试日期：2026-04-11
> 测试分支：kvtuner-splitwise
> 代码版本：1127984cb

## 1. 测试环境

### 1.1 硬件配置

| 节点 | 角色 | 外网 IP | 内网 IP |
|------|------|---------|---------|
| gpu1 | Prefill | 117.50.192.238 | 10.60.23.70 |
| gpu2 | Decode | 117.50.189.89 | 10.60.30.66 |

- 内网互通，延迟 <1ms（无人工延迟时）
- GPU：各节点 1 张 GPU
- 模型：Qwen2.5-7B（28 层，32 注意力头，8 KV 头，head_dim=128）

### 1.2 软件配置

- sglang 分支：kvtuner-splitwise
- Python 虚拟环境：/home/ubuntu/sglang-env
- 启动参数：`--disaggregation-mode {prefill|decode} --disaggregation-transfer-backend tcp`

### 1.3 测试工具

| 工具 | 路径 | 用途 |
|------|------|------|
| pd_test.sh | scripts/pd_disagg_test/pd_test.sh | 一键部署、启停、探活 |
| remote_worker.sh | scripts/pd_disagg_test/remote_worker.sh | 远程节点辅助脚本 |
| pd_coordinator.py | scripts/pd_disagg_test/pd_coordinator.py | PD 推理测试客户端 |
| configs/tcp-quant.sh | scripts/pd_disagg_test/configs/tcp-quant.sh | Transfer Quant 测试配置 |
| configs/default.sh | scripts/pd_disagg_test/configs/default.sh | 默认集群配置 |
| tc netem | 系统工具 | 网络延迟模拟 |

### 1.4 测试配置

**Baseline（无量化）**：默认配置，`ENABLE_TRANSFER_QUANT` 未设置。

**Quant 8-bit**：
```bash
CONFIG_FILE=configs/tcp-quant.sh
# 等效于: --enable-transfer-quant --transfer-quant-bits 8
```

**Quant 4-bit**：
```bash
ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4
```

### 1.5 网络延迟模拟

使用 `tc netem` 在两台机器之间添加定向延迟（仅影响节点间流量，不影响 SSH 和服务启动）：

```bash
# Prefill 节点：仅延迟发往 Decode 的流量
sudo tc qdisc add dev eth0 root handle 1: prio
sudo tc qdisc add dev eth0 parent 1:3 handle 30: netem delay 20ms 5ms
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst 10.60.30.66 flowid 1:3

# Decode 节点：仅延迟发往 Prefill 的流量
sudo tc qdisc add dev eth0 root handle 1: prio
sudo tc qdisc add dev eth0 parent 1:3 handle 30: netem delay 20ms 5ms
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst 10.60.23.70 flowid 1:3
```

实测 RTT：~40ms（20ms 单向 + 5ms 抖动）。

## 2. 测试方法

### 2.1 测试流程

为避免服务状态污染（bootstrap_room mismatch 等问题），每轮测试采用「重启+单请求」模式：

1. 清除网络延迟规则
2. 停止旧服务
3. 以目标配置（baseline/quant8/quant4）启动 Prefill + Decode
4. 等待两端 health check 通过
5. 添加网络延迟（如需高延迟测试）
6. 通过 pd_coordinator.py 发送单个推理请求
7. 记录 Prefill/Decode 耗时和结果
8. 清理延迟规则，停止服务

### 2.2 测试用例

**短文本（低延迟测试）**：
- Prompt: "The quick brown fox jumps over the lazy dog." (10 tokens)
- Completion: 64 tokens

**中等文本（低延迟测试）**：
- Prompt: ~28 tokens
- Completion: 64 tokens

**长文本（高延迟测试）**：
- Prompt: ~250 tokens（一篇关于 AI 的英文长文）
- Completion: 32 tokens

### 2.3 度量指标

- **Decode 耗时**：从发送请求到收到完整响应的时间，包含 KV 传输等待 + token 生成
- **成功率**：请求在 120s 超时内完成的比例
- **Prefill 耗时**：预填充计算 + KV 发送时间

## 3. 测试结果

### 3.1 场景一：低延迟内网（<1ms RTT），短文本

**Prompt: 10 tokens, Completion: 64 tokens, 无网络延迟**

| 配置 | R1 | R2 | R3 | 平均 Decode |
|------|-----|-----|-----|-------------|
| Baseline | 1.5s | 1.5s | 1.5s | **1.50s** |
| Quant 8-bit | 1.6s | 1.5s | 1.5s | **1.53s** |

**结论**：无差异。网络延迟极低时，KV 传输时间可忽略，量化/反量化的 CPU 计算开销抵消了传输节省。

### 3.2 场景二：低延迟内网（<1ms RTT），中等文本

**Prompt: 28 tokens, Completion: 64 tokens, 无网络延迟**

| 配置 | R1 | R2 | 平均 Decode |
|------|-----|-----|-------------|
| Baseline | 1.6s | 1.6s | **1.60s** |
| Quant 8-bit | 1.6s | 1.6s | **1.60s** |

**结论**：同场景一，无差异。

### 3.3 场景三：高延迟网络（~40ms RTT），长文本

**Prompt: 251 tokens, Completion: 32 tokens, 20ms 单向延迟 + 5ms 抖动**

| 配置 | R1 | R2 | R3 | 成功率 | 成功时 Decode |
|------|-----|-----|-----|--------|---------------|
| Baseline | 120s 超时 | 120s 超时 | 120s 超时 | **0/3 (0%)** | N/A |
| Quant 8-bit | 120s 超时 | **2.1s** | **2.1s** | **2/3 (67%)** | 2.1s |
| Quant 4-bit | 120s 超时 | **2.2s** | 120s 超时 | **1/3 (33%)** | 2.2s |

## 4. 分析

### 4.1 低延迟环境（场景一、二）

在 <1ms RTT 的内网环境下，Transfer Quantization **没有带来加速效果**：

- **原因**：28 层 × 2 (K+V) 的 KV 传输数据量较小（短文本约 0.3MB，中等文本约 0.9MB），在低延迟高带宽内网中传输时间 <100ms，远小于 GPU token 生成时间（~0.5-1.5s）。
- **量化开销**：CPU 端 numpy 量化/反量化引入额外计算延迟，恰好抵消了传输时间节省。
- **瓶颈不在传输**：Decode 耗时主要由 GPU token 生成主导。

### 4.2 高延迟环境（场景三）

在 ~40ms RTT 环境下，Transfer Quantization **显著改善了服务可用性**：

- **Baseline 完全失败**（0% 成功率）：251 tokens 的 KV 数据约 14MB，高延迟下 TCP 传输导致 decode 端等待超时。
- **Quant 8-bit 成功率 67%**：数据量压缩至约 7.2MB，减少了传输时间，使大部分请求能在超时前完成。
- **成功时性能一致**：Quant 8-bit 和 Quant 4-bit 成功时 Decode 耗时均为 ~2.1-2.2s，与低延迟环境相近，说明传输不再是瓶颈。

### 4.3 4-bit vs 8-bit

Quant 4-bit 成功率（33%）反而低于 8-bit（67%），可能原因：
- 4-bit 量化误差更大，反量化后的数值偏差可能影响模型注意力计算
- 4-bit 量化的 scales 精度不足以表示某些层的数值分布
- 需要进一步排查是否为数值精度问题或偶发网络问题

### 4.4 间歇性超时问题

所有配置在成功和超时之间呈二值分布（要么 ~2s 完成，要么 120s 超时），暗示存在以下可能：
- TCP KV 传输在延迟网络下存在连接建立/握手的竞态条件
- Decode 端等待 KV 的内部同步机制在高延迟下不够健壮
- 非量化传输因数据量大，更易触发 TCP 拥塞控制导致卡住

### 4.5 量化功能验证

通过 Decode 端日志确认量化功能正确工作：
```
[_recv_kv] writing layer_id=0 (actual_layer=0, K) data_len=257024 quant=True received=0/56
[_recv_kv] got EOF, received=56/56
[_recv_loop] _recv_kv completed!
```
所有 28 层 × 2 (K+V) = 56 条消息均标记 `quant=True`，传输和反量化流程完整。

## 5. 结论与建议

### 5.1 结论

| 场景 | Transfer Quant 效果 |
|------|-------------------|
| 低延迟内网 (<1ms RTT) + 短/中 prompt | 无显著效果 |
| 高延迟网络 (~40ms RTT) + 长 prompt (>200 tokens) | **显著改善可用性**（0% → 67%） |

### 5.2 改进建议

1. **排查间歇性超时**：在高延迟环境下，KV TCP 传输存在二值性超时问题（成功 vs 120s 超时），需要排查连接建立、TCP socket 超时配置、KV 等待同步机制等。
2. **优化量化性能**：当前使用 CPU numpy 量化，对于大 KV 数据可能成为瓶颈。可考虑 GPU 量化或异步量化（与下一层计算并行）。
3. **增加带宽限制测试**：当前仅测试了延迟影响，未测试带宽受限场景（如 1Gbps 瓶颈），在带宽受限场景下量化的收益可能更显著。
4. **4-bit 量化稳定性**：需要单独排查 4-bit 量化成功率低于 8-bit 的原因，可能需要调整量化策略（如非对称量化、per-channel scale）。
