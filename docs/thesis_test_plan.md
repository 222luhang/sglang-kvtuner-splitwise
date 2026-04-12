# 硕士毕业论文实验测试方案

> 论文方向：基于逐层自适应量化的 PD 分离架构 KV Cache 传输加速
> 项目分支：kvtuner-splitwise
> 编写日期：2026-04-12

---

## 一、实验总体设计

### 1.1 研究问题与实验目标

本论文的核心研究问题是：**在 Prefill-Decode 分离推理架构中，如何通过 KV Cache 传输量化降低网络传输瓶颈，同时保持模型输出质量？**

实验需要回答以下关键问题：

| 编号 | 研究问题 | 对应实验 |
|------|---------|---------|
| RQ1 | 传输量化能否有效降低 TTFT（Time To First Token）？ | 实验 3.1 |
| RQ2 | 不同量化位宽（4-bit/8-bit）对模型输出质量的影响有多大？ | 实验 3.2 |
| RQ3 | 逐层自适应量化（混合精度）相比统一量化有何优势？ | 实验 3.3 |
| RQ4 | GPU 加速量化相比 CPU 量化的性能提升有多少？ | 实验 3.4 |
| RQ5 | 在不同网络条件下，传输量化的收益如何变化？ | 实验 3.5 |
| RQ6 | 系统在高并发场景下的稳定性和可扩展性如何？ | 实验 3.6 |

### 1.2 实验变量定义

**自变量（Independent Variables）**：
- 量化位宽：无量化（BF16）、8-bit、4-bit
- 量化策略：统一量化、逐层自适应量化（混合精度）
- 量化执行位置：CPU（numpy）、GPU（torch CUDA）
- 网络延迟：<1ms（内网）、10ms、20ms、40ms、80ms
- 网络带宽：无限制、10Gbps、1Gbps、100Mbps
- 输入长度：短（1-10 tokens）、中（28-60 tokens）、长（128-512 tokens）、超长（1024+ tokens）
- 输出长度：短（16 tokens）、中（64 tokens）、长（128-256 tokens）
- 并发请求数：1、2、4、8、16、32
- 模型规模：Qwen2.5-7B（28层）、Llama-3-8B（32层）

**因变量（Dependent Variables）**：
- TTFT（Time To First Token）
- 端到端延迟（End-to-End Latency）
- 吞吐量（Tokens/s）
- 请求成功率
- 模型输出质量指标（见 3.2 节）
- GPU 显存占用
- CPU 利用率
- 网络传输数据量

**控制变量（Controlled Variables）**：
- 硬件配置固定
- 模型权重一致
- 采样参数固定（temperature=0, greedy decoding）
- group_size=64（量化分组大小）

---

## 二、实验环境

### 2.1 硬件环境

| 节点 | 角色 | IP | GPU | 内存 |
|------|------|-----|-----|------|
| gpu1 | Prefill | 10.60.23.70 | 1× GPU | 待补充 |
| gpu2 | Decode | 10.60.30.66 | 1× GPU | 待补充 |

> **建议补充**：GPU 型号（如 A100-80G / A10 / L40S）、CPU 型号与核数、系统内存、NVLink/PCIe 版本、网卡型号与带宽。这些信息对论文的可复现性至关重要。

### 2.2 软件环境

| 组件 | 版本 |
|------|------|
| OS | Ubuntu (Linux 6.8.0-31-generic) |
| Python | 3.x（待确认） |
| PyTorch | 待确认 |
| CUDA | 待确认 |
| SGLang | kvtuner-splitwise 分支 |
| 模型 | Qwen2.5-7B（28层, 32 attn heads, 8 KV heads, head_dim=128） |

### 2.3 网络模拟工具

使用 Linux `tc netem` 模拟不同网络条件：

```bash
# 添加延迟（单向 Xms + Yms 抖动）
sudo tc qdisc add dev eth0 root handle 1: prio
sudo tc qdisc add dev eth0 parent 1:3 handle 30: netem delay ${X}ms ${Y}ms
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 \
    match ip dst ${TARGET_IP} flowid 1:3

# 添加带宽限制
sudo tc qdisc add dev eth0 parent 1:3 handle 30: netem delay ${X}ms rate ${BW}mbit

# 清除规则
sudo tc qdisc del dev eth0 root
```

### 2.4 测试工具

| 工具 | 路径 | 用途 |
|------|------|------|
| pd_test.sh | scripts/pd_disagg_test/pd_test.sh | 一键部署、启停、探活 |
| pd_coordinator.py | scripts/pd_disagg_test/pd_coordinator.py | PD 推理测试客户端 |
| remote_worker.sh | scripts/pd_disagg_test/remote_worker.sh | 远程节点辅助脚本 |
| tc netem | 系统工具 | 网络延迟/带宽模拟 |
| nvidia-smi | 系统工具 | GPU 监控 |
| sar / iostat | 系统工具 | CPU/IO 监控 |

---

## 三、实验方案详细设计

### 3.1 实验一：传输量化对 TTFT 的影响（核心实验）

**目的**：验证传输量化能否有效降低 TTFT，量化不同位宽的加速效果。

**配置矩阵**：

| 配置名 | 量化 | 位宽 | 启动参数 |
|--------|------|------|---------|
| baseline | 无 | BF16 | 默认 |
| quant-8bit | 统一 | 8-bit | `--enable-transfer-quant --transfer-quant-bits 8` |
| quant-4bit | 统一 | 4-bit | `--enable-transfer-quant --transfer-quant-bits 4` |
| mixed | 逐层 | 8/4混合 | `--enable-transfer-quant --kvtuner-layer-config <json>` |

**输入矩阵**（每个配置 × 每个输入 × 重复 5 次）：

| 输入类别 | Prompt Tokens | Max New Tokens | 预估 KV 大小 |
|---------|---------------|----------------|-------------|
| 极短 | 1 | 32 | ~0.03 MB |
| 短 | 10 | 64 | ~0.3 MB |
| 中 | 60 | 64 | ~1.8 MB |
| 长 | 256 | 32 | ~7.7 MB |
| 超长 | 512 | 32 | ~15.4 MB |
| 极长 | 1024 | 16 | ~30.7 MB |

> KV 大小估算：`num_layers × 2(K+V) × num_kv_heads × head_dim × seq_len × 2(BF16)`
> = 28 × 2 × 8 × 128 × seq_len × 2 bytes = 114,688 × seq_len bytes

**度量指标**：
- TTFT（从 Decode 端发送请求到收到第一个 token 的时间）
- Prefill 计算时间
- KV 传输时间（从 Prefill 日志中提取）
- 量化/反量化耗时
- 端到端延迟
- 实际传输数据量（bytes）

**执行步骤**：
1. 以目标配置启动 Prefill + Decode 服务
2. 等待 health check 通过
3. 发送单个推理请求，记录各阶段耗时
4. 停止服务，清理状态
5. 重复步骤 1-4 共 5 次（避免服务状态污染）
6. 切换到下一个配置

**数据记录格式**：

```csv
config,input_type,prompt_tokens,max_new_tokens,run_id,ttft_ms,prefill_ms,transfer_ms,quant_ms,dequant_ms,e2e_ms,transfer_bytes,success
baseline,short,10,64,1,1500,200,50,0,0,1500,327680,true
quant-8bit,short,10,64,1,1530,200,30,5,5,1530,168960,true
```

**预期结果**：
- 短文本（<60 tokens）：量化无显著加速（传输不是瓶颈）
- 长文本（>256 tokens）：8-bit 量化 TTFT 降低 20-40%，4-bit 降低 40-60%
- 混合精度接近 4-bit 的加速效果

**统计分析**：
- 对每组 5 次重复取均值和标准差
- 使用配对 t 检验（paired t-test）比较 baseline vs 各量化配置的 TTFT 差异
- 显著性水平 α = 0.05

---

### 3.2 实验二：量化对模型输出质量的影响

**目的**：评估不同量化位宽对模型推理质量的影响，确定质量-速度的 Pareto 最优点。

#### 3.2.1 数值精度测试（单元级）

使用已有单元测试框架 `test/unit/test_transfer_quant.py` 扩展：

| 测试项 | 指标 | 8-bit 阈值 | 4-bit 阈值 |
|--------|------|-----------|-----------|
| 量化-反量化往返误差 | MSE | < 1e-4 | < 1e-2 |
| 余弦相似度 | Cosine Sim | > 0.999 | > 0.98 |
| 最大绝对误差 | Max Abs Error | < 0.05 | < 0.5 |
| 相对误差分布 | P95/P99 | 报告 | 报告 |

**测试数据**：
- 随机正态分布张量（模拟典型 KV 值）
- 从实际模型推理中截取的真实 KV cache 数据
- 极端值张量（接近 FP16 上下界）
- 稀疏张量（大量零值）

#### 3.2.2 端到端质量评估（系统级）

**评估基准**：

| 基准 | 数据集 | 指标 | 样本数 |
|------|--------|------|--------|
| 数学推理 | GSM8K | Accuracy (%) | 200+ |
| 常识推理 | HellaSwag | Accuracy (%) | 200+ |
| 语言理解 | MMLU (5-shot) | Accuracy (%) | 200+ |
| 代码生成 | HumanEval | Pass@1 (%) | 164 |
| 文本生成质量 | MT-Bench | Score (1-10) | 80 |

**配置对比**：

| 配置 | 说明 |
|------|------|
| reference | 单机推理（无 PD 分离），作为质量上界 |
| pd-baseline | PD 分离 + BF16 传输（无量化） |
| pd-quant-8bit | PD 分离 + 统一 8-bit 传输量化 |
| pd-quant-4bit | PD 分离 + 统一 4-bit 传输量化 |
| pd-mixed | PD 分离 + 逐层混合精度传输量化 |

**执行方法**：
```bash
# 以 GSM8K 为例
# 1. 启动 PD 服务（指定配置）
./pd_test.sh start --config configs/tcp-quant.sh

# 2. 运行评估脚本（需要开发）
python3 eval/run_benchmark.py \
    --benchmark gsm8k \
    --prefill-host 10.60.23.70 --prefill-port 30000 \
    --decode-host 10.60.30.66 --decode-port 30001 \
    --num-samples 200 \
    --output results/gsm8k_quant8.json
```

**质量损失容忍度**：
- 8-bit 量化：准确率下降 < 1%（相对于 pd-baseline）
- 4-bit 量化：准确率下降 < 3%
- 混合精度：准确率下降 < 1.5%

---

### 3.3 实验三：逐层自适应量化策略对比（消融实验）

**目的**：验证逐层自适应量化（混合精度）相比统一量化的优势，证明"敏感层高精度 + 非敏感层低精度"策略的有效性。

#### 3.3.1 层敏感度分析

使用离线校准工具 `scripts/pd_quant_validation/kvtuner_offline_calib.py`：

```bash
python3 kvtuner_offline_calib.py \
    --model /data/Qwen/Qwen2.5-7B \
    --dataset wikitext2 \
    --num-samples 128 \
    --output layer_sensitivity.json
```

**输出**：每层在不同量化位宽下的 perplexity 变化（ΔPPL），用于确定层敏感度排序。

#### 3.3.2 量化策略对比

| 策略 | 配置 | 平均位宽 | 预期压缩比 |
|------|------|---------|-----------|
| uniform-8bit | 全部 28 层 8-bit | 8.0 | ~1.94x |
| uniform-4bit | 全部 28 层 4-bit | 4.0 | ~3.77x |
| mixed-A | 前/后 5 层 8-bit + 中间 18 层 4-bit | 5.14 | ~3.0x |
| mixed-B | 敏感度 Top-5 层 8-bit + 其余 4-bit | ~5.14 | ~3.0x |
| mixed-C | 敏感度 Top-10 层 8-bit + 其余 4-bit | ~6.0 | ~2.5x |
| mixed-D | 3 级量化：Top-5 层 8-bit + 中间 13 层 4-bit + 底 10 层 2-bit | ~4.0 | ~3.5x |

**评估维度**：
- 各策略在 GSM8K / MMLU 上的准确率
- 各策略的实际压缩比和传输时间
- 绘制 **质量-压缩比 Pareto 曲线**

**关键图表**：
1. 层敏感度热力图（x=层编号, y=量化位宽, color=ΔPPL）
2. Pareto 曲线（x=压缩比, y=准确率）
3. 各策略的 TTFT 对比柱状图

---

### 3.4 实验四：GPU 加速量化 vs CPU 量化性能对比

**目的**：验证将量化/反量化从 CPU（numpy）迁移到 GPU（torch CUDA）的性能提升。

#### 3.4.1 微基准测试（Microbenchmark）

对不同大小的张量，分别测量 CPU 和 GPU 路径的量化/反量化耗时：

| 张量大小 | 对应场景 | 元素数 |
|---------|---------|--------|
| 0.1 MB | 极短 prompt (1 token, 单层) | 51,200 |
| 1 MB | 短 prompt (10 tokens, 单层) | 512,000 |
| 10 MB | 中等 prompt (100 tokens, 单层) | 5,120,000 |
| 50 MB | 长 prompt (500 tokens, 单层) | 25,600,000 |

**测量项**：

| 路径 | 量化耗时 | 反量化耗时 | DtoH/HtoD 耗时 | 总耗时 |
|------|---------|-----------|----------------|--------|
| CPU (numpy) | ✓ | ✓ | BF16 全量 | ✓ |
| GPU (torch) | ✓ | ✓ | 压缩后数据 | ✓ |

**测试代码框架**：
```python
import torch, time
from sglang.srt.disaggregation.tcp.transfer_quant import (
    quantize_for_transfer, dequantize_from_transfer,
    quantize_on_gpu, dequantize_on_gpu,
)

def bench_quantize(tensor_size, nbits, num_warmup=5, num_runs=50):
    # GPU 路径
    t_gpu = torch.randn(tensor_size, dtype=torch.bfloat16, device='cuda')
    for _ in range(num_warmup):
        quantize_on_gpu(t_gpu, nbits=nbits)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        packed = quantize_on_gpu(t_gpu, nbits=nbits)
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / num_runs

    # CPU 路径
    raw_bytes = t_gpu.cpu().numpy().tobytes()
    for _ in range(num_warmup):
        quantize_for_transfer(raw_bytes, 2, True, nbits=nbits)

    t0 = time.perf_counter()
    for _ in range(num_runs):
        quantize_for_transfer(raw_bytes, 2, True, nbits=nbits)
    cpu_time = (time.perf_counter() - t0) / num_runs

    return gpu_time, cpu_time
```

#### 3.4.2 端到端对比

在实际 PD 推理中对比 CPU 和 GPU 量化路径的 TTFT：

| 配置 | 量化位置 | 数据流 |
|------|---------|--------|
| cpu-quant-8bit | CPU | GPU→CPU(BF16) → CPU量化 → TCP(int8) → CPU反量化 → CPU→GPU(BF16) |
| gpu-quant-8bit | GPU | GPU量化 → GPU→CPU(int8) → TCP(int8) → CPU→GPU(int8) → GPU反量化 |

> 注意：需要在代码中添加开关来切换 CPU/GPU 量化路径，或通过环境变量控制。

**预期结果**：
- GPU 量化路径在长文本场景下 TTFT 降低 10-30%（减少 PCIe 传输量 + 消除 CPU numpy 开销）
- 短文本场景差异较小（量化数据量小，PCIe 不是瓶颈）

---

### 3.5 实验五：网络条件对传输量化收益的影响

**目的**：系统性地评估不同网络延迟和带宽条件下传输量化的收益变化，找到量化"收益拐点"。

#### 3.5.1 延迟敏感性测试

固定带宽（无限制），变化延迟：

| 单向延迟 | RTT | 模拟场景 |
|---------|-----|---------|
| 0ms | <1ms | 同机房内网 |
| 5ms | ~10ms | 同城跨机房 |
| 10ms | ~20ms | 同区域跨城市 |
| 20ms | ~40ms | 跨区域 |
| 40ms | ~80ms | 跨国 |

每个延迟 × 3 种配置（baseline/quant-8bit/quant-4bit）× 3 种输入长度（短/中/长）× 5 次重复。

**总测试数 = 5 × 3 × 3 × 5 = 225 次**

#### 3.5.2 带宽敏感性测试

固定延迟（<1ms），变化带宽：

| 带宽限制 | 模拟场景 |
|---------|---------|
| 无限制 | 高速内网（25Gbps+） |
| 10 Gbps | 标准数据中心 |
| 1 Gbps | 普通以太网 |
| 100 Mbps | 受限网络 |

每个带宽 × 3 种配置 × 3 种输入长度 × 5 次重复。

**总测试数 = 4 × 3 × 3 × 5 = 180 次**

#### 3.5.3 延迟-带宽联合测试

选取典型组合：

| 场景 | 延迟 | 带宽 | 现实对应 |
|------|------|------|---------|
| 理想 | <1ms | 无限制 | 同机房 RDMA |
| 标准 | 5ms | 10Gbps | 同城数据中心 |
| 受限 | 20ms | 1Gbps | 跨区域普通网络 |
| 恶劣 | 40ms | 100Mbps | 边缘计算场景 |

**关键图表**：
1. 延迟-TTFT 曲线（不同量化配置的折线图）
2. 带宽-TTFT 曲线
3. 热力图（x=延迟, y=带宽, color=量化加速比）
4. 量化收益拐点分析（在什么网络条件下量化开始产生正收益）

---

### 3.6 实验六：并发与稳定性测试

**目的**：评估系统在高并发场景下的稳定性、吞吐量和资源利用率。

#### 3.6.1 并发吞吐量测试

| 并发数 | 请求总数 | 输入 | 输出 |
|--------|---------|------|------|
| 1 | 20 | 60 tokens | 64 tokens |
| 2 | 40 | 60 tokens | 64 tokens |
| 4 | 80 | 60 tokens | 64 tokens |
| 8 | 160 | 60 tokens | 64 tokens |
| 16 | 320 | 60 tokens | 64 tokens |
| 32 | 640 | 60 tokens | 64 tokens |

**度量指标**：
- 吞吐量（requests/s, tokens/s）
- P50 / P90 / P99 延迟
- 请求成功率
- GPU 利用率（nvidia-smi 采样）
- GPU 显存峰值
- CPU 利用率

**执行方法**：
```bash
# 需要扩展 pd_coordinator.py 支持持续发送模式
python3 pd_coordinator.py \
    --num-requests ${CONCURRENCY} \
    --total-requests ${TOTAL} \
    --input-tokens 60 \
    --max-new-tokens 64 \
    --output results/concurrency_${CONCURRENCY}.json
```

#### 3.6.2 长时间稳定性测试（Soak Test）

持续运行 30 分钟，每 5 秒发送一个请求：

| 配置 | 持续时间 | 请求间隔 | 预计请求数 |
|------|---------|---------|-----------|
| baseline | 30 min | 5s | ~360 |
| quant-8bit | 30 min | 5s | ~360 |
| quant-4bit | 30 min | 5s | ~360 |

**监控项**：
- 请求成功率随时间的变化
- 延迟随时间的变化（是否有退化）
- GPU 显存是否泄漏（持续增长）
- 是否出现 scheduler segfault（已知 P0 问题）
- 是否出现 bootstrap_room mismatch（已知 P1 问题）

#### 3.6.3 故障恢复测试

| 故障场景 | 注入方式 | 预期行为 |
|---------|---------|---------|
| TCP 连接中断 | `iptables -A OUTPUT -p tcp --dport X -j DROP` | 请求超时，后续请求正常 |
| Prefill 节点重启 | kill + restart | Decode 端等待超时后恢复 |
| 网络抖动 | `tc netem delay 50ms 30ms distribution normal` | 部分请求延迟增大，不应崩溃 |

---

## 四、单元测试与集成测试方案

### 4.1 单元测试（已有 + 需扩展）

#### 4.1.1 量化正确性测试（test_transfer_quant.py）

**已有测试**：
- [x] BF16 往返转换
- [x] 8-bit / 4-bit 量化-反量化往返
- [x] 非对齐长度处理
- [x] 零值输入
- [x] 大值输入（FP16 上下界）
- [x] 压缩比验证
- [x] 余弦相似度

**需新增测试**：

| 测试项 | 描述 | 优先级 |
|--------|------|--------|
| GPU 量化往返 | `quantize_on_gpu` → `dequantize_on_gpu` 一致性 | P0 |
| GPU-CPU 互操作 | `quantize_on_gpu` → `dequantize_from_transfer` 和反向 | P0 |
| 不同 group_size | group_size = 32, 64, 128 的正确性 | P1 |
| 大张量 | 100MB+ 张量的量化正确性和内存安全 | P1 |
| 数值分布覆盖 | 均匀分布、正态分布、长尾分布、双峰分布 | P1 |
| 边界条件 | 单元素、group_size 个元素、group_size+1 个元素 | P1 |
| 多 dtype | FP16 和 BF16 分别测试 | P1 |
| 确定性 | 相同输入多次量化结果一致 | P2 |

#### 4.1.2 TCP 协议测试（test_tcp_kv_transfer.py）

**已有测试**：
- [x] 发送-接收往返
- [x] 空 payload（EOF 标记）
- [x] 多层顺序传输
- [x] 连接等待超时
- [x] Pipeline 模式标志
- [x] Backend 注册

**需新增测试**：

| 测试项 | 描述 | 优先级 |
|--------|------|--------|
| 量化标志传递 | `_MSG_QUANT_FLAG` 在 layer_id 中正确设置和解析 | P0 |
| 量化+传输端到端 | 发送端量化 → TCP → 接收端反量化，数据一致 | P0 |
| 大 payload | 单层 50MB+ 数据的传输正确性 | P1 |
| 并发连接 | 多个 sender 同时连接同一 receiver | P1 |
| 连接断开恢复 | 传输中途断开连接的行为 | P2 |

### 4.2 集成测试

#### 4.2.1 PD 端到端推理测试

使用 `pd_test.sh` + `pd_coordinator.py`：

```bash
# 完整 batch 测试（9 种输入/输出长度组合）
./pd_test.sh full --config configs/tcp-quant.sh

# 验证所有 9 个 batch test 通过
# 预期：至少 8/9 PASS（long_long 可能因 scheduler 问题超时）
```

#### 4.2.2 量化功能验证测试

**验证清单**：

| 验证项 | 方法 | 预期 |
|--------|------|------|
| 量化标志生效 | 检查 Decode 日志中 `quant=True` | 所有 56 条消息均标记 |
| 压缩比正确 | 对比传输数据量 vs BF16 原始大小 | 8-bit ~1.94x, 4-bit ~3.77x |
| 输出一致性 | 对比量化 vs baseline 的输出文本 | 高度相似（非完全一致） |
| 逐层配置生效 | 检查日志中各层的量化位宽 | 与配置文件一致 |

---

## 五、评估脚本开发计划

### 5.1 需要开发的脚本

| 脚本 | 功能 | 优先级 |
|------|------|--------|
| `eval/run_benchmark.py` | 通过 PD 架构运行标准评估基准 | P0 |
| `eval/bench_ttft.py` | TTFT 精确测量（含各阶段耗时分解） | P0 |
| `eval/bench_quant_micro.py` | 量化微基准测试（CPU vs GPU） | P0 |
| `eval/bench_throughput.py` | 并发吞吐量测试 | P1 |
| `eval/bench_network.py` | 网络条件自动化测试（自动设置 tc netem） | P1 |
| `eval/soak_test.py` | 长时间稳定性测试 | P1 |
| `eval/collect_metrics.py` | GPU/CPU 资源监控采集 | P2 |
| `eval/plot_results.py` | 结果可视化（生成论文图表） | P2 |

### 5.2 bench_ttft.py 设计

```python
"""
TTFT 精确测量脚本

对每个请求记录：
- t_request: 请求发送时间
- t_prefill_start: Prefill 开始计算（从 Prefill 日志）
- t_prefill_done: Prefill 计算完成
- t_transfer_start: KV 传输开始
- t_transfer_done: KV 传输完成
- t_decode_start: Decode 开始生成
- t_first_token: 第一个 token 生成时间

TTFT = t_first_token - t_request
Transfer Time = t_transfer_done - t_transfer_start
Quant Overhead = (量化耗时 + 反量化耗时)
"""

# 需要在 conn.py 中添加时间戳日志：
# [TIMING] layer={layer_id} quant_ms={quant_time} send_ms={send_time}
# [TIMING] layer={layer_id} recv_ms={recv_time} dequant_ms={dequant_time}
```

### 5.3 run_benchmark.py 设计

```python
"""
通过 PD 架构运行标准评估基准

支持的基准：
- gsm8k: 数学推理（accuracy）
- mmlu: 语言理解（5-shot accuracy）
- hellaswag: 常识推理（accuracy）
- humaneval: 代码生成（pass@1）

用法：
python3 eval/run_benchmark.py \
    --benchmark gsm8k \
    --prefill-host 10.60.23.70 --prefill-port 30000 \
    --decode-host 10.60.30.66 --decode-port 30001 \
    --num-samples 200 \
    --output results/gsm8k_quant8.json
"""
```

---

## 六、实验执行计划与时间线

### 6.1 执行顺序（按依赖关系排列）

| 阶段 | 内容 | 前置条件 | 预计工作量 |
|------|------|---------|-----------|
| **Phase 0** | 环境准备与工具开发 | — | 3-5 天 |
| 0.1 | 记录完整硬件/软件环境信息 | — | 0.5 天 |
| 0.2 | 开发 bench_ttft.py（含日志时间戳） | — | 1-2 天 |
| 0.3 | 开发 run_benchmark.py | — | 1-2 天 |
| 0.4 | 扩展单元测试（GPU 量化、互操作） | — | 1 天 |
| **Phase 1** | 核心实验 | Phase 0 | 5-7 天 |
| 1.1 | 实验一：TTFT 测试（4 配置 × 6 输入 × 5 重复） | 0.2 | 2-3 天 |
| 1.2 | 实验二：质量评估（5 配置 × 5 基准） | 0.3 | 2-3 天 |
| 1.3 | 实验四：GPU vs CPU 量化微基准 | 0.2 | 1 天 |
| **Phase 2** | 深入实验 | Phase 1 | 5-7 天 |
| 2.1 | 实验三：逐层自适应策略消融 | 1.2 | 2-3 天 |
| 2.2 | 实验五：网络条件测试 | 1.1 | 2-3 天 |
| 2.3 | 实验六：并发与稳定性测试 | 1.1 | 1-2 天 |
| **Phase 3** | 数据分析与可视化 | Phase 2 | 3-5 天 |
| 3.1 | 数据清洗与统计分析 | — | 1-2 天 |
| 3.2 | 图表绘制 | 3.1 | 1-2 天 |
| 3.3 | 结果撰写 | 3.2 | 1-2 天 |

**总计：约 16-24 个工作日**

### 6.2 实验记录规范

每次实验需记录：
1. 实验编号和日期
2. Git commit hash
3. 完整启动命令
4. 环境变量和配置文件
5. 网络延迟/带宽设置
6. 原始输出日志
7. 提取的度量数据（CSV 格式）
8. 异常情况说明

目录结构：
```
results/
├── exp1_ttft/
│   ├── 2026-04-15_baseline_short/
│   │   ├── config.json          # 实验配置
│   │   ├── prefill.log          # Prefill 日志
│   │   ├── decode.log           # Decode 日志
│   │   ├── metrics.csv          # 提取的度量
│   │   └── notes.md             # 异常说明
│   └── ...
├── exp2_quality/
├── exp3_ablation/
├── exp4_gpu_vs_cpu/
├── exp5_network/
├── exp6_concurrency/
└── summary/
    ├── all_metrics.csv          # 汇总数据
    └── figures/                 # 论文图表
```

---

## 七、论文图表规划

### 7.1 必要图表清单

| 图号 | 类型 | 内容 | 对应实验 |
|------|------|------|---------|
| Fig.1 | 架构图 | PD 分离 + 传输量化数据流 | — |
| Fig.2 | 柱状图 | 不同量化配置的 TTFT 对比（分输入长度） | 3.1 |
| Fig.3 | 折线图 | TTFT 随输入长度的变化趋势 | 3.1 |
| Fig.4 | 表格 | 各基准的准确率对比 | 3.2 |
| Fig.5 | 散点图 | 质量-压缩比 Pareto 曲线 | 3.3 |
| Fig.6 | 热力图 | 层敏感度分析 | 3.3 |
| Fig.7 | 柱状图 | GPU vs CPU 量化耗时对比 | 3.4 |
| Fig.8 | 折线图 | 延迟-TTFT 曲线（不同量化配置） | 3.5 |
| Fig.9 | 热力图 | 延迟×带宽条件下的量化加速比 | 3.5 |
| Fig.10 | 折线图 | 并发数-吞吐量曲线 | 3.6 |
| Fig.11 | 折线图 | 长时间运行的延迟稳定性 | 3.6 |
| Fig.12 | 饼图/堆叠图 | TTFT 耗时分解（计算/量化/传输/反量化） | 3.1+3.4 |

### 7.2 表格清单

| 表号 | 内容 | 对应实验 |
|------|------|---------|
| Tab.1 | 实验环境详细配置 | 2.1-2.2 |
| Tab.2 | 量化压缩比理论值 vs 实测值 | 3.1 |
| Tab.3 | 各基准准确率（含标准差） | 3.2 |
| Tab.4 | 逐层量化策略对比 | 3.3 |
| Tab.5 | 网络条件测试结果汇总 | 3.5 |
| Tab.6 | 并发测试 P50/P90/P99 延迟 | 3.6 |

---

## 八、已知风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| P0: Scheduler segfault | 长时间/高并发测试中断 | 每次测试重启服务；记录崩溃频率作为稳定性指标 |
| P1: bootstrap_room mismatch | 并发测试失败 | 串行执行并发测试；每次请求使用独立 bootstrap_room |
| 4-bit 量化不稳定 | 质量评估结果波动 | 增加重复次数；排查是数值问题还是网络问题 |
| 网络模拟不精确 | tc netem 抖动影响结果 | 增加重复次数；使用固定延迟（无抖动）做对照 |
| 评估基准运行时间长 | 实验周期延长 | 先用小样本验证流程，再跑完整评估 |
| 单模型泛化性不足 | 审稿人质疑 | 如有条件，增加 Llama-3-8B 作为第二个模型 |

---

## 九、与论文章节的对应关系

| 论文章节 | 对应实验 | 核心论点 |
|---------|---------|---------|
| 第三章 系统设计 | — | 传输量化架构设计与实现 |
| 第四章 实验评估 | | |
| 4.1 实验设置 | 第二节 | 环境、工具、方法论 |
| 4.2 传输加速效果 | 实验 3.1 | 量化有效降低 TTFT |
| 4.3 模型质量影响 | 实验 3.2 | 量化对输出质量影响可控 |
| 4.4 逐层自适应策略 | 实验 3.3 | 混合精度优于统一量化 |
| 4.5 GPU 加速优化 | 实验 3.4 | GPU 量化消除 CPU 瓶颈 |
| 4.6 网络适应性分析 | 实验 3.5 | 量化在高延迟/低带宽场景收益显著 |
| 4.7 系统可扩展性 | 实验 3.6 | 系统在并发场景下表现稳定 |
| 第五章 总结与展望 | — | 局限性与未来工作 |

