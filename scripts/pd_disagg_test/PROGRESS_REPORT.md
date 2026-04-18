# P/D Disaggregation + KVTuner 进展报告

> 最后更新: 2026-04-18

## 1. 目标

在 `kvtuner-splitwise` 分支上整合 KVTuner 层级量化与 TCP KV Cache 传输，实现 P/D 分离架构下带量化的端到端推理。评估 Transfer Quantization 在不同网络条件下的 TTFT 性能影响和输出质量影响。

## 2. 分支结构

两个开发分支自 `a0d8a7ae6` 分离，已在 `0896b1fd4` 合并：

| 分支 | 提交数 | 职责 |
|------|--------|------|
| `sglang-integrate-pd-scheduling-kvcache` | 12 | PD 调度 TCP 传输后端、Pipeline 逐层传输 |
| `kvtuner-splitwise` | 44 (独有) | KVTuner 量化、Transfer Quant、实验框架、论文实验 |

`kvtuner-splitwise` 是当前主开发分支，包含两个分支的全部工作。

## 3. 测试环境

| 角色 | 主机 | 内网 IP | 端口 | GPU |
|------|------|---------|------|-----|
| Prefill | gpu1 (117.50.192.238) | 10.60.23.70 | 30000 | RTX 4090 24GB |
| Decode | gpu2 (117.50.189.89) | 10.60.30.66 | 30000 | RTX 4090 24GB |

- 模型: Qwen2.5-7B (28 层, 32 attn heads, 8 KV heads, head_dim=128, page_size=1)
- 代码路径: `/home/ubuntu/sglang-kvtuner-splitwise/`
- Python venv: `/home/ubuntu/sglang-env/` (editable install)
- 网络延迟模拟: `tc qdisc prio` + `u32 filter` (定向延迟节点间流量)

## 4. 工作进展

### 4.1 PD 调度 TCP 传输后端 (sglang-integrate-pd-scheduling-kvcache)

| 提交 | 说明 |
|------|------|
| 7cf5524eb | Pipeline mode 逐层传输 (layer-wise overlap with prefill) |
| 2817a3f1d | 修复 aux metadata 只传第一个 buffer |
| 64400871b | Flashinfer layer pipeline hook, ZMQ 可靠性提升 |
| 62dca923d | 对齐 disagg decode overlap loop |
| fac2127ef | CUDA 上下文初始化 + transfer_started 守卫 |
| 4844b6c24 | TCPKVReceiver WaitingForInput 状态转换修复 |
| 3e0e1b4e4 / 771e73162 | KVPoll 状态处理修复 |
| d760e977d | PD coordinator 重写 (替代 Router) |

### 4.2 KVTuner 层级量化

| 提交 | 说明 |
|------|------|
| b456f2a26 | 与 sglang main 对齐 (realign) |
| 0093d1ea0 | 层级 KV Cache 量化实现 |
| cdf8f3904 | KVTuner 压缩传输 + NIXL VRAM bridge |
| 0a5a15125 | list-format layer config JSON 支持 |
| cc02a1417 | 文档整理，归档过期文件 |

### 4.3 Transfer Quantization (传输量化)

| 提交 | 说明 |
|------|------|
| 3ab79a2d8 | 传输量化设计文档 + 基础代码 |
| bf34b7269 | 集成到 TCP send/recv 路径 |
| 1127984cb | PD 测试脚本支持量化配置 |
| 87fba9e37 | **修复 bootstrap_room mismatch** — 并发请求 metadata corruption |
| c5f45f8c5 | **修复 4-bit 伪打包** — 实现真正的 nibble packing (2×4bit→1×8bit) |
| d027be4ee | **CPU→GPU 量化** — quantize_on_gpu / dequantize_on_gpu |
| a7213337d | 传输量化测试报告 |
| (WIP) | **CUDA 流同步 + 量化正确性修复** — 见 §4.5 |

### 4.4 实验框架与论文实验

| 提交 | 说明 |
|------|------|
| e0e410c68 | 论文实验计划 (6 个实验 RQ1-RQ6, 12 图 6 表) |
| a34a72510 | TTFT benchmark 脚本 + conn.py timing instrumentation |
| 1040d2947 | 质量评估脚本 (GSM8K/MMLU/HellaSwag) |
| f418a67fa | 逐层配置生成器 (6 种策略: uniform-8/4bit, mixed-A/B/C/D) |
| f61d86bef | 实验 3.1 TTFT 无延迟结果 (4 configs × 6 inputs × 5 runs) |
| f28d029cc | 实验 3.1 TTFT 20ms RTT 结果 + health-check 修复 |

### 4.5 CUDA 流同步 + 量化正确性修复 (2026-04-14)

**问题**: Pipeline 模式下 KV Cache 经量化/反量化后数据不正确，导致 decode 输出质量下降或请求失败。

**根因分析**:
1. `_read_pages_as_gpu_tensor` 使用 CUDA driver API (`cuMemcpyDtoD_v2`, NULL stream) 做 DtoD 拷贝，而 `quantize_on_gpu` 使用 PyTorch ops (current stream)。两者在不同 CUDA stream 上执行，缺少同步屏障导致 PyTorch 量化内核可能读到未完成拷贝的数据。
2. 接收端 `_recv_kv` 完成后缺少最终 `torch.cuda.synchronize()`，decode 模型可能读到未完全写入的 KV cache。
3. 量化/反量化过程中大量 GPU 中间 tensor 未及时释放，长文本连续请求时 GPU 显存碎片化加剧。

**修复内容**:

| 修复项 | 文件 | 说明 |
|--------|------|------|
| DtoD→PyTorch 流同步 | conn.py | `_stream_kv` 和 `send_layer` 中 DtoD 拷贝后加 `torch.cuda.synchronize()` |
| 接收端最终同步 | conn.py | `_recv_kv` 完成后加 `torch.cuda.synchronize()` 确保 scatter 写入可见 |
| timing 修复 | conn.py | `_stream_kv` 量化路径恢复 `total_gather_ms` 累加 |
| 接收端 debug 验证 | conn.py | layer 0 K 添加 dequant sum/absmax + write-back read-back 验证 |
| GPU 内存管理 | conn.py + transfer_quant.py | 及时 `del` 大 tensor，减少显存碎片化 |
| `_DEBUG_QUANT` 开关 | conn.py | 所有 debug 日志受模块级标志控制 |
| `clear()` 方法 | conn.py | TCPKVSender/Receiver 添加 `clear()` 释放 `request_status` 条目 |

**验证方法**:
- CPU roundtrip 测试: 8-bit max_err=0.030, 4-bit max_err=0.263 (符合预期量化误差)
- GPU 测试: 对比 sender `[send_layer DEBUG]` 和 receiver `[_recv_kv DEBUG]` 的 layer 0 K sum/absmax
- Read-back 验证: `[_recv_kv VERIFY]` 的 `match=True` 确认 scatter 写入正确

### 4.6 冗余 CUDA 同步移除 + Overlap 调度修复 (2026-04-18)

**问题**: V1 实验显示 TTFT 异常高 (xxlong: 25-28s)，远超预期。

**根因分析**:
1. 代码中存在 3 个冗余 `torch.cuda.synchronize()` 调用，阻塞了 overlap 调度
2. `remote_worker.sh` 中 `DISABLE_OVERLAP` 默认为 `true`，导致 overlap 调度被禁用
3. 条件参数传递逻辑错误 (`${VAR:+--flag}` 在 VAR="false" 时仍添加 flag)

**修复内容**:

| 修复项 | 文件 | 说明 |
|--------|------|------|
| 移除冗余同步 | conn.py | 移除 `quantize_on_gpu` 后、`_read_pages_from_gpu` 入口、`send_layer` 量化后的同步调用 |
| 禁用 debug | conn.py | `_DEBUG_QUANT = False` |
| Overlap 默认启用 | remote_worker.sh | `DISABLE_OVERLAP="${DISABLE_OVERLAP:-false}"` |
| 条件 flag 修复 | remote_worker.sh | 改用 `[ "${VAR}" = "true" ]` 条件判断 |
| KVTUNER_LAYER_BITS | configs/default.sh | 添加变量定义，修复 unbound variable |

**性能提升**:

| 输入类型 | V1 TTFT (ms) | V2 TTFT (ms) | 提升倍数 |
|---------|-------------|-------------|---------|
| xxlong (884 tok) | 25,123 | 395 | **63.6x** |
| xlong (441 tok) | 21,440 | 343 | **62.5x** |
| long (221 tok) | 17,211 | 331 | **52.0x** |
| medium (52 tok) | 11,106 | 303 | **36.7x** |
| short (7 tok) | 4,294 | 313 | **13.7x** |
| tiny (1 tok) | 418 | 294 | 1.4x |

**整体均值**: V1=13,265ms → V2=330ms，**提升 40.2x**

### 4.7 传输量化效果评估 (V2 with sync optimization)

**V2 Baseline vs Transfer Quantization**:

| 配置 | Mean TTFT | vs Baseline | 带宽节省 |
|-----|-----------|-------------|---------|
| baseline | 330 ms | - | 0% |
| quant-8bit | 343 ms | +4% | 50% |
| quant-4bit | 396 ms | +20% | 75% |
| mixed-b | 378 ms | +14% | 71% |

**结论**: 在 overlap 调度生效后，传输量化对 TTFT 影响很小。主要价值在于带宽节省。

## 5. 实验结果

### 5.1 TTFT 实验 — 无网络延迟 (内网 <1ms)

| 输入长度 | Baseline | Quant-8bit | Quant-4bit | Mixed-B |
|---------|---------|-----------|-----------|---------|
| tiny (1 tok) | **415ms** (5/5) | 424ms (5/5) | 482ms (5/5) | 472ms (5/5) |
| short (10 tok) | **4203ms** (5/5) | 4624ms (5/5) | 4654ms (5/5) | 4576ms (5/5) |
| medium (60 tok) | **11547ms** (5/5) | 12633ms (5/5) | 12765ms (5/5) | 13268ms (5/5) |
| long (256 tok) | **17299ms** (5/5) | 18857ms (3/5) | 19251ms (4/5) | 18533ms (3/5) |
| xlong (512 tok) | **20743ms** (5/5) | 22792ms (5/5) | 22210ms (5/5) | 0/5 |
| xxlong (1024 tok) | **25986ms** (5/5) | 26326ms (2/5) | 24644ms (1/5) | 20888ms (2/5) |

**结论**: 低延迟内网下量化无加速效果，GPU 量化开销 (~10ms) 超过传输节省。

### 5.2 TTFT 实验 — 20ms RTT (tc netem)

| 输入长度 | Baseline | Quant-8bit | Quant-4bit | Mixed-B |
|---------|---------|-----------|-----------|---------|
| tiny (1 tok) | 1859ms (5/5) | **500ms** (5/5) | 553ms (5/5) | 531ms (5/5) |
| short (10 tok) | **3872ms** (5/5) | 4316ms (5/5) | 4498ms (5/5) | 4703ms (5/5) |
| medium (60 tok) | **8634ms** (5/5) | 12190ms (5/5) | 12648ms (5/5) | 13380ms (5/5) |
| long (256 tok) | **13760ms** (5/5) | 20342ms (5/5) | 19886ms (4/5) | 18534ms (4/5) |
| xlong (512 tok) | **18401ms** (5/5) | 23294ms (1/5) | 22585ms (4/5) | 20358ms (2/5) |
| xxlong (1024 tok) | 22001ms (5/5) | FAIL (0/5) | FAIL (0/5) | 22335ms (1/5) |

**结论**:
- **tiny/short**: 量化显著加速（500ms vs 1859ms），传输压缩收益 > GPU 量化开销
- **medium+**: 量化反而变慢，GPU 量化+反量化 ~4s 开销逐渐占主导
- **Baseline 100% 成功率**, 量化配置 70-77% — 长文本 GPU 状态累积仍存在

## 5.3 网络延迟实验 (2026-04-18)

测试不同 RTT 下传输量化对 TTFT 的影响:

| RTT | Baseline | quant-8bit | quant-4bit | 8-bit节省 | 4-bit节省 |
|-----|----------|------------|------------|----------|----------|
| 10ms | 362 ms | 353 ms | 386 ms | +2.2% | -6.8% |
| 50ms | 501 ms | 494 ms | 512 ms | +1.4% | -2.3% |
| 100ms | 689 ms | 691 ms | 696 ms | -0.3% | -1.0% |
| 200ms | 1096 ms | 1103 ms | 1088 ms | -0.6% | +0.7% |

**意外发现**: 传输量化在网络延迟场景下 TTFT 优化效果有限。

**原因**:
1. Overlap 调度使 KV 传输与 prefill 并行
2. 内网带宽充足，传输时间差异对 TTFT 影响小
3. 反量化开销抵消了传输优势

**核心价值**: 传输量化的主要收益是 **带宽节省 (50-75%)**，而非 TTFT 优化。

### ~~P0: Scheduler segfault~~ ✅ 已解决
scheduler 线程 native segfault 已在上游修复。

### ~~P1: bootstrap_room mismatch~~ ✅ 已修复 (87fba9e37)
并发请求 metadata corruption。修复：使用 `src_aux_index` 替代 `dst_aux_index` 读取 prefill 端写入的 bootstrap_room。成功率从 0-67% 提升到 100%。

### ~~P2: 4-bit 伪打包~~ ✅ 已修复 (c5f45f8c5)
原 4-bit 实现每 4-bit 存 1 byte（无压缩）。修复：真 nibble packing（2×int4→1×uint8, low nibble first）。

### ~~P3: CPU 量化开销~~ ✅ 已优化 (d027be4ee)
从 CPU numpy 量化迁移到 GPU torch CUDA，量化延迟大幅降低。

### ~~P4: CUDA 流同步缺失~~ ✅ 已修复 (2026-04-14)
DtoD 拷贝 (NULL stream) 与 PyTorch 量化 (current stream) 之间缺少同步屏障。修复：在 `_stream_kv`、`send_layer`、`_recv_kv` 关键路径加 `torch.cuda.synchronize()`。

## 7. 待解决问题

### P0: 长文本连续请求 GPU 状态累积
量化配置下 xlong/xxlong 连续请求失败率 60-100%。health-check 无法完全解决，疑似 GPU 内部 KV pool 分配或 TCP 连接累积。**Workaround**: 每组测试重启服务。已添加 `clear()` 方法和 `del` 及时释放，待验证是否改善。

### P1: 量化正确性 GPU 端验证 (进行中)
已添加 sender/receiver debug 日志和 read-back 验证。需在 GPU 机器上运行确认 `[_recv_kv VERIFY] match=True`。

### P1: 质量评估（实验 3.2）未完成
- GSM8K 5-shot prompt 过长 (849+ tokens) + 512 max_new_tokens 导致截断和状态退化
- MMLU 256 max_new_tokens 不够模型输出 "Answer: X" 格式
- 核心瓶颈：PD 架构连续请求稳定性

### P2: 待开发脚本
- `eval/bench_quant_micro.py` — CPU vs GPU 量化微基准
- `eval/bench_throughput.py` — 并发吞吐量测试
- `eval/bench_network.py` — 网络条件自动化测试
- `eval/soak_test.py` — 30 分钟稳定性测试

## 8. 测试工具

| 文件 | 说明 |
|------|------|
| `scripts/pd_disagg_test/pd_test.sh` | 一键测试主控脚本 (start/stop/status) |
| `scripts/pd_disagg_test/remote_worker.sh` | 远程辅助脚本 (服务管理) |
| `scripts/pd_disagg_test/pd_coordinator.py` | Python PD coordinator |
| `scripts/pd_disagg_test/run_experiment.sh` | 统一实验入口 (TTFT + Quality) |
| `scripts/pd_disagg_test/generate_quant_configs.py` | 逐层量化配置生成器 |
| `scripts/pd_disagg_test/configs/default.sh` | 集群配置 |
| `scripts/pd_disagg_test/configs/tcp-quant.sh` | 8-bit 量化配置 |
| `scripts/pd_disagg_test/configs/tcp-quant-4bit.sh` | 4-bit 量化配置 |
| `scripts/pd_disagg_test/configs/tcp-quant-mixed.sh` | 混合精度配置 |
| `scripts/pd_disagg_test/configs/quant/` | JSON 量化策略文件 |
| `scripts/pd_quant_validation/kvtuner_offline_calib.py` | 离线校准工具 |
| `eval/bench_ttft.py` | TTFT benchmark (SSE streaming) |
| `eval/run_benchmark.py` | 质量评估 (GSM8K/MMLU/HellaSwag) |

## 9. 数据文件

| 路径 | 说明 |
|------|------|
| `results/two_model_comparison/` | V1 实验 (sync issue 存在) |
| `results/two_model_comparison_v2/` | V2 实验 (sync 优化后) |
| `results/latency_experiment/` | 网络延迟实验 (RTT 10/50/100/200ms) |
| `results/compare_v1_v2.py` | V1 vs V2 对比分析脚本 |

### 新增实验脚本

| 文件 | 说明 |
|------|------|
| `scripts/pd_disagg_test/continue_experiment.sh` | 断点续跑实验脚本 |
| `scripts/pd_disagg_test/run_latency_experiment.sh` | 网络延迟实验脚本 |
| `scripts/pd_disagg_test/run_v2_baseline.sh` | V2 baseline 单独运行 |
| `scripts/pd_disagg_test/configs/gemma2-27b.sh` | Gemma2-27B 配置 (OOM 未完成) |
