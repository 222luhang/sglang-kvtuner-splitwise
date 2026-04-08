# P/D Disaggregation + KVTuner 进展报告

> 最后更新: 2026-04-08

## 1. 目标

在 `kvtuner-splitwise` 分支上整合 KVTuner 层级量化与 TCP KV Cache 传输，实现 P/D 分离架构下带量化的端到端推理。

## 2. 测试环境

| 角色 | 主机 | 内网 IP | 端口 | 模型 |
|------|------|---------|------|------|
| Prefill | gpu1 (117.50.192.238) | 10.60.23.70 | 30000 | Qwen2.5-7B |
| Decode | gpu2 (117.50.189.89) | 10.60.30.66 | 30000 | Qwen2.5-7B |

- 代码路径: `/home/ubuntu/sglang-kvtuner-splitwise/`
- Python venv: `/home/ubuntu/sglang-env/` (editable install)
- KVTuner 量化配置: `/home/ubuntu/qwen2.5-7b_layer_quant.json` (32层, 5层8-bit + 27层4-bit)
- 启动参数: `--enable-kvtuner-quant --kvtuner-layer-config <json> --disable-cuda-graph`

## 3. 测试结果

### KVTuner + TCP 合并测试 (2026-04-08)

| 测试 | 输入 | 生成 | 延迟 | 结果 |
|------|------|------|------|------|
| 单请求 | 10 tokens | 32 tokens | 0.8s | PASS |
| batch short_short | 1 | 16 | 0.4s | PASS |
| batch short_medium | 1 | 64 | 1.5s | PASS |
| batch short_long | 1 | 128 | 2.9s | PASS |
| batch medium_short | 10 | 16 | 0.8s | PASS |
| batch medium_medium | 10 | 64 | 7.8s | PASS |
| batch medium_long | 28 | 128 | 39.1s | PASS |
| batch long_short | 60 | 16 | 7.1s | PASS |
| batch long_medium | 60 | 64 | 32.8s | PASS |
| batch long_long | 60 | 256 | 120s | TIMEOUT |

**总计: 8/9 PASS** (long_long 超时为已知 scheduler 问题，非 KVTuner 相关)

### 并发测试
- 3 并发请求: 1/3 PASS (bootstrap_room mismatch 竞态条件)

## 4. 代码修复历史

### TCP KV Transfer (已在 main 分支)
- 修复 aux metadata 只传第一个 buffer (2817a3f1d)
- Pipeline mode 逐层传输 (7cf5524eb)
- 对齐 disagg decode overlap loop (62dca923d)
- CUDA 上下文初始化 + transfer_started 守卫 (fac2127ef)
- TCPKVReceiver WaitingForInput 转换 (4844b6c24)

### KVTuner 量化
- 分支合并: sglang-integrate-kvtuner-quantization → kvtuner-splitwise (0896b1fd4)
- 代码清理: 删除重复文件、过期脚本、未使用模块 (2026-04-08)

## 5. 未解决问题

### P0: Scheduler 线程 segfault
scheduler 线程运行 2-5 分钟后 native segfault，影响 long_long 测试和并发测试。

### P1: 并发 bootstrap_room mismatch
多个并发请求通过 pd_coordinator 发送时，可能出现 metadata corruption 错误。

### P2: TCP 传输性能优化
GPU→CPU 传输、TCP 非阻塞 IO、pipeline 吞吐 benchmark 待优化。

## 6. 测试工具

| 文件 | 说明 |
|------|------|
| `scripts/pd_disagg_test/pd_test.sh` | 一键测试主控脚本 |
| `scripts/pd_disagg_test/remote_worker.sh` | 远程辅助脚本 |
| `scripts/pd_disagg_test/pd_coordinator.py` | Python PD coordinator |
| `scripts/pd_disagg_test/configs/default.sh` | 集群配置 |
| `scripts/pd_quant_validation/kvtuner_offline_calib.py` | 离线校准工具 |
