# P/D Disaggregation TCP KV Transfer 进展报告

> 最后更新: 2026-04-08

## 1. 目标

在 `sglang-integrate-pd-scheduling-kvcache` 分支上实现并验证 TCP backend 的 KV cache 传输逻辑，支持 Prefill 和 Decode 服务器之间的端到端 disaggregation。

## 2. 测试环境

| 角色 | 主机 | 内网 IP | 端口 | 模型 |
|------|------|---------|------|------|
| Prefill | gpu1 (117.50.192.238) | 10.60.23.70 | 30000 | Qwen2.5-7B |
| Decode | gpu2 (117.50.189.89) | 10.60.30.66 | 30001 | Qwen2.5-7B |

- 代码路径: `/home/ubuntu/sglang-kvtuner-splitwise/`
- Python venv: `/home/ubuntu/sglang-env/` (已安装本地 sglang，无需 PYTHONPATH)
- 分支: `sglang-integrate-pd-scheduling-kvcache`
- 启动参数: `--disable-overlap-schedule` (scheduler 在 overlap 模式下会 segfault)

## 3. 当前状态: 端到端已跑通 ✅

TCP KV transfer 的完整流程（batch + pipeline 两种模式）均已在双机环境下验证通过。

### 测试结果

| 测试 | 输入 tokens | 生成 tokens | 模式 | 延迟 | 结果 |
|------|------------|------------|------|------|------|
| 短文本 | 1 | 32 | pipeline | 0.7s | ✅ |
| 长文本 | 43 | 64 | pipeline | 1.3s | ✅ |

### 验证通过的完整流程

```
[Decode] init() → ZMQ descriptor → [Prefill] _process_zmq_msg() → _PendingTransfer
[Prefill] forward pass → send_layer() 逐层发送 KV (pipeline)
                    → _send_tail() aux metadata (10 buffers) + EOF
                    → _finish() status=Success
[Decode] _recv_loop() → TCP connect → _recv_kv() 56 条消息
         → _write_pages_to_gpu() → poll()=Success → generation 完成
```

## 4. 代码修复历史（按时间顺序）

### 4.0 修复 aux metadata 只传第一个 buffer (2817a3f1d) ★★★

**根因**: `_send_tail()` 只发送 `aux_data_ptrs[0]`，但 `MetadataBuffers` 有 10 个 buffer。Decode 侧检查 `bootstrap_room[0].item() == 0` 认为 metadata 未就绪，请求永远卡在 TransferQueue。

**修复**: `_send_tail()` / `_recv_kv()` 改为遍历所有 aux buffers，拼接/拆分发送。

### 4.1 Pipeline mode 逐层传输 (7cf5524eb)

**改动**: `tp_worker.py` 新增 `_build_layer_kv_send_fn()`，在 forward pass 每层 attention 后立即调用 `send_layer()` 发送该层 KV，实现 GPU 计算与 TCP 传输的流水线重叠。

**改进**: `send_layer()` 增加 `_pipeline_aborted` 标志、torch.Tensor 支持、pipeline 失败智能回退。

### 4.2 对齐 disagg decode overlap loop (62dca923d)

将 `event_loop_overlap_disagg_decode` 对齐标准 `event_loop_overlap`，修复 `_engine_paused`、`is_disable_overlap_for_batch`、`self_check_during_idle` 等缺失守卫。

### 4.3 CUDA 上下文初始化 + transfer_started 守卫 (fac2127ef)

`_PendingTransfer.run()` / `_receive_layer()` 加 `torch.cuda.set_device()`。TCPKVReceiver 加 `_transfer_started` 状态守卫。

### 4.4 TCPKVReceiver WaitingForInput 转换 (4844b6c24)

TCPKVReceiver.__init__() 获取 bootstrap info 后调用 `update_status(WaitingForInput)`。

### 4.5 其他修复

- `TCPKVSender.poll()` 不应包装 `check_status()` 为 `KVPoll()`
- KVPoll.Transferring/Success 在 decode handshake waiters 中的处理
- FlashInfer layer pipeline hook 修复
- ZMQ 可靠性改进

## 5. 未解决的问题

### P0: Scheduler 线程 segfault

**现象**: scheduler 线程运行 2-5 分钟后变成 `<defunct>` zombie（静默 native segfault）。`--disable-overlap-schedule` 下同样发生。

**影响**: scheduler defunct 后整个进程被内核回收，所有子线程（包括传输线程）被杀。当前通过在 scheduler 崩溃前完成测试来绕过。

**排查方向**:
1. `ulimit -c unlimited` + core dump 分析
2. `dmesg | tail` 查看 segfault 信息
3. 在原版 sglang（无 TCP 修改）上复现确认是否为 pre-existing bug
4. 检查 sglang 上游 issue

### P1: Pipeline 回退后的数据一致性

`send_layer()` 失败时设 `_pipeline_aborted=True`，已发送的层不重复但未发送的层走 batch mode。当前注释承认 "Decode will timeout on missing layers"，这不是一个优雅的失败处理。

### P1: `wait_for_conn` 阻塞

`send_layer()` 在每层都调用 `wait_for_conn(timeout=_TCP_CONN_WAIT_S)`，如果 TCP 连接建立慢，会阻塞 forward pass。可以考虑非阻塞检查或只在第一次等待。

## 6. 可优化项

### 性能优化

1. **GPU→CPU 传输**: `_read_pages_from_gpu()` 用 `cuMemcpyDtoH`，考虑 pinned memory + stream 异步传输
2. **TCP 发送**: 当前逐层同步发送，考虑 `socket.sendall()` 改为非阻塞 IO + buffer
3. **多 buffer 拼接**: `_send_tail()` 中 `b"".join(chunks)` 产生额外拷贝，考虑直接 writev
4. **Pipeline 吞吐 benchmark**: 需要在长文本 (1K+ tokens)、多并发请求下测量端到端延迟
5. **Batch vs Pipeline 对比**: 在相同条件下对比两种模式的延迟和吞吐差异

### 代码清理

1. **Debug 日志**: `conn.py` 有 28 处 `logger.warning`，`prefill.py` 有 5 处，需要降级为 `logger.debug` 或移除
2. **`configs/default.sh`**: SSH host 已改为外网 IP (`ubuntu@117.50.192.238`)，需要同时支持内外网访问
3. **tp_worker.py FIXME**: 两处 `FIXME(lsyin)` 注释需要清理

### 功能完善

1. **多卡支持**: 当前仅验证 TP=1，需验证 TP>1 和 CP 场景
2. **错误恢复**: TCP 连接断开后的重试机制、超时后的资源清理
3. **指标观测**: 添加传输时间、吞吐量等 metrics（Prometheus/日志）
4. **断开 PYTHONPATH 依赖**: 确认所有环境都通过 venv 安装 sglang，无需 `PYTHONPATH`

## 7. 代码结构

| 文件 | 行数 | 说明 |
|------|------|------|
| `python/sglang/srt/disaggregation/tcp/conn.py` | ~1200 | TCP backend 核心实现 |
| `python/sglang/srt/disaggregation/prefill.py` | ~750 | Prefill 事件循环和 bootstrap queue |
| `python/sglang/srt/disaggregation/decode.py` | ~1200 | Decode prealloc queue 和请求处理 |
| `python/sglang/srt/managers/tp_worker.py` | ~500 | Pipeline mode `layer_kv_send_fn` 注入 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | ~870 | Layer pipeline hook 调用点 |

## 8. 测试工具

| 文件 | 说明 |
|------|------|
| `scripts/pd_disagg_test/pd_test.sh` | 一键测试主控脚本 (full/start/stop/status/test/logs) |
| `scripts/pd_disagg_test/remote_worker.sh` | 远程辅助脚本 |
| `scripts/pd_disagg_test/pd_coordinator.py` | Python PD coordinator（替代 Rust Router） |
| `scripts/pd_disagg_test/configs/default.sh` | 集群配置 |
