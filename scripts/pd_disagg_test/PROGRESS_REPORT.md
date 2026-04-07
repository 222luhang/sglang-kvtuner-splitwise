# P/D Disaggregation TCP KV Transfer 调试进展报告

> 最后更新: 2026-04-08

## 1. 目标

在 `sglang-integrate-pd-scheduling-kvcache` 分支上，验证 TCP backend 的 KV cache 传输逻辑能在 Prefill 和 Decode 两台服务器之间端到端跑通。

## 2. 测试环境

| 角色 | 主机 | IP | 端口 | 模型 |
|------|------|----|------|------|
| Prefill | gpu1 | 10.60.23.70 | 30000 | Qwen2.5-7B (/data/Qwen/Qwen2.5-7B) |
| Decode | gpu2 | 10.60.30.66 | 30001 | Qwen2.5-7B (/data/Qwen/Qwen2.5-7B) |

- 代码路径: `/home/ubuntu/sglang-kvtuner-splitwise/`
- 必须设置 `PYTHONPATH=/home/ubuntu/sglang-kvtuner-splitwise/python` 才能加载本地 TCP backend 代码
- 分支: `sglang-integrate-pd-scheduling-kvcache`
- 启动参数: `--disable-overlap-schedule` (scheduler 在 overlap 模式下会 segfault)

## 3. 代码修复历史

### 3.0 ★★★ 修复 aux metadata 只传输第一个 buffer 的 bug (2026-04-08)

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py`

**根本原因**: `_send_tail()` 只发送了 `aux_data_ptrs[0]`（output_ids），但 `MetadataBuffers` 有 10 个 buffer（output_ids, cached_tokens, logprobs, ..., bootstrap_room）。Decode 侧 `_commit_transfer_to_req()` 检查 `bootstrap_room[0].item() == 0` 时认为 metadata 还没准备好，导致请求永远卡在 TransferQueue。

**修复**:
- `_send_tail()`: 遍历所有 `aux_data_ptrs`，拼接成一个 blob 发送
- `_recv_kv()`: 按 `aux_item_lens` 拆分 blob，写入所有 aux buffers

**验证**: 端到端测试通过，Decode 在 0.7s 内完成 KV 接收 + 32 token generation。

### 3.1 TCPKVReceiver WaitingForInput 转换 (4844b6c24)

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py` — TCPKVReceiver.__init__()

在获取 bootstrap 信息后立即调用 `update_status(bootstrap_room, KVPoll.WaitingForInput)`，与 MooncakeKVReceiver 保持一致。

### 3.2 CUDA 上下文初始化 + transfer_started 守卫 (fac2127ef)

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py`

- `_PendingTransfer.run()`: 添加 `torch.cuda.set_device()` 确保 CUDA 上下文可用
- `_receive_layer()`: 添加 `torch.cuda.set_device()` 确保 GPU 写入
- `TCPKVReceiver.__init__()`: 新增 `_transfer_started = False`
- `TCPKVReceiver.init()`: 设置 `_transfer_started = True`
- `TCPKVReceiver.poll()`: `_transfer_started` 为 False 时返回 `WaitingForInput`

### 3.3 对齐 disagg decode overlap loop (62dca923d)

**文件**: `python/sglang/srt/disaggregation/decode.py`

将 `event_loop_overlap_disagg_decode` 对齐标准 `event_loop_overlap`，新增:
- `_engine_paused` 守卫
- `is_disable_overlap_for_batch` 判断
- `self_check_during_idle()` 空闲检查
- `is_generation` 守卫在 `launch_batch_sample_if_needed`

## 4. 已验证的 TCP 传输流程

### 4.1 完整流程追踪结果

```
[Decode scheduler] recv_requests() → process_input_requests()
    ✅ 请求从 ZMQ 接收到 scheduler
    ✅ handle_generate_request() 正确处理
    ✅ disagg_decode_prealloc_queue.add() 加入 pending_reqs

[Decode] _resolve_pending_reqs()
    ✅ ensure_parallel_info() 从 bootstrap server 获取 prefill 信息
    ✅ _resolve_prefill_dp_rank() 返回 dp_rank
    ✅ _create_receiver_and_enqueue() 创建 TCPKVReceiver

[Decode] pop_preallocated()
    ✅ _pre_alloc() 分配 KV 内存
    ✅ kv_receiver.init() 发送 ZMQ 描述符给 Prefill

[Prefill] _process_zmq_msg()
    ✅ ZMQ 消息到达 Prefill
    ✅ _PendingTransfer 创建，状态设为 WaitingForInput

[Prefill] pop_bootstrapped()
    ✅ sender 状态从 Bootstrapping 变为 WaitingForInput
    ✅ 请求移入 bootstrapped 列表

[Prefill] get_next_disagg_prefill_batch_to_run() + run_batch()
    ✅ Forward pass 执行完成

[Prefill] send_kv_chunk() → TCPKVSender.send()
    ✅ send_kv_chunk 被调用 (pages=2)
    ✅ send() 找到 _PendingTransfer (pending=found)
    ✅ ready() 被调用

[Prefill] _PendingTransfer.serve()
    ✅ TCP accept-loop 接受连接
    ✅ _ready.wait() 被唤醒 (_ready set! pipeline=False)
    ✅ _stream_kv 开始: cuda sync done, sending layers...
    ✅ 发送 layer 0/28, 7/28, 14/28, 21/28

[Decode] _recv_loop() (后台线程)
    ✅ 线程启动 (bootstrap_infos=yes)
    ❌ 无后续日志 — 卡在 TCP 接收或 GPU 写入
```

### 4.2 端到端验证结果 (2026-04-08)

| 步骤 | 状态 | 备注 |
|------|------|------|
| Prefill `_stream_kv` 完成 28 层发送 | ✅ | 全部 28 层在 <1s 内发送完成 |
| Prefill `_send_tail` + `_finish` | ✅ | aux metadata (10 buffers) + EOF 发送成功 |
| Decode `_recv_loop` TCP 连接 | ✅ | TCP connect 成功 |
| Decode `_recv_kv` 接收数据 | ✅ | 56 条消息全部接收 |
| Decode `_write_pages_to_gpu` GPU 写入 | ✅ | 所有 KV pages 写入 GPU |
| Decode poll() 返回 Success | ✅ | bootstrap_room 验证通过 |
| Decode generation 输出 | ✅ | 32 tokens 生成，0.7s 完成 |

## 5. 已解决的关键问题

### 5.1 ★★★ Scheduler 线程 segfault (部分解决)

**现象**: Scheduler 线程在运行 2-5 分钟后变成 `<defunct>` zombie。无 Python traceback，无 core dump，静默 native segfault。

**影响**: Scheduler 进程 defunct 后，整个进程（包括所有子线程）被内核回收，导致所有进行中的传输中断。

**排查结论**:
- 不是 TCP backend 修改导致的（原版 sglang 也有此问题）
- 与 overlap schedule 模式无关（`--disable-overlap-schedule` 下同样发生）
- 触发条件尚不明确，可能与 GPU 计算的时序有关

**当前绕过**: `--disable-overlap-schedule` 可延长 scheduler 存活时间（~3-5 分钟），但最终仍会崩溃。

### 5.2 Decode 侧 bootstrap_infos 缓存导致 TCP 连接失败

**现象**: Decode 首次连接正确，但 Prefill 重启后 Decode 缓存了旧的 tcp_port，导致 `Connection refused`。

**解决**: 每次重启 Decode 时清除 `connection_pool` 缓存。

### 5.3 Prefill scheduler CPU 空转 (Tight Loop)

**现象**: Scheduler 进程在 while True 循环中 90%+ CPU，`recv_requests()` 每次返回空列表（ZMQ NOBLOCK）。

**分析**: 这是预期行为 — 没有请求时 event loop 空转。标准 `event_loop_overlap` 也有相同行为。`maybe_sleep_on_idle()` 在 Decode 模式下，当各队列为空时会 sleep。

### 5.4 Debug 日志输出到 stderr 不被捕获

**解决**: 使用 `logger.warning()` 替代 `sys.stderr.write()`。

### 5.5 Debug 日志加到错误的 event loop

**解决**: Prefill 默认用 `event_loop_overlap_disagg_prefill`（不是 `event_loop_normal`）。

## 6. 架构理解总结

### 6.1 进程架构

```
Main Process (launch_server)
├── HTTP Server (uvicorn)  ← /health, /generate API
├── Tokenizer Manager     ← tokenize, send_to_scheduler (ZMQ PUSH)
├── Scheduler Process (fork)
│   ├── event_loop_*()     ← recv_requests (ZMQ PULL), process, forward
│   ├── TCP Accept Loop  ← accept TCP connections from Decode
│   ├── _PendingTransfer.serve() ← _stream_kv (GPU→CPU→TCP)
│   └── Detokenizer Process
└── Bootstrap Server (aiohttp) ← /route PUT/GET for registration
```

### 6.2 ZMQ 通信

- tokenizer_manager → scheduler: ZMQ PUSH/PULL (IPC)
- scheduler_recv_skipper: `scheduler_recv_interval <= 1` 时为 None（每次都 recv）

### 6.3 TCP KV Transfer 数据流

```
Decode init() → ZMQ descriptor → Prefill _process_zmq_msg()
                                          → _PendingTransfer 创建
Decode _recv_loop() → TCP connect → Prefill accept-loop
                                          → _PendingTransfer.serve()

Prefill send_kv_chunk() → TCPKVSender.send() → _PendingTransfer.ready()
                                    → _stream_kv() → GPU→CPU→TCP → socket.sendall()

Decode _recv_loop() → TCP recv → _recv_kv() → socket.recv()
                                    → _write_pages_to_gpu() → cuMemcpyHtoD
```

## 7. 下一步计划

### P0: 解决 Scheduler 线程 segfault

1. 启用 core dump: `ulimit -c unlimited` + `echo '/tmp/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern`
2. 用 gdb attach 到 scheduler 进程在 crash 前分析
3. 检查 sglang 上游是否有相关 issue/fix

### P1: 清理 debug 日志

端到端验证通过后，将 `logger.warning` debug 日志降级为 `logger.debug` 或移除。

### P1: 性能优化

1. 评估 TCP 传输延迟，考虑 pipeline mode 优化
2. 评估是否需要 pinned memory 加速 HtoD/DtoH 传输

### P2: 自动化测试

创建一键测试脚本，自动重启服务 + 发送请求 + 收集日志。

## 8. 测试脚本和文档

| 文件 | 说明 |
|------|------|
| `scripts/pd_disagg_test/pd_coordinator.py` | Python PD coordinator（绕过 Rust Router） |
| `scripts/pd_disagg_test/pd_test.sh` | 自动化测试部署脚本 |
| `scripts/pd_disagg_test/remote_worker.sh` | 远程工作进程管理 |
| `scripts/pd_disagg_test/configs/default.sh` | 默认配置 |
| `scripts/pd_disagg_test/PROGRESS_REPORT.md` | 本调试进展报告 |
