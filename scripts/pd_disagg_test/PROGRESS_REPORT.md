# P/D Disaggregation TCP KV Transfer 调试进展报告

> 最后更新: 2026-04-04

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

## 3. 架构理解（已确认正确）

### TCP KV Transfer 完整流程

```
[Client Request via Router]
        |
        v
[Router] -- 分配 bootstrap_room, 转发请求到 P 和 D
        |
   ┌────┴────┐
   v         v
[Prefill]  [Decode]
   |         |
   |         |-- _resolve_pending_reqs() → ensure_parallel_info() → GET bootstrap server
   |         |-- 获取 Prefill 注册信息 (tcp_port 等)
   |         |-- pop_preallocated() → _create_receiver_and_enqueue() → 创建 TCPKVReceiver
   |         |   TCPKVReceiver.__init__() 调用 update_status(bootstrap_room, WaitingForInput) ★
   |         |   TCPKVReceiver.init() 通过 ZMQ 发送连接描述符给 Prefill ★
   |         |
   |-- 注册到 bootstrap server (PUT, 包含 tcp_port)
   |-- TCPKVSender 创建, 状态 = Bootstrapping
   |-- 等待 Decode 的 ZMQ 消息...
   |         |
   |    [TCPKVManager._process_zmq_msg() 收到 ZMQ]
   |         |-- 创建 _PendingTransfer
   |         |-- 调用 update_status(bootstrap_room, WaitingForInput) ★
   |
   |-- send() 发现有 _PendingTransfer, 开始 TCP 传输
   |-- 状态: WaitingForInput → Transferring → Success
   |
   v
[Decode 收到 KV cache, 继续生成 token]
```

### 关键状态机 (KVPoll)

```
Failed(0) → Bootstrapping(1) → WaitingForInput(2) → Transferring(3) → Success(4)
```

### 请求路由

- **Prefill 侧**: `_add_request_to_queue()` 将请求放入 `disagg_prefill_bootstrap_queue`（不是 `waiting_queue`！）
  - `pop_bootstrapped()` 轮询 sender 状态，只有 `poll() != Bootstrapping` 才移入 bootstrapped 列表
- **Decode 侧**: 请求先进 `pending_reqs`，`_resolve_pending_reqs()` 获取 prefill 信息后创建 TCPKVReceiver

### Event Loop 选择

- `disable_overlap_schedule=False`（默认）→ Prefill 用 `event_loop_overlap_disagg_prefill()`
- `disable_overlap_schedule=True` → Prefill 用 `event_loop_normal_disagg_prefill()`
- 之前调试时加日志加错了 event loop，浪费了不少时间

## 4. 已应用的代码修改

### 4.1 TCPKVReceiver WaitingForInput 转换（唯一保留的修改）

**文件**: `python/sglang/srt/disaggregation/tcp/conn.py` — TCPKVReceiver.__init__()

**问题**: CommonKVReceiver.__init__() 获取 bootstrap 信息后不会将状态从 Bootstrapping 转换为 WaitingForInput。Decode 侧创建 receiver 后，`pop_preallocated()` 调用 `kv_receiver.init()` 发送 ZMQ，但 sender 端的 `send()` 方法在等待 `_PendingTransfer` 存在。如果 Decode 端的 receiver 初始化在 ZMQ 消息发送前没有正确更新状态，可能导致时序问题。

**修复**: 在 TCPKVReceiver.__init__() 中，获取 bootstrap 信息后立即调用 `update_status(bootstrap_room, KVPoll.WaitingForInput)`，与 MooncakeKVReceiver 保持一致的模式。

```python
def __init__(self, mgr, bootstrap_addr, bootstrap_room=None, prefill_dp_rank=None):
    super().__init__(
        mgr=mgr,
        bootstrap_addr=bootstrap_addr,
        bootstrap_room=bootstrap_room,
        prefill_dp_rank=prefill_dp_rank,
    )
    # Transition from Bootstrapping to WaitingForInput now that bootstrap
    # info has been fetched (same pattern as MooncakeKVReceiver).
    if self.bootstrap_infos is not None:
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
    self._transfer_done = threading.Event()
    self._transfer_ok = True
```

## 5. 已验证的中间结果

| 验证项 | 结果 |
|--------|------|
| gpu1 ↔ gpu2 网络连通性 | ✅ 通过 |
| ZMQ 端口（默认 20123）可达 | ✅ 通过 |
| TCP 端口可达 | ✅ 通过 |
| Bootstrap server 返回正确 tcp_port 字段 | ✅ 通过 |
| Prefill 服务启动 + /health 探活 | ✅ 通过 |
| Decode 服务启动 + /health 探活 | ✅ 通过 |
| Prefill 正确注册到 bootstrap server | ✅ 通过 |
| Bootstrap server 查询 Decode 侧 | ✅ 通过 |
| TCP backend 代码结构正确 | ✅ 通过 |
| Coordinator 发送请求到 P/D | ✅ 请求到达 |

## 6. 主要阻塞问题（未解决）

### 6.1 ★★★ Decode Scheduler 线程崩溃（核心阻塞）

**现象**:
- `--enable-overlap-schedule`（默认）: Decode 的 scheduler 线程在 warmup 后变成 `<defunct>` zombie。无 Python traceback，无 core dump，静默 native segfault。
- `--disable-overlap-schedule`: scheduler 线程存活，但 CPU 86% 空转，不处理外部请求。ZMQ recv 无数据。

**影响**:
- Decode scheduler 崩溃 → 不处理请求 → 不创建 TCPKVReceiver → 不发送 ZMQ 描述符 → Prefill sender 永远停留在 Bootstrapping 状态 → 整个 KV transfer 流程死锁

**排查记录**:
1. 用 `fake` backend 测试 Decode: scheduler 存活，但发请求报 `AttributeError: 'FakeKVManager' object has no attribute 'prefill_info_table'`（scheduler 没崩但功能不完整）
2. 用 `tcp` backend + `--disable-overlap-schedule`: scheduler 存活但空转，日志无任何请求处理痕迹
3. 尝试在 Decode 上启用 core dump (`ulimit -c unlimited`)，SSH 连接超时

**初步判断**: 这可能是 sglang 分支的 pre-existing 问题，不是 TCP backend 修改导致的。需要进一步验证。

### 6.2 Prefill Sender 在 Bootstrapping 状态卡住

**现象**: Coordinator 发送请求后，Prefill 侧的 sender 状态始终为 Bootstrapping。

**根因**: 这是 6.1 的直接后果 — Decode 没有发送 ZMQ，`_process_zmq_msg()` 没有被触发，没有 `_PendingTransfer` 被创建。

## 7. 已排除的误判

| 误判 | 实际情况 |
|------|----------|
| "请求没有进入 waiting_queue 是 bug" | 不是 bug，prefill 模式下请求进 `bootstrap_queue`，这是设计如此 |
| "Debug 日志没有输出 = 代码没执行" | 日志用了 `sys.stderr.write()`，输出到 scheduler 进程的 stderr，不被日志重定向捕获 |
| "加了日志但 overlap loop 没执行" | 日志加到了 `event_loop_normal`，实际用的是 `event_loop_overlap` |
| "巨大的日志文件 (973K 行) 是异常" | overlap event loop 轮询极快，每次迭代都打日志，几秒就 973K 行 |

## 8. 下一步建议

### 优先级 P0: 解决 Decode Scheduler 崩溃

1. **验证是否为 pre-existing 问题**: 在原版 sglang（不含 TCP 修改）上用 decode disaggregation 模式启动，看 scheduler 是否同样崩溃
2. **启用 core dump 并分析**:
   ```bash
   # 在 gpu2 上
   ulimit -c unlimited
   echo '/tmp/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern
   # 启动 Decode，等崩溃后用 gdb 分析 core
   ```
3. **检查 dmesg**: `dmesg | tail -20` 看 segfault 信息
4. **降低 scheduler 频率**: `--scheduler-recv-interval 2` 减慢循环
5. **检查 sglang 分支更新**: 看是否有更新的 commit 修复了此问题
6. **尝试 Rust Router**: 当前用 Python coordinator 绕过 Router，原版 Rust Router 可能有不同的行为

### 优先级 P1: 端到端验证

一旦 Decode scheduler 稳定：
1. 用 coordinator 发送请求，验证完整 ZMQ → TCP transfer 流程
2. 检查 Prefill 日志：sender 是否从 Bootstrapping → WaitingForInput → Transferring → Success
3. 检查 Decode 日志：receiver 是否成功接收 KV cache
4. 验证最终生成的 token 正确性

### 优先级 P2: 自动化测试脚本

按照 `crystalline-mapping-horizon.md` 计划创建 `pd_test.sh` 自动化脚本。

## 9. 参考文件

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/disaggregation/tcp/conn.py` | TCP backend 核心实现 (~1180 行) |
| `python/sglang/srt/disaggregation/prefill.py` | Prefill 侧事件循环和 bootstrap queue |
| `python/sglang/srt/disaggregation/decode.py` | Decode 侧 prealloc queue 和请求处理 |
| `python/sglang/srt/managers/scheduler.py` | 调度器，请求路由，event loop 选择 |
| `python/sglang/srt/disaggregation/common/conn.py` | 公共 KV 管理器基类 |
| `python/sglang/srt/disaggregation/base/conn.py` | KVPoll 状态枚举定义 |
| `scripts/pd_disagg_test/pd_coordinator.py` | Python PD coordinator（绕过 Rust Router） |
