# 动态调度器 - 实现 vLLM PR #2809 思想

## 📋 概述

本调度器实现了 vLLM PR #2809 中关于**跨机器 P/D 分离调度**的核心算法，包括：

1. ✅ **动态负载均衡** - 根据节点实时负载分配请求
2. ✅ **节点健康检查** - 自动检测和排除故障节点
3. ✅ **Prefix-Aware 路由** - 利用 KV Cache 位置优化路由
4. ✅ **弹性节点管理** - 动态添加/移除 Prefill/Decode 节点

**不包含**：通信优化部分（如 KV Cache 压缩、零拷贝传输等）

---

## 🎯 核心算法

### 1. 节点选择算法

```python
def select_prefill_node(prompt_hash):
    # 1. Prefix-Aware: 检查是否有缓存
    if prompt_hash in prefix_cache:
        cached_node = prefix_cache[prompt_hash]
        if cached_node is healthy:
            return cached_node  # Cache Hit
    
    # 2. Load-Based: 选择负载最低的节点
    return node with minimum load
```

### 2. 健康评分算法

```python
health_score = 100 
    - load * 30              # 负载惩罚 (0-30 分)
    - kv_cache_usage * 20    # 显存惩罚 (0-20 分)
    - latency_penalty        # 延迟惩罚 (0-20 分)
    + prefix_hit_rate * 10   # 命中率奖励 (0-10 分)
```

### 3. 节点状态机

```
HEALTHY ──load > threshold──> OVERLOADED
   │                              │
   │ timeout                      │ load < threshold*0.5
   ↓                              ↓
UNHEALTHY <─────────────────────┘
```

---

## 🚀 快速开始

### 1. 启动调度器

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation

python3 dynamic_scheduler.py \
    --prefill 10.60.6.75:30000,10.60.9.62:30000 \
    --decode 10.60.19.152:30001,10.60.176.217:30001 \
    --host 0.0.0.0 \
    --port 9000
```

### 2. 发送请求

```bash
# 通过调度器发送请求
curl -X POST http://localhost:9000/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 256,
    "temperature": 0.7
  }'

# 查看调度器状态
curl http://localhost:9000/status | python3 -m json.tool
```

### 3. 动态添加节点

```bash
# 添加新的 Prefill 节点
curl -X POST http://localhost:9000/node/add \
  -H "Content-Type: application/json" \
  -d '{"url": "10.60.6.75:30002", "role": "prefill"}'

# 移除节点
curl -X POST http://localhost:9000/node/remove \
  -H "Content-Type: application/json" \
  -d '{"url": "10.60.9.62:30000", "role": "prefill"}'
```

---

## 📊 调度器架构

```
┌─────────────────────────────────────────────────────────┐
│                   Dynamic Scheduler                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Request   │  │    Load     │  │   Health    │     │
│  │   Router    │  │  Balancer   │  │   Checker   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│  ┌──────┴────────────────┴────────────────┴──────┐     │
│  │           Node Manager & Metrics              │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
         │                │                │
         │                │                │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │Prefill 1│      │Prefill 2│      │  ...    │
    │10.60.6.75│      │10.60.9.62│      │         │
    └─────────┘      └─────────┘      └─────────┘
    
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │  Decode 1  │   │  Decode 2  │   │    ...     │
    │10.60.19.152│   │10.60.176.217│   │            │
    └────────────┘   └────────────┘   └────────────┘
```

---

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prefill` | 必需 | Prefill 节点列表（逗号分隔） |
| `--decode` | 必需 | Decode 节点列表（逗号分隔） |
| `--host` | 0.0.0.0 | 调度器监听地址 |
| `--port` | 9000 | 调度器监听端口 |
| `--heartbeat-interval` | 5.0 | 心跳检查间隔（秒） |
| `--heartbeat-timeout` | 15.0 | 心跳超时时间（秒） |
| `--load-threshold` | 0.8 | 过载阈值（0-1） |
| `--enable-prefix-aware` | True | 启用 Prefix-Aware 路由 |

---

## 📈 监控指标

### 节点状态

```json
{
  "prefill_nodes": {
    "10.60.6.75:30000": {
      "status": "healthy",
      "load": 0.45,
      "health_score": 85.5,
      "prefix_hit_rate": 0.78
    }
  },
  "decode_nodes": {
    "10.60.19.152:30001": {
      "status": "healthy",
      "load": 0.32,
      "health_score": 92.1
    }
  },
  "prefix_cache_size": 1234,
  "active_requests": 15
}
```

### 关键指标

- **health_score**: 节点健康评分（0-100）
- **load**: 当前负载（0-1）
- **prefix_hit_rate**: Prefix Cache 命中率
- **kv_cache_usage**: KV Cache 使用率

---

## 🔧 与 SGLang 集成

### 方案 A: 独立调度器（当前实现）

```
Client → Dynamic Scheduler → Prefill/Decode Nodes
```

**优点**:
- 独立部署，不影响现有 SGLang 服务
- 易于测试和调试
- 可以管理多个 SGLang 集群

**缺点**:
- 需要额外的网络跳转
- 无法深度集成 SGLang 内部调度

### 方案 B: 集成到 SGLang Router（推荐）

将调度逻辑集成到 `sglang_router` 中：

```python
# python/sglang/srt/disaggregation/scheduler.py
class PDScheduler:
    def __init__(self, prefill_nodes, decode_nodes):
        self.dynamic_scheduler = DynamicScheduler(
            prefill_nodes, decode_nodes
        )
    
    def route_request(self, request):
        return self.dynamic_scheduler.schedule_request(
            request.prompt,
            request.max_tokens
        )
```

**集成步骤**:

1. 将 `dynamic_scheduler.py` 移动到 `python/sglang/srt/disaggregation/`
2. 修改 `sglang_router/mini_lb.py` 使用新的调度器
3. 添加配置参数支持

---

## 🧪 测试

### 单元测试

```bash
python3 -m pytest test_dynamic_scheduler.py -v
```

### 压力测试

```bash
# 使用 locust 进行压力测试
locust -f locustfile.py --host http://localhost:9000
```

### 验证测试

```bash
# 验证调度器功能
./test_scheduler.sh
```

---

## 📝 与 vLLM PR #2809 的对应关系

| vLLM PR #2809 功能 | SGLang 实现 | 状态 |
|-------------------|------------|------|
| 动态负载均衡 | `DynamicScheduler._select_prefill_node()` | ✅ 已实现 |
| 节点健康检查 | `DynamicScheduler._check_node_health()` | ✅ 已实现 |
| Prefix-Aware 路由 | `prefix_cache` + `_select_prefill_node()` | ✅ 已实现 |
| 弹性节点管理 | `add_node()` / `remove_node()` | ✅ 已实现 |
| KV Cache 压缩 | - | ❌ 不需要（按用户要求） |
| 零拷贝传输 | - | ❌ 不需要（按用户要求） |

---

## 🚦 最佳实践

### 1. 节点部署建议

- **Prefill 节点**: 部署在计算能力强的机器上
- **Decode 节点**: 部署在显存大的机器上
- **调度器**: 部署在单独的控制节点

### 2. 参数调优

```bash
# 高吞吐场景
--load-threshold 0.9 --heartbeat-interval 3.0

# 低延迟场景
--load-threshold 0.7 --heartbeat-interval 2.0

# 长文本场景（Prefix Cache 更重要）
--enable-prefix-aware true
```

### 3. 监控告警

```python
# 监控节点健康
if health_score < 50:
    alert("Node health degraded")

# 监控负载
if load > 0.9:
    alert("Node overloaded")

# 监控命中率
if prefix_hit_rate < 0.5:
    alert("Prefix cache efficiency low")
```

---

## 🔮 未来改进

1. **更智能的负载均衡**: 考虑请求类型、长度等因素
2. **预测性扩缩容**: 基于历史负载预测节点需求
3. **多集群支持**: 跨数据中心调度
4. **优先级调度**: 支持不同优先级的请求

---

## 📞 故障排除

### 问题：节点频繁被标记为 unhealthy

**解决方案**:
```bash
# 增加心跳超时时间
--heartbeat-timeout 30.0

# 检查网络连接
ping <node_ip>
```

### 问题：Prefix Cache 命中率低

**解决方案**:
```bash
# 检查是否有重复的 prompt
# 增加调度器运行时间（缓存需要积累）
# 考虑使用更长的 prompt hash
```

### 问题：负载不均衡

**解决方案**:
```bash
# 降低负载阈值
--load-threshold 0.6

# 减少心跳间隔
--heartbeat-interval 2.0
```

---

## 📚 参考资料

- vLLM PR #2809: Disaggregated Serving
- SGLang P/D Disaggregation: `docs/advanced_features/pd_disaggregation.md`
- Mooncake Transfer Engine: KV Cache 传输优化

---

**版本**: 1.0  
**更新日期**: 2026-02-28
