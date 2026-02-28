# Bootstrap 服务使用指南

## 📋 概述

Bootstrap 服务为 P/D 分离架构提供 room id 获取功能，解决了之前"Disaggregated request received without bootstrap room id"的问题。

**服务状态**: ✅ 运行中  
**服务地址**: `http://10.60.179.106:8998`  
**部署位置**: `~/kvtuner_offline/bootstrap_service_simple.py`

---

## 🚀 快速开始

### 1. 获取 Bootstrap Room ID

```bash
curl -X POST http://10.60.179.106:8998/bootstrap
```

**响应**:
```json
{
  "success": true,
  "room_id": "855bc1e4-bedc-40fe-82e2-ea53771a55ec",
  "prefill_node": "10.60.6.75:30000",
  "decode_node": "10.60.19.152:30001",
  "expires_in": 300
}
```

### 2. 使用 Room ID 发送 P/D 请求

**Prefill 请求**:
```bash
curl -X POST http://10.60.6.75:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello",
    "sampling_params": {"max_new_tokens": 32},
    "bootstrap_room": "855bc1e4-bedc-40fe-82e2-ea53771a55ec"
  }'
```

**Decode 请求**:
```bash
curl -X POST http://10.60.19.152:30001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Continue",
    "sampling_params": {"max_new_tokens": 32},
    "bootstrap_room": "855bc1e4-bedc-40fe-82e2-ea53771a55ec"
  }'
```

### 3. 完成 Room（可选）

```bash
curl -X POST http://10.60.179.106:8998/bootstrap/855bc1e4-bedc-40fe-82e2-ea53771a55ec/complete
```

---

## 📡 API 参考

### POST /bootstrap

创建新的 bootstrap room。

**请求**:
```bash
curl -X POST http://10.60.179.106:8998/bootstrap
```

**响应**:
```json
{
  "success": true,
  "room_id": "uuid",
  "prefill_node": "node:port",
  "decode_node": "node:port",
  "expires_in": 300
}
```

### GET /bootstrap/{room_id}

获取 room 信息。

**请求**:
```bash
curl http://10.60.179.106:8998/bootstrap/{room_id}
```

**响应**:
```json
{
  "success": true,
  "room": {
    "room_id": "uuid",
    "prefill_node": "node:port",
    "decode_node": "node:port",
    "status": "active",
    "created_at": 1234567890,
    "expires_at": 1234568190
  }
}
```

### POST /bootstrap/{room_id}/complete

标记 room 完成。

**请求**:
```bash
curl -X POST http://10.60.179.106:8998/bootstrap/{room_id}/complete
```

**响应**:
```json
{"success": true}
```

### GET /health

健康检查。

**请求**:
```bash
curl http://10.60.179.106:8998/health
```

**响应**:
```json
{"status": "healthy", "active_rooms": 5}
```

### GET /status

服务状态。

**请求**:
```bash
curl http://10.60.179.106:8998/status
```

**响应**:
```json
{
  "active_rooms": 5,
  "prefill_nodes": ["10.60.6.75:30000", "10.60.9.62:30000"],
  "decode_nodes": ["10.60.19.152:30001", "10.60.176.217:30001"],
  "node_load": {...}
}
```

---

## 🔧 配置选项

### 启动参数

```bash
python3 bootstrap_service_simple.py \
    --host 0.0.0.0 \
    --port 8998 \
    --prefill-nodes "10.60.6.75:30000,10.60.9.62:30000" \
    --decode-nodes "10.60.19.152:30001,10.60.176.217:30001" \
    --room-ttl 300
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 8998 | 监听端口 |
| `--prefill-nodes` | - | Prefill 节点列表（逗号分隔） |
| `--decode-nodes` | - | Decode 节点列表（逗号分隔） |
| `--room-ttl` | 300 | Room 存活时间（秒） |

---

## 💡 使用示例

### Python 客户端

```python
import requests

# 1. 获取 Room ID
response = requests.post("http://10.60.179.106:8998/bootstrap")
room_info = response.json()
room_id = room_info['room_id']

# 2. 发送 Prefill 请求
prefill_response = requests.post(
    f"http://{room_info['prefill_node']}/generate",
    json={
        "text": "Hello",
        "sampling_params": {"max_new_tokens": 32},
        "bootstrap_room": room_id
    }
)

# 3. 发送 Decode 请求
decode_response = requests.post(
    f"http://{room_info['decode_node']}/generate",
    json={
        "text": "Continue",
        "sampling_params": {"max_new_tokens": 32},
        "bootstrap_room": room_id
    }
)

# 4. 完成 Room
requests.post(f"http://10.60.179.106:8998/bootstrap/{room_id}/complete")
```

### 负载均衡

Bootstrap 服务自动进行负载均衡：

```python
# 第一次请求
room1 = requests.post("http://10.60.179.106:8998/bootstrap").json()
# 可能分配到 Node 1

# 第二次请求
room2 = requests.post("http://10.60.179.106:8998/bootstrap").json()
# 可能分配到 Node 2（如果 Node 1 负载更高）
```

---

## 📊 服务状态

### 当前配置

| 项目 | 值 |
|------|-----|
| **服务地址** | http://10.60.179.106:8998 |
| **Prefill 节点** | 10.60.6.75:30000, 10.60.9.62:30000 |
| **Decode 节点** | 10.60.19.152:30001, 10.60.176.217:30001 |
| **Room TTL** | 300 秒 |
| **清理间隔** | 60 秒 |

### 监控指标

- **active_rooms**: 当前活跃的 room 数量
- **node_load**: 各节点的负载情况
- **过期清理**: 每分钟自动清理过期 rooms

---

## 🔍 故障排除

### 问题 1: 无法获取 Room ID

**错误**: `Connection refused`

**解决方案**:
```bash
# 检查服务是否运行
ssh ubuntu@10.60.179.106 "ps aux | grep bootstrap_service"

# 检查端口
ssh ubuntu@10.60.179.106 "netstat -tlnp | grep 8998"

# 重启服务
ssh ubuntu@10.60.179.106 "
cd ~/kvtuner_offline
pkill -f bootstrap_service
nohup python3 bootstrap_service_simple.py --port 8998 > /tmp/bootstrap.log 2>&1 &
"
```

### 问题 2: Room 已过期

**错误**: `Room not found or expired`

**解决方案**:
- Room 默认 300 秒后过期
- 重新获取新的 Room ID
- 或增加 `--room-ttl` 参数

### 问题 3: 节点负载不均衡

**现象**: 某些节点负载很高，某些很低

**解决方案**:
- Bootstrap 服务会自动负载均衡
- 检查节点健康状态
- 手动调整节点列表

---

## 📁 相关文件

| 文件 | 位置 | 说明 |
|------|------|------|
| `bootstrap_service_simple.py` | `~/kvtuner_offline/` | Bootstrap 服务 |
| `test_bootstrap_client.py` | `scripts/pd_quant_validation/` | 测试客户端 |
| `BOOTSTRAP_GUIDE.md` | `scripts/pd_quant_validation/` | 本文档 |

---

## 📞 联系与支持

- **服务地址**: http://10.60.179.106:8998
- **日志位置**: `/tmp/bootstrap_service.log`
- **文档**: 本文件

---

**服务启动时间**: 2026-02-28 23:45 GMT+8  
**服务状态**: ✅ 运行正常  
**下一步**: 测试完整的 P/D 分离流程
