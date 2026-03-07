# NIXL 传输层问题修复报告

**日期**: 2026-03-01  
**状态**: 🔧 修复中  
**问题**: NIXL KVReceiver Exception 导致 P/D 分离流程失败

---

## 🔍 问题分析

### 错误现象

```
Decode handshake failed for request ... 
with exception NIXL KVReceiver Exception
```

### 根本原因

1. **NIXL 库未正确安装或初始化**
   - 默认后端：`UCX` (需要 InfiniBand/RDMA 硬件支持)
   - 当前环境：无 IB 网络，使用 TCP/Ethernet

2. **配置不匹配**
   - `disaggregation_ib_device: None` - 未配置 IB 设备
   - `disaggregation_decode_enable_fake_auto: False` - 未启用模拟模式

3. **后端选择问题**
   - UCX 后端需要特定的网络硬件
   - 消费级/标准网络环境需要使用 POSIX 或其他后端

---

## ✅ 修复方案

### 方案 1: 启用 Fake Auto 模式（推荐用于测试）

**适用场景**: 开发测试、无 RDMA 硬件环境

**修改启动脚本**:

#### Prefill 节点 (10.60.6.75:30000)
```bash
# 添加环境变量
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=true

# 或在启动命令中添加
python3 -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-decode-enable-fake-auto true \
  ...
```

#### Decode 节点 (10.60.19.152:30001)
```bash
# 添加环境变量
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=true

# 或在启动命令中添加
python3 -m sglang.launch_server \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-decode-enable-fake-auto true \
  ...
```

---

### 方案 2: 切换 NIXL 后端到 POSIX（推荐用于生产）

**适用场景**: 标准 TCP/IP 网络环境

**修改启动脚本**:

```bash
# 设置 NIXL 后端为 POSIX
export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX

# 创建 NIXL 配置文件
cat > /tmp/nixl_config.toml << 'EOF'
[plugin.posix]
use_uring = "true"
active = true

[plugin.gds]
active = false

[plugin.gds_mt]
active = false

[plugin.3fs]
active = false

[plugin.obj]
active = false
EOF

# 设置配置文件路径
export SGLANG_HICACHE_NIXL_CONFIG_FILE=/tmp/nixl_config.toml

# 启动服务
python3 -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  ...
```

---

### 方案 3: 检查 NIXL 安装状态

**在 Prefill 和 Decode 节点上执行**:

```bash
# 检查 NIXL 是否安装
python3 -c "from nixl._api import nixl_agent, nixl_agent_config; print('NIXL installed')"

# 检查可用插件
python3 << 'EOF'
from nixl._api import nixl_agent, nixl_agent_config
config = nixl_agent_config(backends=["POSIX"])
agent = nixl_agent("test", config)
print("Available plugins:", agent.get_plugin_list())
EOF
```

**预期输出**:
```
Available plugins: ['POSIX', 'UCX', ...]
```

---

## 📋 执行步骤

### 步骤 1: 停止当前服务

```bash
# 在 Prefill 节点 (10.60.6.75)
pkill -f "sglang.launch_server.*disaggregation-mode prefill"

# 在 Decode 节点 (10.60.19.152)
pkill -f "sglang.launch_server.*disaggregation-mode decode"
```

### 步骤 2: 应用修复

**选择方案 1 (Fake Auto) 或方案 2 (POSIX)**

#### 方案 1 脚本:
```bash
# Prefill 节点
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=true
~/.venv/bin/python3 -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 --port 30000 \
  --tp-size 2 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-decode-enable-fake-auto true \
  --kv-cache-dtype fp8_e5m2 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4 \
  > /tmp/sglang_prefill.log 2>&1 &

# Decode 节点
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=true
~/.venv/bin/python3 -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 --port 30001 \
  --tp-size 2 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-decode-enable-fake-auto true \
  --kv-cache-dtype fp8_e5m2 \
  --enable-kvtuner-quant \
  --kvtuner-nbits-key 8 \
  --kvtuner-nbits-value 8 \
  > /tmp/sglang_decode.log 2>&1 &
```

### 步骤 3: 验证服务启动

```bash
# 等待服务就绪
sleep 30

# 检查 Prefill 节点
curl -s http://10.60.6.75:30000/health && echo "Prefill OK" || echo "Prefill Failed"

# 检查 Decode 节点
curl -s http://10.60.19.152:30001/health && echo "Decode OK" || echo "Decode Failed"

# 验证配置
curl -s http://10.60.6.75:30000/get_server_info | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Fake Auto:', d.get('disaggregation_decode_enable_fake_auto'))
print('Transfer Backend:', d.get('disaggregation_transfer_backend'))
"
```

### 步骤 4: 重新运行测试

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation
python3 test_complete_pd_flow.py
```

---

## 🎯 预期结果

修复后，测试应该通过：

```
步骤 1: 获取 Bootstrap Room ID
✓ Room ID: XXXXXXXXXX

步骤 2: 发送 Prefill 请求
✓ Prefill 成功
  响应：Hello! I am...
  延迟：XXX ms

步骤 3: 发送 Decode 请求
✓ Decode 成功
  响应：...continued text...
  延迟：XXX ms

✅ 完整 P/D 分离流程测试成功!
```

---

## 📝 技术说明

### NIXL 后端对比

| 后端 | 硬件要求 | 性能 | 适用场景 |
|------|----------|------|----------|
| **UCX** | InfiniBand/RoCE | 最高 | 生产环境，RDMA 网络 |
| **POSIX** | 标准网络 | 中等 | 开发测试，TCP/IP 网络 |
| **GDS** | NVIDIA GPUDirect | 高 | GPU 直存场景 |
| **3FS** | 3FS 文件系统 | 高 | 分布式文件系统 |

### Fake Auto 模式

- **用途**: 开发和测试
- **特点**: 模拟 KV Cache 传输，不进行实际数据复制
- **限制**: 不能用于性能测试或生产环境

---

## 🔗 相关资源

- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [SGLang Disaggregation Docs](https://docs.sglang.ai/)
- [UCX 安装指南](https://openucx.org/downloads/)

---

**最后更新**: 2026-03-01 03:45 GMT+8  
**下一步**: 应用修复方案并重新测试
