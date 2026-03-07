# POSIX 后端部署指南 - KVTuner 层级量化验证

**日期**: 2026-03-01  
**目标**: 使用 POSIX 后端完成 KV Cache 量化压缩传输验证  
**环境**: 无 InfiniBand 网络

---

## 📋 部署步骤

### 步骤 1: 准备层级量化配置文件

确保层级量化配置文件存在：

```bash
# 检查配置文件
ls -la ~/sglang-config/qwen2.5-7b_layer_quant.json

# 如果不存在，创建示例配置
cat > ~/sglang-config/qwen2.5-7b_layer_quant.json << 'EOF'
{
  "model": "Qwen2.5-7B",
  "layer_bits": {
    "0": 8,
    "1": 6,
    "2": 4,
    "3": 4,
    "4": 4,
    "5": 4,
    "6": 4,
    "7": 4,
    "8": 4,
    "9": 4,
    "10": 4,
    "11": 4,
    "12": 4,
    "13": 4,
    "14": 4,
    "15": 4,
    "16": 4,
    "17": 4,
    "18": 4,
    "19": 4,
    "20": 4,
    "21": 4,
    "22": 4,
    "23": 4,
    "24": 4,
    "25": 4,
    "26": 4,
    "27": 8
  },
  "residual_length": 256,
  "axis_key": 0,
  "axis_value": 0,
  "q_group_size": 64
}
EOF
```

---

### 步骤 2: 在 Prefill 节点启动服务

**节点**: 10.60.6.75:30000

```bash
# 方法 A: 使用启动脚本（推荐）
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation
./start_prefill_posix.sh

# 方法 B: 手动启动
export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
~/.venv/bin/python3 -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 2 \
  --dp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-kvtuner-quant \
  --kvtuner-layer-config ~/sglang-config/qwen2.5-7b_layer_quant.json \
  --mem-fraction-static 0.8 \
  --max-running-requests 256 \
  --disable-custom-all-reduce \
  > /tmp/sglang_prefill_posix.log 2>&1 &
```

**验证启动**:
```bash
# 等待 30 秒后检查
curl -s http://10.60.6.75:30000/health
curl -s http://10.60.6.75:30000/get_server_info | python3 -m json.tool | grep -E "kvtuner|disaggregation"
```

---

### 步骤 3: 在 Decode 节点启动服务

**节点**: 10.60.19.152:30001

```bash
# 方法 A: 使用启动脚本（推荐）
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation
./start_decode_posix.sh

# 方法 B: 手动启动
export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
~/.venv/bin/python3 -m sglang.launch_server \
  --model-path /data/Qwen/Qwen2.5-7B \
  --host 0.0.0.0 \
  --port 30001 \
  --tp-size 2 \
  --dp-size 1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-kvtuner-quant \
  --kvtuner-layer-config ~/sglang-config/qwen2.5-7b_layer_quant.json \
  --kvtuner-nbits-scale 1.2 \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --disable-custom-all-reduce \
  > /tmp/sglang_decode_posix.log 2>&1 &
```

**验证启动**:
```bash
# 等待 30 秒后检查
curl -s http://10.60.19.152:30001/health
curl -s http://10.60.19.152:30001/get_server_info | python3 -m json.tool | grep -E "kvtuner|disaggregation"
```

---

### 步骤 4: 验证 KVTuner 配置

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation
python3 verify_kvtuner_quant.py
```

**预期输出**:
```
======================================================================
  Prefill 节点 - KVTuner 配置检查
======================================================================
KV Cache 类型：     fp8_e5m2
KVTuner 启用：      True
Key bits:          4
Value bits:        4

层级量化:
  启用：           True
  配置文件：       /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json
  层级配置：
    Layer 0: 8-bit
    Layer 1: 6-bit
    Layer 2: 4-bit
    ...

显存使用:
  权重：      7.22 GB
  KV Cache:   11.4 GB
  Token 容量：854209

量化效果:
  压缩比：    2.00x
  显存节省：  50.0%

✓ 压缩效果符合预期
```

---

### 步骤 5: 运行完整 P/D 流程测试

```bash
cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation
python3 test_complete_pd_flow.py
```

**预期结果**:
```
步骤 1: 获取 Bootstrap Room ID
✓ Room ID: XXXXXXXXXX
  Prefill 节点：10.60.6.75:30000
  Decode 节点：10.60.19.152:30001

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

## 🔍 关键验证点

### 1. 层级量化配置检查

```bash
curl -s http://10.60.6.75:30000/get_server_info | python3 << 'EOF'
import sys, json
info = json.load(sys.stdin)

print("KVTuner 配置:")
print(f"  启用：         {info.get('enable_kvtuner_quant')}")
print(f"  层级量化：     {info.get('enable_kvtuner_layer_wise')}")
print(f"  配置文件：     {info.get('kvtuner_layer_config_file')}")
print(f"  Key bits:      {info.get('kvtuner_nbits_key')}")
print(f"  Value bits:    {info.get('kvtuner_nbits_value')}")

# 检查层级配置
layer_bits = info.get('kvtuner_layer_bits')
if layer_bits:
    print(f"\n层级比特配置:")
    if isinstance(layer_bits, dict):
        for layer, bits in sorted(layer_bits.items(), key=lambda x: int(x[0]))[:5]:
            print(f"  Layer {layer}: {bits}-bit")
EOF
```

### 2. 传输后端检查

```bash
# Prefill 节点
curl -s http://10.60.6.75:30000/get_server_info | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Prefill - Transfer Backend:', d.get('disaggregation_transfer_backend'))
print('Prefill - NIXL Backend:', 'POSIX (env: SGLANG_DISAGGREGATION_NIXL_BACKEND)')
"

# Decode 节点
curl -s http://10.60.19.152:30001/get_server_info | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Decode - Transfer Backend:', d.get('disaggregation_transfer_backend'))
print('Decode - NIXL Backend:', 'POSIX (env: SGLANG_DISAGGREGATION_NIXL_BACKEND)')
"
```

### 3. 显存压缩效果

| 节点 | 预期 KV Cache | 预期压缩比 |
|------|--------------|-----------|
| Prefill (4-bit) | ~5-6 GB | ~4x |
| Decode (8-bit) | ~11-12 GB | ~2x |

---

## 📁 相关文件

| 文件 | 说明 |
|------|------|
| `start_prefill_posix.sh` | Prefill 启动脚本 |
| `start_decode_posix.sh` | Decode 启动脚本 |
| `verify_kvtuner_quant.py` | 量化配置验证脚本 |
| `test_complete_pd_flow.py` | 完整 P/D 流程测试 |
| `NIXL_FIX_REPORT.md` | NIXL 技术分析 |

---

## ⚠️ 常见问题

### Q1: 服务启动失败

**检查日志**:
```bash
tail -100 /tmp/sglang_prefill_posix.log
tail -100 /tmp/sglang_decode_posix.log
```

**常见错误**:
- NIXL 库未安装 → 检查 Python 环境
- 端口被占用 → 更改端口或停止旧服务
- 模型路径错误 → 确认 `/data/Qwen/Qwen2.5-7B` 存在

### Q2: 层级量化未生效

**检查**:
1. 配置文件路径是否正确
2. 配置文件格式是否正确（JSON）
3. 启动命令是否包含 `--kvtuner-layer-config`

### Q3: 传输失败

**检查**:
1. Bootstrap 服务是否运行 (`curl http://10.60.179.106:8998/health`)
2. 节点间网络是否通畅
3. NIXL POSIX 后端是否正确初始化

---

## 📊 性能基准（预期）

| 指标 | 预期值 |
|------|--------|
| Prefill 延迟 | 100-300 ms |
| Decode 延迟 | 50-150 ms |
| KV 传输延迟 | 10-50 ms (取决于网络) |
| 显存节省 | 50-75% |
| 压缩比 | 2-4x |

---

**最后更新**: 2026-03-01 17:30 GMT+8  
**状态**: 待部署
