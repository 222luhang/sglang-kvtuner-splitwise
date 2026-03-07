#!/bin/bash
# restart_all_services_with_layer_quant.sh - 重新配置并重启所有服务（层级量化 + POSIX 后端）
#
# 使用说明:
#   1. 在每个节点上分别执行对应的部分
#   2. 或手动执行每个节点的重启命令

set -e

echo "=========================================="
echo "  P/D 分离服务重启 - 层级量化 + POSIX 后端"
echo "=========================================="
echo ""

# 层级量化配置文件
LAYER_CONFIG="$HOME/sglang-config/qwen2.5-7b_layer_quant.json"

# 检查配置文件
if [ ! -f "$LAYER_CONFIG" ]; then
    echo "✗ 层级量化配置文件不存在：$LAYER_CONFIG"
    echo "  请先创建配置文件"
    exit 1
fi

echo "✓ 层级量化配置文件：$LAYER_CONFIG"
echo ""

# ==================== Prefill 节点重启命令 ====================

echo "=========================================="
echo "【Prefill 节点重启命令】"
echo "=========================================="
echo ""
echo "在以下 Prefill 节点执行:"
echo "  - 10.60.6.75"
echo "  - 10.60.9.62"
echo ""
echo "----------------------------------------"
echo "复制以下命令并在每个 Prefill 节点执行:"
echo "----------------------------------------"
cat << 'PREFILL_CMD'
# Prefill 节点重启命令
export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=false

pkill -f "sglang.launch_server.*disaggregation-mode prefill" || true
sleep 2

cd /home/ubuntu/sglang-kvtuner-splitwise
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
  --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \
  --kvtuner-nbits-key 4 \
  --kvtuner-nbits-value 4 \
  --kvtuner-residual-length 256 \
  --mem-fraction-static 0.8 \
  --max-running-requests 256 \
  --disable-custom-all-reduce \
  > /tmp/sglang_prefill_layer_quant.log 2>&1 &

echo "Prefill 服务已启动，等待就绪..."
sleep 30
curl -s http://localhost:30000/health && echo "✓ Prefill 就绪" || echo "✗ Prefill 未就绪"
PREFILL_CMD

echo ""
echo ""

# ==================== Decode 节点重启命令 ====================

echo "=========================================="
echo "【Decode 节点重启命令】"
echo "=========================================="
echo ""
echo "在以下 Decode 节点执行:"
echo "  - 10.60.19.152"
echo "  - 10.60.176.217"
echo ""
echo "----------------------------------------"
echo "复制以下命令并在每个 Decode 节点执行:"
echo "----------------------------------------"
cat << 'DECODE_CMD'
# Decode 节点重启命令
export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=false

pkill -f "sglang.launch_server.*disaggregation-mode decode" || true
sleep 2

cd /home/ubuntu/sglang-kvtuner-splitwise
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
  --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \
  --kvtuner-nbits-key 8 \
  --kvtuner-nbits-value 8 \
  --kvtuner-residual-length 32 \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --disable-custom-all-reduce \
  > /tmp/sglang_decode_layer_quant.log 2>&1 &

echo "Decode 服务已启动，等待就绪..."
sleep 30
curl -s http://localhost:30001/health && echo "✓ Decode 就绪" || echo "✗ Decode 未就绪"
DECODE_CMD

echo ""
echo ""

# ==================== 验证命令 ====================

echo "=========================================="
echo "【验证命令】"
echo "=========================================="
echo ""
echo "在所有节点重启完成后，执行以下验证:"
echo ""
echo "----------------------------------------"
echo "1. 检查 Prefill 节点配置:"
echo "----------------------------------------"
cat << 'VERIFY_PREFILL'
curl -s http://10.60.6.75:30000/get_server_info | python3 << 'EOF'
import sys, json
info = json.load(sys.stdin)
print("Prefill 节点配置:")
print(f"  KVTuner 启用：{info.get('enable_kvtuner_quant')}")
print(f"  层级量化：  {info.get('enable_kvtuner_layer_wise')}")
print(f"  配置文件：  {info.get('kvtuner_layer_config_file')}")
print(f"  Key bits:   {info.get('kvtuner_nbits_key')}")
print(f"  Value bits: {info.get('kvtuner_nbits_value')}")

# 显示层级配置
layer_bits = info.get('kvtuner_layer_bits')
if layer_bits:
    print(f"\n层级量化配置 (共 {len(layer_bits)} 层):")
    if isinstance(layer_bits, dict):
        for layer, bits in sorted(layer_bits.items(), key=lambda x: int(x[0]))[:5]:
            print(f"  Layer {layer}: {bits}-bit")
        print("  ...")
EOF
VERIFY_PREFILL

echo ""
echo "----------------------------------------"
echo "2. 检查 Decode 节点配置:"
echo "----------------------------------------"
cat << 'VERIFY_DECODE'
curl -s http://10.60.19.152:30001/get_server_info | python3 << 'EOF'
import sys, json
info = json.load(sys.stdin)
print("Decode 节点配置:")
print(f"  KVTuner 启用：{info.get('enable_kvtuner_quant')}")
print(f"  层级量化：  {info.get('enable_kvtuner_layer_wise')}")
print(f"  配置文件：  {info.get('kvtuner_layer_config_file')}")
print(f"  Key bits:   {info.get('kvtuner_nbits_key')}")
print(f"  Value bits: {info.get('kvtuner_nbits_value')}")
EOF
VERIFY_DECODE

echo ""
echo "----------------------------------------"
echo "3. 运行完整测试:"
echo "----------------------------------------"
echo "cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation"
echo "python3 test_complete_pd_flow.py"
echo ""

echo "=========================================="
echo "重启脚本完成"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 在 2 个 Prefill 节点执行 Prefill 重启命令"
echo "  2. 在 2 个 Decode 节点执行 Decode 重启命令"
echo "  3. 等待所有节点就绪"
echo "  4. 执行验证命令检查配置"
echo "  5. 运行完整测试"
echo ""
