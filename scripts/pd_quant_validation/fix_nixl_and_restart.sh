#!/bin/bash
# fix_nixl_and_restart.sh - 修复 NIXL 问题并重启 P/D 服务
# 
# 使用方法:
#   ./fix_nixl_and_restart.sh [fake_auto|posix]
#
# 参数:
#   fake_auto - 启用 Fake Auto 模式（推荐用于测试）
#   posix     - 使用 POSIX 后端（推荐用于生产）
#
# 默认：fake_auto

set -e

MODE="${1:-fake_auto}"
PREFILL_NODE="10.60.6.75"
DECODE_NODE="10.60.19.152"
BOOTSTRAP_NODE="10.60.179.106"

echo "=========================================="
echo "NIXL 修复脚本"
echo "模式：$MODE"
echo "=========================================="

# 生成启动命令
generate_prefill_cmd() {
    local extra_args=""
    
    if [ "$MODE" = "fake_auto" ]; then
        extra_args="--disaggregation-decode-enable-fake-auto true"
    elif [ "$MODE" = "posix" ]; then
        export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
    fi
    
    cat << EOF
cd /home/ubuntu/sglang-kvtuner-splitwise
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=$([ "$MODE" = "fake_auto" ] && echo "true" || echo "false")
~/.venv/bin/python3 -m sglang.launch_server \\
  --model-path /data/Qwen/Qwen2.5-7B \\
  --host 0.0.0.0 \\
  --port 30000 \\
  --tp-size 2 \\
  --dp-size 1 \\
  --disaggregation-mode prefill \\
  --disaggregation-transfer-backend nixl \\
  $extra_args \\
  --kv-cache-dtype fp8_e5m2 \\
  --enable-kvtuner-quant \\
  --kvtuner-nbits-key 4 \\
  --kvtuner-nbits-value 4 \\
  --kvtuner-residual-length 256 \\
  --mem-fraction-static 0.8 \\
  --max-running-requests 256 \\
  --disable-custom-all-reduce \\
  > /tmp/sglang_prefill_$(date +%Y%m%d_%H%M%S).log 2>&1 &
EOF
}

generate_decode_cmd() {
    local extra_args=""
    
    if [ "$MODE" = "fake_auto" ]; then
        extra_args="--disaggregation-decode-enable-fake-auto true"
    elif [ "$MODE" = "posix" ]; then
        export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
    fi
    
    cat << EOF
cd /home/ubuntu/sglang-kvtuner-splitwise
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=$([ "$MODE" = "fake_auto" ] && echo "true" || echo "false")
~/.venv/bin/python3 -m sglang.launch_server \\
  --model-path /data/Qwen/Qwen2.5-7B \\
  --host 0.0.0.0 \\
  --port 30001 \\
  --tp-size 2 \\
  --dp-size 1 \\
  --disaggregation-mode decode \\
  --disaggregation-transfer-backend nixl \\
  $extra_args \\
  --kv-cache-dtype fp8_e5m2 \\
  --enable-kvtuner-quant \\
  --kvtuner-nbits-key 8 \\
  --kvtuner-nbits-value 8 \\
  --kvtuner-residual-length 32 \\
  --mem-fraction-static 0.8 \\
  --max-running-requests 128 \\
  --disable-custom-all-reduce \\
  > /tmp/sglang_decode_$(date +%Y%m%d_%H%M%S).log 2>&1 &
EOF
}

echo ""
echo "⚠️  注意：由于无法 SSH 访问远程节点，请手动在以下节点执行命令："
echo ""

echo "=========================================="
echo "在 Prefill 节点 ($PREFILL_NODE) 执行:"
echo "=========================================="
echo ""
generate_prefill_cmd
echo ""

echo "=========================================="
echo "在 Decode 节点 ($DECODE_NODE) 执行:"
echo "=========================================="
echo ""
generate_decode_cmd
echo ""

echo "=========================================="
echo "验证步骤"
echo "=========================================="
echo ""
echo "1. 等待 30 秒让服务启动"
echo "2. 检查 Prefill 节点:"
echo "   curl -s http://$PREFILL_NODE:30000/health"
echo ""
echo "3. 检查 Decode 节点:"
echo "   curl -s http://$DECODE_NODE:30001/health"
echo ""
echo "4. 验证 NIXL 配置:"
echo "   curl -s http://$PREFILL_NODE:30000/get_server_info | python3 -m json.tool | grep -E 'fake_auto|transfer_backend'"
echo ""
echo "5. 运行测试:"
echo "   cd /home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise/scripts/pd_quant_validation"
echo "   python3 test_complete_pd_flow.py"
echo ""

# 如果可以通过 SSH 访问，自动执行
if command -v ssh &> /dev/null; then
    echo ""
    echo "尝试自动执行（需要 SSH 访问）..."
    echo ""
    
    # 检查 SSH 访问
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PREFILL_NODE "echo test" &>/dev/null; then
        echo "✓ SSH 访问 Prefill 节点成功"
        
        # 停止旧服务
        ssh ubuntu@$PREFILL_NODE "pkill -f 'sglang.launch_server.*disaggregation-mode prefill' || true"
        
        # 启动新服务
        ssh ubuntu@$PREFILL_NODE "$(generate_prefill_cmd)"
        
        echo "✓ Prefill 服务已重启"
    else
        echo "✗ 无法 SSH 访问 Prefill 节点，请手动执行上面的命令"
    fi
    
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$DECODE_NODE "echo test" &>/dev/null; then
        echo "✓ SSH 访问 Decode 节点成功"
        
        # 停止旧服务
        ssh ubuntu@$DECODE_NODE "pkill -f 'sglang.launch_server.*disaggregation-mode decode' || true"
        
        # 启动新服务
        ssh ubuntu@$DECODE_NODE "$(generate_decode_cmd)"
        
        echo "✓ Decode 服务已重启"
    else
        echo "✗ 无法 SSH 访问 Decode 节点，请手动执行上面的命令"
    fi
fi

echo ""
echo "=========================================="
echo "修复脚本完成"
echo "=========================================="
