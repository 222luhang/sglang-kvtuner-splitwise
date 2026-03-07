#!/bin/bash
# restart_with_mooncake.sh - 使用 Mooncake 后端重启所有服务

set -e

echo "=========================================="
echo "  使用 Mooncake 后端重启 P/D 服务"
echo "=========================================="
echo ""

# 节点列表
PREFILL_NODES=("10.60.6.75" "10.60.9.62")
DECODE_NODES=("10.60.19.152" "10.60.176.217")

# 重启 Prefill 节点
echo "[1/2] 重启 Prefill 节点..."
for node in "${PREFILL_NODES[@]}"; do
  echo "  重启 $node..."
  sshpass -p 'luhang222' ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$node "
    pkill -f sglang || true
    sleep 2
    
    export MOONCAKE_TE_META_DATA_SERVER='http://10.60.179.106:8080/metadata'
    export MOONCAKE_MASTER='10.60.179.106:50051'
    export MOONCAKE_PROTOCOL='tcp'
    export MOONCAKE_GLOBAL_SEGMENT_SIZE='4GB'
    
    cd /home/ubuntu/sglang-kvtuner-splitwise
    nohup ~/.venv/bin/python3 -m sglang.launch_server \\
      --model-path /data/Qwen/Qwen2.5-7B \\
      --port 30000 --tp-size 2 \\
      --disaggregation-mode prefill \\
      --disaggregation-transfer-backend mooncake \\
      --disaggregation-bootstrap-port 8998 \\
      --kv-cache-dtype fp8_e5m2 \\
      --enable-kvtuner-quant \\
      --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \\
      --kvtuner-nbits-key 4 --kvtuner-nbits-value 4 \\
      --mem-fraction-static 0.8 --max-running-requests 256 \\
      --disable-custom-all-reduce \\
      > /tmp/sglang_prefill_mooncake.log 2>&1 &
    
    echo '  Prefill started on $node'
  " &
done

wait
echo "✓ Prefill 节点重启完成"
echo ""

# 重启 Decode 节点
echo "[2/2] 重启 Decode 节点..."
for node in "${DECODE_NODES[@]}"; do
  echo "  重启 $node..."
  sshpass -p 'luhang222' ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$node "
    pkill -f sglang || true
    sleep 2
    
    export MOONCAKE_TE_META_DATA_SERVER='http://10.60.179.106:8080/metadata'
    export MOONCAKE_MASTER='10.60.179.106:50051'
    export MOONCAKE_PROTOCOL='tcp'
    export MOONCAKE_GLOBAL_SEGMENT_SIZE='4GB'
    
    cd /home/ubuntu/sglang-kvtuner-splitwise
    nohup ~/.venv/bin/python3 -m sglang.launch_server \\
      --model-path /data/Qwen/Qwen2.5-7B \\
      --port 30001 --tp-size 2 \\
      --disaggregation-mode decode \\
      --disaggregation-transfer-backend mooncake \\
      --disaggregation-bootstrap-port 8998 \\
      --kv-cache-dtype fp8_e5m2 \\
      --enable-kvtuner-quant \\
      --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \\
      --kvtuner-nbits-key 8 --kvtuner-nbits-value 8 \\
      --mem-fraction-static 0.8 --max-running-requests 128 \\
      --disable-custom-all-reduce \\
      > /tmp/sglang_decode_mooncake.log 2>&1 &
    
    echo '  Decode started on $node'
  " &
done

wait
echo "✓ Decode 节点重启完成"
echo ""

echo "=========================================="
echo "  所有服务已重启"
echo "=========================================="
echo ""
echo "等待 60 秒让服务启动..."
sleep 60
echo ""
echo "检查服务状态..."
curl -s http://10.60.6.75:30000/health && echo "✓ Prefill 1 OK" || echo "✗ Prefill 1 Failed"
curl -s http://10.60.19.152:30001/health && echo "✓ Decode 1 OK" || echo "✗ Decode 1 Failed"
