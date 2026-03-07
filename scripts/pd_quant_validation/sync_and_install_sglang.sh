#!/bin/bash
# sync_and_install_sglang.sh - 同步最新代码到所有节点并安装
#
# 使用说明:
#   1. 在本地机器运行此脚本
#   2. 会自动打包代码并同步到各节点
#   3. 在各节点上执行安装

set -e

echo "=========================================="
echo "  SGLang KVTuner 代码同步与安装"
echo "=========================================="
echo ""

# 节点列表
PREFILL_NODES=("10.60.6.75" "10.60.9.62")
DECODE_NODES=("10.60.19.152" "10.60.176.217")
BOOTSTRAP_NODE="10.60.179.106"

# 代码路径
LOCAL_CODE="/home/ubuntu/.openclaw/workspace/sglang-kvtuner-splitwise"
REMOTE_CODE="/home/ubuntu/sglang-kvtuner-splitwise"
OLD_CODE="/home/ubuntu/sglang-kvtuner-adapter"

# 打包代码
echo "[1/5] 打包本地代码..."
cd /home/ubuntu/.openclaw/workspace
tar -czf /tmp/sglang-kvtuner-splitwise.tar.gz \
  --exclude='.git' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  sglang-kvtuner-splitwise
echo "✓ 代码包已创建：/tmp/sglang-kvtuner-splitwise.tar.gz"
echo ""

# 同步到节点函数
sync_to_node() {
  local node=$1
  local role=$2
  
  echo "----------------------------------------"
  echo "同步到 $role 节点：$node"
  echo "----------------------------------------"
  
  # 传输代码包
  echo "  [1/3] 传输代码包..."
  scp /tmp/sglang-kvtuner-splitwise.tar.gz ubuntu@$node:/tmp/
  
  # 执行安装脚本
  echo "  [2/3] 执行安装..."
  ssh ubuntu@$node << 'SSH_SCRIPT'
set -e

REMOTE_CODE="/home/ubuntu/sglang-kvtuner-splitwise"
OLD_CODE="/home/ubuntu/sglang-kvtuner-adapter"

# 停止旧服务
echo "  停止旧服务..."
pkill -f "sglang.launch_server" || true
sleep 2

# 删除旧代码
if [ -d "$OLD_CODE" ]; then
  echo "  删除旧代码：$OLD_CODE"
  rm -rf "$OLD_CODE"
fi

# 解压新代码
echo "  解压新代码..."
cd /home/ubuntu
if [ -d "$REMOTE_CODE" ]; then
  rm -rf "$REMOTE_CODE"
fi
tar -xzf /tmp/sglang-kvtuner-splitwise.tar.gz
mv sglang-kvtuner-splitwise /home/ubuntu/

# 安装依赖
echo "  安装依赖..."
cd "$REMOTE_CODE"
pip3 install -e python/ -q

# 清理
rm -f /tmp/sglang-kvtuner-splitwise.tar.gz

echo "✓ 安装完成"
SSH_SCRIPT
  
  echo "  [3/3] 验证安装..."
  ssh ubuntu@$node "cd $REMOTE_CODE && python3 -c 'import sglang; print(\"  SGLang 版本:\", sglang.__version__)'"
  
  echo "✓ $node 同步完成"
  echo ""
}

# 同步到所有节点
echo "[2/5] 同步到 Prefill 节点..."
for node in "${PREFILL_NODES[@]}"; do
  sync_to_node "$node" "Prefill"
done

echo "[3/5] 同步到 Decode 节点..."
for node in "${DECODE_NODES[@]}"; do
  sync_to_node "$node" "Decode"
done

# 创建层级量化配置文件
echo "[4/5] 创建层级量化配置文件..."
cat > /tmp/qwen2.5-7b_layer_quant.json << 'EOF'
{
  "model": "Qwen2.5-7B",
  "description": "Layer-wise quantization config for Qwen2.5-7B",
  "layer_bits": {
    "0": 8, "1": 8,
    "2": 6, "3": 6,
    "4": 4, "5": 4, "6": 4, "7": 4, "8": 4,
    "9": 4, "10": 4, "11": 4, "12": 4, "13": 4,
    "14": 4, "15": 4, "16": 4, "17": 4, "18": 4,
    "19": 4, "20": 4, "21": 4, "22": 4, "23": 4,
    "24": 6, "25": 6,
    "26": 8, "27": 8
  },
  "residual_length": 256,
  "axis_key": 0,
  "axis_value": 0,
  "q_group_size": 64,
  "asym": false
}
EOF

# 同步配置文件到所有节点
for node in "${PREFILL_NODES[@]}" "${DECODE_NODES[@]}"; do
  echo "  同步配置文件到 $node..."
  scp /tmp/qwen2.5-7b_layer_quant.json ubuntu@$node:/home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json
done
echo "✓ 配置文件已同步"
echo ""

# 创建启动脚本
echo "[5/5] 创建启动脚本..."
cat > /tmp/start_services.sh << 'START_SCRIPT'
#!/bin/bash
set -e

export SGLANG_DISAGGREGATION_NIXL_BACKEND=POSIX
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO=false

# 检查节点角色
HOSTNAME=$(hostname -I | awk '{print $1}')

if [[ "$HOSTNAME" == "10.60.6.75" ]] || [[ "$HOSTNAME" == "10.60.9.62" ]]; then
  # Prefill 节点
  echo "启动 Prefill 服务..."
  cd /home/ubuntu/sglang-kvtuner-splitwise
  ~/.venv/bin/python3 -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --host 0.0.0.0 --port 30000 --tp-size 2 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8998 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-kvtuner-quant \
    --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \
    --kvtuner-nbits-key 4 --kvtuner-nbits-value 4 \
    --mem-fraction-static 0.8 --max-running-requests 256 \
    --disable-custom-all-reduce \
    > /tmp/sglang_prefill_lq.log 2>&1 &
  echo "Prefill 服务已启动"
  
elif [[ "$HOSTNAME" == "10.60.19.152" ]] || [[ "$HOSTNAME" == "10.60.176.217" ]]; then
  # Decode 节点
  echo "启动 Decode 服务..."
  cd /home/ubuntu/sglang-kvtuner-splitwise
  ~/.venv/bin/python3 -m sglang.launch_server \
    --model-path /data/Qwen/Qwen2.5-7B \
    --host 0.0.0.0 --port 30001 --tp-size 2 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --disaggregation-bootstrap-port 8998 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-kvtuner-quant \
    --kvtuner-layer-config /home/ubuntu/sglang-config/qwen2.5-7b_layer_quant.json \
    --kvtuner-nbits-key 8 --kvtuner-nbits-value 8 \
    --mem-fraction-static 0.8 --max-running-requests 128 \
    --disable-custom-all-reduce \
    > /tmp/sglang_decode_lq.log 2>&1 &
  echo "Decode 服务已启动"
fi
START_SCRIPT

chmod +x /tmp/start_services.sh

# 同步启动脚本到所有节点
for node in "${PREFILL_NODES[@]}" "${DECODE_NODES[@]}"; do
  echo "  同步启动脚本到 $node..."
  scp /tmp/start_services.sh ubuntu@$node:/home/ubuntu/start_services.sh
  ssh ubuntu@$node "chmod +x /home/ubuntu/start_services.sh"
done
echo "✓ 启动脚本已同步"
echo ""

echo "=========================================="
echo "  代码同步与安装完成"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 在各节点执行启动脚本:"
echo "     ssh ubuntu@<node> '/home/ubuntu/start_services.sh'"
echo ""
echo "  2. 或使用以下命令批量启动:"
echo "     for node in 10.60.6.75 10.60.9.62 10.60.19.152 10.60.176.217; do"
echo "       ssh ubuntu@\$node '/home/ubuntu/start_services.sh' &"
echo "     done"
echo ""
echo "  3. 等待 30 秒后验证服务"
echo "     curl http://10.60.6.75:30000/health"
echo "     curl http://10.60.19.152:30001/health"
echo ""
echo "  4. 运行测试"
echo "     python3 test_complete_pd_flow.py"
echo ""
