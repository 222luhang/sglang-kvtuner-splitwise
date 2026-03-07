#!/bin/bash
# start_prefill_posix.sh - 启动 Prefill 服务（POSIX 后端 + KVTuner 层级量化）

set -e

# ==================== 配置区 ====================

# 模型配置
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"

# KVTuner 层级量化配置
KV_TUNER_LAYER_CONFIG="${KV_TUNER_LAYER_CONFIG:-$HOME/sglang-config/qwen2.5-7b_layer_quant.json}"
KV_TUNER_NBITS_SCALE="${KV_TUNER_NBITS_SCALE:-1.0}"

# KVTuner 量化参数
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
ENABLE_KVTUNER="${ENABLE_KVTUNER:-true}"
KV_TUNER_NBITS_KEY="${KV_TUNER_NBITS_KEY:-4}"
KV_TUNER_NBITS_VALUE="${KV_TUNER_NBITS_VALUE:-4}"
KV_TUNER_RESIDUAL_LENGTH="${KV_TUNER_RESIDUAL_LENGTH:-256}"

# P/D 分离配置 - 使用 POSIX 后端
DISAGGREGATION_MODE="prefill"
DISAGGREGATION_TRANSFER_BACKEND="nixl"
NIXL_BACKEND="POSIX"

# Bootstrap 配置
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

# 日志配置
LOG_DIR="${LOG_DIR:-/tmp/sglang_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prefill_posix_$(date +%Y%m%d_%H%M%S).log"

# ==================== 打印配置 ====================

echo "=========================================="
echo "  Prefill 服务启动 (POSIX 后端)"
echo "=========================================="
echo "模型路径：    $MODEL_PATH"
echo "监听地址：    $HOST:$PORT"
echo "TP/DP 大小：   $TP_SIZE / $DP_SIZE"
echo ""
echo "KV Cache 量化：$KV_CACHE_DTYPE"
echo "KVTuner 启用：  $ENABLE_KVTUNER"
echo "  - Key bits:   $KV_TUNER_NBITS_KEY"
echo "  - Value bits: $KV_TUNER_NBITS_VALUE"
echo "  - 残差长度：  $KV_TUNER_RESIDUAL_LENGTH"
echo "  - 层级配置：  $KV_TUNER_LAYER_CONFIG"
echo ""
echo "P/D 分离：     $DISAGGREGATION_MODE"
echo "传输后端：    $DISAGGREGATION_TRANSFER_BACKEND"
echo "NIXL 后端：    $NIXL_BACKEND"
echo "Bootstrap 端口：$BOOTSTRAP_PORT"
echo ""
echo "日志文件：    $LOG_FILE"
echo "=========================================="

# ==================== 环境变量 ====================

export SGLANG_DISAGGREGATION_NIXL_BACKEND="$NIXL_BACKEND"
export SGLANG_DISAGGREGATION_DECODE_ENABLE_FAKE_AUTO="false"
export KVTUNER_PREFILL_NBITS_SCALE="${KV_TUNER_NBITS_SCALE}"

# ==================== 构建命令 ====================

CMD="$HOME/.venv/bin/python3 -m sglang.launch_server"
CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --tp-size $TP_SIZE"
CMD="$CMD --dp-size $DP_SIZE"
CMD="$CMD --disaggregation-mode $DISAGGREGATION_MODE"
CMD="$CMD --disaggregation-transfer-backend $DISAGGREGATION_TRANSFER_BACKEND"
CMD="$CMD --disaggregation-bootstrap-port $BOOTSTRAP_PORT"

# KV Cache 量化
if [ "$KV_CACHE_DTYPE" != "none" ] && [ -n "$KV_CACHE_DTYPE" ]; then
    CMD="$CMD --kv-cache-dtype $KV_CACHE_DTYPE"
fi

# KVTuner 量化
if [ "$ENABLE_KVTUNER" = "true" ]; then
    CMD="$CMD --enable-kvtuner-quant"
    
    # 优先使用层级量化配置
    if [ -n "$KV_TUNER_LAYER_CONFIG" ] && [ -f "$KV_TUNER_LAYER_CONFIG" ]; then
        CMD="$CMD --kvtuner-layer-config $KV_TUNER_LAYER_CONFIG"
        echo "✓ 使用层级量化配置：$KV_TUNER_LAYER_CONFIG"
    else
        CMD="$CMD --kvtuner-nbits-key $KV_TUNER_NBITS_KEY"
        CMD="$CMD --kvtuner-nbits-value $KV_TUNER_NBITS_VALUE"
        CMD="$CMD --kvtuner-residual-length $KV_TUNER_RESIDUAL_LENGTH"
        echo "✓ 使用统一量化配置：${KV_TUNER_NBITS_KEY}-bit"
    fi
fi

# 优化选项
CMD="$CMD --mem-fraction-static 0.8"
CMD="$CMD --max-running-requests 256"
CMD="$CMD --schedule-conservativeness 1.0"
CMD="$CMD --disable-custom-all-reduce"

echo ""
echo "启动命令:"
echo "$CMD"
echo ""

# ==================== 启动服务 ====================

echo "正在启动服务..."
nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
PID=$!

echo "✓ 服务已启动 (PID: $PID)"
echo ""

# ==================== 等待就绪 ====================

echo "等待服务就绪..."
READY=false
for i in {1..120}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 1
done

if [ "$READY" = true ]; then
    echo "✓ 服务就绪!"
    echo ""
    echo "=========================================="
    echo "  服务信息"
    echo "=========================================="
    
    # 获取并显示关键配置
    curl -s "http://localhost:$PORT/get_server_info" | python3 << 'PYTHON'
import sys, json

try:
    info = json.load(sys.stdin)
    
    print(f"模型：        {info.get('model_path', 'N/A')}")
    print(f"P/D 模式：     {info.get('disaggregation_mode', 'N/A')}")
    print(f"传输后端：    {info.get('disaggregation_transfer_backend', 'N/A')}")
    print(f"KV Cache:     {info.get('kv_cache_dtype', 'N/A')}")
    print(f"KVTuner 启用： {info.get('enable_kvtuner_quant', False)}")
    print(f"KVTuner K:    {info.get('kvtuner_nbits_key', 'N/A')}-bit")
    print(f"KVTuner V:    {info.get('kvtuner_nbits_value', 'N/A')}-bit")
    print(f"层级量化：    {info.get('enable_kvtuner_layer_wise', False)}")
    
    # 显存使用
    mem = info.get('internal_states', [{}])[0].get('memory_usage', {})
    if mem:
        print(f"\n显存使用:")
        print(f"  权重：      {mem.get('weight', 'N/A')} GB")
        print(f"  KV Cache:   {mem.get('kvcache', 'N/A')} GB")
        print(f"  Token 容量：{mem.get('token_capacity', 'N/A')}")
        
        # 计算压缩比
        kvcache = mem.get('kvcache')
        if kvcache:
            baseline = 22.8  # BF16 基线
            compression = baseline / kvcache
            savings = (1 - kvcache / baseline) * 100
            print(f"\n量化效果:")
            print(f"  压缩比：    {compression:.2f}x")
            print(f"  显存节省：  {savings:.1f}%")
    
except Exception as e:
    print(f"解析失败：{e}")
PYTHON
    
    echo ""
    echo "=========================================="
    echo "✓ Prefill 服务启动完成"
    echo "=========================================="
    exit 0
else
    echo "✗ 服务启动超时"
    echo ""
    echo "请检查日志：$LOG_FILE"
    tail -50 "$LOG_FILE"
    exit 1
fi
