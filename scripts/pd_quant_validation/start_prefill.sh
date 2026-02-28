#!/bin/bash
# start_prefill.sh - 启动 Prefill 服务（带 KVTuner 量化）

set -e

# 配置
# 模型路径：使用 /data/Qwen 目录下的 Qwen 模型
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"

# KVTuner 层级量化配置
KV_TUNER_LAYER_CONFIG="${KV_TUNER_LAYER_CONFIG:-~/sglang-config/qwen2.5-7b_layer_quant.json}"
KV_TUNER_NBITS_SCALE="${KV_TUNER_NBITS_SCALE:-1.0}"

# KVTuner 量化配置
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
ENABLE_KVTUNER="${ENABLE_KVTUNER:-true}"
# 如果使用层级量化，这些参数会被配置文件覆盖
KV_TUNER_NBITS_KEY="${KV_TUNER_NBITS_KEY:-4}"
KV_TUNER_NBITS_VALUE="${KV_TUNER_NBITS_VALUE:-4}"
KV_TUNER_RESIDUAL_LENGTH="${KV_TUNER_RESIDUAL_LENGTH:-256}"

# P/D 分离配置
# 无 IB 网络，使用 TCP 传输后端
DISAGGREGATION_MODE="prefill"
DISAGGREGATION_TRANSFER_BACKEND="${DISAGGREGATION_TRANSFER_BACKEND:-nixl}"

# 日志
LOG_DIR="${LOG_DIR:-/tmp/sglang_logs}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动 Prefill 服务"
echo "=========================================="
echo "模型：$MODEL_PATH"
echo "主机：$HOST:$PORT"
echo "TP 大小：$TP_SIZE"
echo "量化：$KV_CACHE_DTYPE"
echo "KVTuner: $ENABLE_KVTUNER (K:${KV_TUNER_NBITS_KEY}-bit, V:${KV_TUNER_NBITS_VALUE}-bit)"
echo "残差长度：$KV_TUNER_RESIDUAL_LENGTH"
echo "传输后端：$DISAGGREGATION_TRANSFER_BACKEND"
echo "=========================================="

# 构建启动命令 - 使用 ~/.venv 环境
PYTHON="~/.venv/bin/python3"
CMD="$PYTHON -m sglang.launch_server"
CMD="$CMD --model-path $MODEL_PATH"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --tp-size $TP_SIZE"
CMD="$CMD --dp-size $DP_SIZE"
CMD="$CMD --disaggregation-mode $DISAGGREGATION_MODE"
CMD="$CMD --disaggregation-transfer-backend $DISAGGREGATION_TRANSFER_BACKEND"

# KV Cache 量化
if [ "$KV_CACHE_DTYPE" != "none" ]; then
    CMD="$CMD --kv-cache-dtype $KV_CACHE_DTYPE"
fi

# KVTuner 量化
if [ "$ENABLE_KVTUNER" = "true" ]; then
    CMD="$CMD --enable-kvtuner-quant"
    
    # 优先使用层级量化配置
    if [ -n "$KV_TUNER_LAYER_CONFIG" ] && [ -f "$KV_TUNER_LAYER_CONFIG" ]; then
        CMD="$CMD --kvtuner-layer-config $KV_TUNER_LAYER_CONFIG"
        echo "使用层级量化配置：$KV_TUNER_LAYER_CONFIG"
    else
        # 回退到统一量化
        CMD="$CMD --kvtuner-nbits-key $KV_TUNER_NBITS_KEY"
        CMD="$CMD --kvtuner-nbits-value $KV_TUNER_NBITS_VALUE"
        CMD="$CMD --kvtuner-residual-length $KV_TUNER_RESIDUAL_LENGTH"
        echo "使用统一量化配置：${KV_TUNER_NBITS_KEY}-bit"
    fi
    
    # Prefill 模式特定配置
    export SGLANG_KVTUNER_PREFILL_NBITS_SCALE=${KV_TUNER_NBITS_SCALE}
fi

# 其他优化选项
CMD="$CMD --mem-fraction-static 0.8"
CMD="$CMD --max-running-requests 256"
CMD="$CMD --schedule-conservativeness 1.0"
# 禁用 custom all reduce（消费级 GPU 不支持 peer access）
CMD="$CMD --disable-custom-all-reduce"

echo ""
echo "启动命令:"
echo "$CMD"
echo ""
echo "日志文件：$LOG_DIR/prefill_$(date +%Y%m%d_%H%M%S).log"
echo ""

# 启动服务
nohup bash -c "$CMD" > "$LOG_DIR/prefill_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID=$!

echo "Prefill 服务已启动 (PID: $PID)"
echo ""

# 等待服务就绪
echo "等待服务就绪..."
for i in {1..60}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✓ 服务就绪!"
        echo ""
        echo "服务信息:"
        curl -s "http://localhost:$PORT/get_server_info" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    sleep 1
done

echo "✗ 服务启动超时，请检查日志"
exit 1
