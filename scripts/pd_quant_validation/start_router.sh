#!/bin/bash
# start_router.sh - 启动 SGLang Router（用于 P/D 分离负载均衡）

set -e

# 配置
PREFILL_NODES="${PREFILL_NODES:-10.60.6.75:30000,10.60.9.62:30000}"
DECODE_NODES="${DECODE_NODES:-10.60.19.152:30001,10.60.176.217:30001}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# 日志
LOG_DIR="${LOG_DIR:-/tmp/sglang_logs}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动 SGLang Router"
echo "=========================================="
echo "Prefill 节点：$PREFILL_NODES"
echo "Decode 节点：$DECODE_NODES"
echo "Router 地址：$ROUTER_HOST:$ROUTER_PORT"
echo "=========================================="

# 构建 Prefill 和 Decode URL 列表
PREFILL_URLS=""
IFS=',' read -ra PREFILL_ARRAY <<< "$PREFILL_NODES"
for i in "${!PREFILL_ARRAY[@]}"; do
    if [ $i -gt 0 ]; then
        PREFILL_URLS="$PREFILL_URLS,"
    fi
    PREFILL_URLS="$PREFILL_URLS http://${PREFILL_ARRAY[$i]}"
done

DECODE_URLS=""
IFS=',' read -ra DECODE_ARRAY <<< "$DECODE_NODES"
for i in "${!DECODE_ARRAY[@]}"; do
    if [ $i -gt 0 ]; then
        DECODE_URLS="$DECODE_URLS,"
    fi
    DECODE_URLS="$DECODE_URLS http://${DECODE_ARRAY[$i]}"
done

# 启动命令 - 使用 ~/.venv 环境
PYTHON="~/.venv/bin/python3"
CMD="$PYTHON -m sglang_router.launch_router"
CMD="$CMD --pd-disaggregation"
CMD="$CMD --prefill $PREFILL_URLS"
CMD="$CMD --decode $DECODE_URLS"
CMD="$CMD --host $ROUTER_HOST"
CMD="$CMD --port $ROUTER_PORT"

echo ""
echo "启动命令:"
echo "$CMD"
echo ""
echo "日志文件：$LOG_DIR/router_$(date +%Y%m%d_%H%M%S).log"
echo ""

# 启动服务
nohup bash -c "$CMD" > "$LOG_DIR/router_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID=$!

echo "Router 已启动 (PID: $PID)"
echo ""

# 等待服务就绪
echo "等待 Router 就绪..."
for i in {1..30}; do
    if curl -s "http://localhost:$ROUTER_PORT/health" > /dev/null 2>&1; then
        echo "✓ Router 就绪!"
        echo ""
        echo "Router 端点：http://localhost:$ROUTER_PORT"
        echo ""
        exit 0
    fi
    sleep 1
done

echo "✗ Router 启动超时，请检查日志"
exit 1
