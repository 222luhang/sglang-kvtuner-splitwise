#!/bin/bash
# ============================================================================
# 远程辅助脚本 — 部署在 gpu1/gpu2 上，被主控脚本通过 SSH 远程调用
# 用法: bash remote_worker.sh <command> [args...]
# ============================================================================

SGLANG_REPO="${SGLANG_REPO:-/home/ubuntu/sglang-kvtuner-splitwise}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/sglang-env}"
PREFILL_IP="${PREFILL_IP:-117.50.192.238}"
DECODE_IP="${DECODE_IP:-117.50.189.89}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"
DISABLE_OVERLAP="${DISABLE_OVERLAP:-true}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
LOG_DIR="${LOG_DIR:-/tmp}"
ENABLE_TRANSFER_QUANT="${ENABLE_TRANSFER_QUANT:-}"
TRANSFER_QUANT_BITS="${TRANSFER_QUANT_BITS:-8}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 内置函数 ----

_remote_stop() {
    pkill -9 -f "python.*sglang" 2>/dev/null || true
    sleep 2
}

_remote_wait_health() {
    local name=$1 url=$2 max_wait=${3:-$HEALTH_TIMEOUT} elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s --max-time 3 "${url}" > /dev/null 2>&1; then
            log_info "${name} 就绪 (${elapsed}s)"
            return 0
        fi
        sleep 3; elapsed=$((elapsed + 3))
    done
    log_error "${name} 在 ${max_wait}s 内未就绪"
    return 1
}

# 构建公共启动参数
_launch_args() {
    local mode=$1 host=$2 port=$3
    echo "--model-path ${MODEL_PATH} \
--disaggregation-transfer-backend ${TRANSFER_BACKEND} \
--disaggregation-mode ${mode} \
--host ${host} \
--port ${port} \
--trust-remote-code \
--dist-init-addr ${host}:${DIST_INIT_PORT} \
--nnodes 1 --node-rank 0 \
--disable-custom-all-reduce \
--log-level ${LOG_LEVEL} \
${DISABLE_OVERLAP:+--disable-overlap-schedule} \
${ENABLE_KVTUNER:+--enable-kvtuner-quant --kvtuner-layer-config ${KVTUNER_LAYER_CONFIG} --disable-cuda-graph} \
${ENABLE_TRANSFER_QUANT:+--enable-transfer-quant --transfer-quant-bits ${TRANSFER_QUANT_BITS}}"
}

_remote_start_prefill() {
    log_info "启动 Prefill (${PREFILL_IP}:${PREFILL_PORT}) ..."
    cd "${SGLANG_REPO}"
    local args
    args=$(_launch_args prefill "${PREFILL_IP}" "${PREFILL_PORT}")
    nohup "${VENV_DIR}/bin/python" -m sglang.launch_server ${args} \
        > "${LOG_DIR}/prefill.log" 2>&1 &
    echo $! > "${LOG_DIR}/prefill.pid"
    log_info "Prefill PID: $!, 日志: ${LOG_DIR}/prefill.log"
}

_remote_start_decode() {
    log_info "启动 Decode (${DECODE_IP}:${DECODE_PORT}) ..."
    cd "${SGLANG_REPO}"
    local args
    args=$(_launch_args decode "${DECODE_IP}" "${DECODE_PORT}")
    nohup "${VENV_DIR}/bin/python" -m sglang.launch_server ${args} \
        > "${LOG_DIR}/decode.log" 2>&1 &
    echo $! > "${LOG_DIR}/decode.pid"
    log_info "Decode PID: $!, 日志: ${LOG_DIR}/decode.log"
}

# ---- 子命令分发 ----

case "${1:-}" in
    setup-env)
        cd "${SGLANG_REPO}"
        log_info "Python: $(${VENV_DIR}/bin/python --version 2>&1)"
        log_info "分支: $(git branch --show-current), 提交: $(git log --oneline -1)"
        log_info "Torch: $(${VENV_DIR}/bin/python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}")' 2>&1)"
        log_info "模型: $(ls ${MODEL_PATH}/config.json 2>/dev/null && echo OK || echo NOT_FOUND)"
        ;;
    sync-code)
        cd "${SGLANG_REPO}"
        log_info "同步代码..."
        git pull 2>&1
        ;;
    start-prefill)
        _remote_start_prefill
        ;;
    start-decode)
        _remote_start_decode
        ;;
    start-prefill-clean)
        _remote_stop
        _remote_start_prefill
        ;;
    start-decode-clean)
        _remote_stop
        _remote_start_decode
        ;;
    stop)
        _remote_stop
        log_info "服务已停止"
        ;;
    wait-prefill)
        _remote_wait_health "Prefill" "http://${PREFILL_IP}:${PREFILL_PORT}/health" "${HEALTH_TIMEOUT}"
        ;;
    wait-decode)
        _remote_wait_health "Decode" "http://${DECODE_IP}:${DECODE_PORT}/health" "${HEALTH_TIMEOUT}"
        ;;
    status)
        echo "=== Prefill ==="
        curl -s --max-time 3 "http://${PREFILL_IP}:${PREFILL_PORT}/health" > /dev/null 2>&1 \
            && echo "  Prefill (${PREFILL_IP}:${PREFILL_PORT}): UP" \
            || echo "  Prefill (${PREFILL_IP}:${PREFILL_PORT}): DOWN"
        echo "=== Decode ==="
        curl -s --max-time 3 "http://${DECODE_IP}:${DECODE_PORT}/health" > /dev/null 2>&1 \
            && echo "  Decode (${DECODE_IP}:${DECODE_PORT}): UP" \
            || echo "  Decode (${DECODE_IP}:${DECODE_PORT}): DOWN"
        ;;
    logs)
        for f in prefill decode; do
            echo "=== ${f}.log (tail -30) ==="
            tail -30 "${LOG_DIR}/${f}.log" 2>/dev/null || echo "(无日志)"
            echo ""
        done
        ;;
    *)
        echo "远程辅助脚本 (remote_worker.sh)"
        echo "用法: bash $0 <command>"
        echo ""
        echo "命令:"
        echo "  setup-env             检查环境"
        echo "  sync-code             同步代码 (git pull)"
        echo "  start-prefill         启动 Prefill"
        echo "  start-decode          启动 Decode"
        echo "  start-prefill-clean   停旧服务 + 启动 Prefill"
        echo "  start-decode-clean    停旧服务 + 启动 Decode"
        echo "  stop                  停止所有服务"
        echo "  wait-prefill          等待 Prefill 就绪"
        echo "  wait-decode           等待 Decode 就绪"
        echo "  status                检查服务状态"
        echo "  logs                  输出日志"
        exit 1
        ;;
esac
