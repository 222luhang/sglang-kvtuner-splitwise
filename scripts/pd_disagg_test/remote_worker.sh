#!/bin/bash
# ============================================================================
# 远程辅助脚本 — 部署在 gpu1/gpu2 上，被主控脚本通过 SSH 远程调用
# 用法: bash remote_worker.sh <command> [args...]
# ============================================================================

# 从主控脚本传入的参数（通过 SSH 环境变量或命令行参数）
SGLANG_REPO="${SGLANG_REPO:-/home/ubuntu/sglang-kvtuner-splitwise}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/.venv}"
PREFILL_IP="${PREFILL_IP:-10.60.23.70}"
DECODE_IP="${DECODE_IP:-10.60.30.66}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
LOG_DIR="${LOG_DIR:-/tmp}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 内置函数 ----

_remote_stop() {
    pkill -9 -f "python.*sglang" 2>/dev/null || true
    pkill -9 -f "sglang_router" 2>/dev/null || true
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

_remote_start_prefill() {
    log_info "启动 Prefill (${PREFILL_IP}:${PREFILL_PORT}) ..."
    cd "${SGLANG_REPO}"
    export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH}"
    nohup "${VENV_DIR}/bin/python" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --disaggregation-transfer-backend "${TRANSFER_BACKEND}" \
        --disaggregation-mode prefill \
        --host "${PREFILL_IP}" \
        --port ${PREFILL_PORT} \
        --trust-remote-code \
        --dist-init-addr "${PREFILL_IP}:${DIST_INIT_PORT}" \
        --nnodes 1 --node-rank 0 \
        ${DISABLE_CUSTOM_ALL_REDUCE:+--disable-custom-all-reduce} \
        > "${LOG_DIR}/prefill.log" 2>&1 &
    echo $! > "${LOG_DIR}/prefill.pid"
    log_info "Prefill PID: $!, 日志: ${LOG_DIR}/prefill.log"
}

_remote_start_decode() {
    log_info "启动 Decode (${DECODE_IP}:${DECODE_PORT}) ..."
    cd "${SGLANG_REPO}"
    export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH}"
    nohup "${VENV_DIR}/bin/python" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --disaggregation-transfer-backend "${TRANSFER_BACKEND}" \
        --disaggregation-mode decode \
        --host "${DECODE_IP}" \
        --port ${DECODE_PORT} \
        --trust-remote-code \
        --dist-init-addr "${DECODE_IP}:${DIST_INIT_PORT}" \
        --nnodes 1 --node-rank 0 \
        ${DISABLE_CUSTOM_ALL_REDUCE:+--disable-custom-all-reduce} \
        > "${LOG_DIR}/decode.log" 2>&1 &
    echo $! > "${LOG_DIR}/decode.pid"
    log_info "Decode PID: $!, 日志: ${LOG_DIR}/decode.log"
}

_remote_start_router() {
    log_info "启动 Router (0.0.0.0:${ROUTER_PORT}) ..."
    cd "${SGLANG_REPO}"
    export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH}"
    nohup "${VENV_DIR}/bin/python" -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" \
        --decode "http://${DECODE_IP}:${DECODE_PORT}" \
        --host 0.0.0.0 \
        --port ${ROUTER_PORT} \
        > "${LOG_DIR}/router.log" 2>&1 &
    echo $! > "${LOG_DIR}/router.pid"
    log_info "Router PID: $!, 日志: ${LOG_DIR}/router.log"
}

_remote_test() {
    log_info "发送推理测试..."
    local resp rc
    resp=$(curl -s --max-time ${TEST_TIMEOUT} --http1.1 \
        "http://localhost:${ROUTER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL_PATH}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}], \"max_tokens\": 32}" 2>&1)
    rc=$?

    if [ $rc -ne 0 ]; then
        log_error "curl 失败 (exit code: $rc)"
        return 1
    fi

    # 提取 content 字段
    local content
    content=$(echo "${resp}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(c)
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

    if echo "${content}" | grep -q "PARSE_ERROR"; then
        log_error "JSON 解析失败: ${content}"
        log_error "原始响应: ${resp}"
        return 1
    fi

    if [ -z "${content}" ]; then
        log_error "推理返回空内容"
        log_error "原始响应: ${resp}"
        return 1
    fi

    log_info "推理成功: ${content}"
    echo "${resp}" | python3 -m json.tool 2>/dev/null || echo "${resp}"
    return 0
}

# ---- 子命令分发 ----

case "${1:-}" in
    setup-env)
        cd "${SGLANG_REPO}"
        log_info "Python: $(${VENV_DIR}/bin/python --version 2>&1)"
        log_info "分支: $(git branch --show-current), 提交: $(git log --oneline -1)"
        log_info "Torch: $(${VENV_DIR}/bin/python -c 'import torch; print(f\"torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}\")' 2>&1)"
        log_info "FlashInfer: $(${VENV_DIR}/bin/python -c 'import flashinfer; print(flashinfer.__version__)' 2>&1)"
        log_info "ZMQ: $(${VENV_DIR}/bin/python -c 'import zmq; print(zmq.__version__)' 2>&1)"
        log_info "模型: $(ls ${MODEL_PATH}/config.json 2>/dev/null && echo OK || echo NOT_FOUND)"
        ;;
    start-prefill)
        _remote_stop
        _remote_start_prefill
        ;;
    start-decode)
        _remote_stop
        _remote_start_decode
        ;;
    start-router)
        _remote_start_router
        ;;
    start-prefill-only)
        # 不停止已有服务，仅启动 prefill
        _remote_start_prefill
        ;;
    start-decode-only)
        # 不停止已有服务，仅启动 decode
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
    wait-router)
        _remote_wait_health "Router" "http://localhost:${ROUTER_PORT}/health" 30
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
        echo "=== Router ==="
        curl -s --max-time 3 "http://localhost:${ROUTER_PORT}/health" > /dev/null 2>&1 \
            && echo "  Router (localhost:${ROUTER_PORT}): UP" \
            || echo "  Router (localhost:${ROUTER_PORT}): DOWN"
        ;;
    test)
        _remote_test
        ;;
    logs)
        for f in prefill decode router; do
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
        echo "  start-prefill         停旧服务 + 启动 Prefill"
        echo "  start-decode          停旧服务 + 启动 Decode"
        echo "  start-router          启动 Router"
        echo "  start-prefill-only    启动 Prefill (不停止已有服务)"
        echo "  start-decode-only     启动 Decode (不停止已有服务)"
        echo "  stop                  停止所有服务"
        echo "  wait-prefill          等待 Prefill 就绪"
        echo "  wait-decode           等待 Decode 就绪"
        echo "  wait-router           等待 Router 就绪"
        echo "  status                检查服务状态"
        echo "  test                  推理测试"
        echo "  logs                  输出日志"
        exit 1
        ;;
esac
