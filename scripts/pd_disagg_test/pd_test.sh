#!/bin/bash
# ============================================================================
# P/D Disaggregation 一键测试主控脚本
# 从本地 Mac 执行，通过 SSH 远程控制两台测试机器
#
# 用法:
#   ./pd_test.sh full          # 全自动一条龙
#   ./pd_test.sh start         # 启动 Prefill + Decode
#   ./pd_test.sh stop          # 停止所有服务
#   ./pd_test.sh status        # 探活检查
#   ./pd_test.sh test          # 通过 pd_coordinator.py 推理测试
#   ./pd_test.sh logs          # 拉取日志到本地
#   ./pd_test.sh setup         # 检查两台机器环境
#   ./pd_test.sh sync          # 同步代码到两台机器
#   ./pd_test.sh clean         # 停服务 + 清日志
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 加载默认配置（可被同名环境变量覆盖）
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/configs/default.sh}"
if [ -f "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
fi

# SSH 选项
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# 远程机器上设置的环境变量（传递给 remote_worker.sh）
REMOTE_ENV="SGLANG_REPO=${SGLANG_REPO} VENV_DIR=${VENV_DIR} PREFILL_IP=${PREFILL_IP} DECODE_IP=${DECODE_IP} PREFILL_PORT=${PREFILL_PORT} DECODE_PORT=${DECODE_PORT} DIST_INIT_PORT=${DIST_INIT_PORT} MODEL_PATH=${MODEL_PATH} TRANSFER_BACKEND=${TRANSFER_BACKEND} DISABLE_OVERLAP=${DISABLE_OVERLAP} LOG_LEVEL=${LOG_LEVEL} ENABLE_KVTUNER=${ENABLE_KVTUNER} KVTUNER_LAYER_CONFIG=${KVTUNER_LAYER_CONFIG} DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH} HEALTH_TIMEOUT=${HEALTH_TIMEOUT} LOG_DIR=${REMOTE_LOG_DIR}"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_ok()    { echo -e "${GREEN}${BOLD}[PASS]${NC} $1"; }
log_fail()  { echo -e "${RED}${BOLD}[FAIL]${NC} $1"; }

# ---- 内置函数 ----

_remote_exec() {
    local host=$1
    shift
    ssh ${SSH_OPTS} "${host}" "bash ${REMOTE_WORKER_PATH} $@"
}

_remote_exec_env() {
    local host=$1
    shift
    ssh ${SSH_OPTS} "${host}" "export ${REMOTE_ENV} && bash ${REMOTE_WORKER_PATH} $@"
}

_deploy_worker() {
    local host=$1
    log_info "部署 remote_worker.sh → ${host}"
    scp ${SSH_OPTS} "${SCRIPT_DIR}/remote_worker.sh" "${host}:${REMOTE_WORKER_PATH}"
}

_stop_remote() {
    local host=$1 name=$2
    log_info "停止 ${name} (${host}) ..."
    _remote_exec "${host}" stop 2>/dev/null || true
}

# 探活（通过 SSH 在远程机器上执行 curl，避免本地无法访问内网 IP）
_check_service() {
    local name=$1 host=$2 port=$3
    if ssh ${SSH_OPTS} "${PREFILL_HOST}" "curl -s --max-time 5 http://${host}:${port}/health" > /dev/null 2>&1; then
        log_ok "${name} (${host}:${port}) UP"
        return 0
    else
        log_fail "${name} (${host}:${port}) DOWN"
        return 1
    fi
}

# ---- 主命令 ----

cmd_setup() {
    log_step "检查环境"
    for host_name in "${PREFILL_HOST}:Prefill" "${DECODE_HOST}:Decode"; do
        IFS=':' read -r host name <<< "${host_name}"
        log_step "${name} (${host})"
        _remote_exec_env "${host}" setup-env
        echo ""
    done
    log_ok "环境检查完成"
}

cmd_sync() {
    log_step "同步代码到两台机器"
    for host_name in "${PREFILL_HOST}:Prefill" "${DECODE_HOST}:Decode"; do
        IFS=':' read -r host name <<< "${host_name}"
        log_info "同步 → ${name} (${host})"
        _remote_exec_env "${host}" sync-code
    done
    log_ok "代码同步完成"
}

cmd_start() {
    log_step "===== 启动 P/D Disaggregation 服务 ====="

    # 1. 部署远程脚本
    log_step "部署远程辅助脚本"
    _deploy_worker "${PREFILL_HOST}"
    _deploy_worker "${DECODE_HOST}"

    # 2. 停止旧服务
    log_step "停止旧服务"
    _stop_remote "${PREFILL_HOST}" "Prefill"
    _stop_remote "${DECODE_HOST}" "Decode"

    # 3. 并行启动 Prefill 和 Decode
    log_step "并行启动 Prefill + Decode"
    _remote_exec_env "${PREFILL_HOST}" start-prefill &
    local pid_prefill=$!
    _remote_exec_env "${DECODE_HOST}" start-decode &
    local pid_decode=$!

    wait $pid_prefill || log_warn "Prefill SSH 会话结束"
    wait $pid_decode || log_warn "Decode SSH 会话结束"

    # 4. 等待 Prefill 和 Decode 就绪（并行等待）
    log_step "等待服务就绪..."
    _remote_exec_env "${PREFILL_HOST}" wait-prefill &
    local pid_wait_p=$!
    _remote_exec_env "${DECODE_HOST}" wait-decode &
    local pid_wait_d=$!

    wait $pid_wait_p
    local rc_wp=$?
    wait $pid_wait_d
    local rc_wd=$?

    if [ $rc_wp -ne 0 ]; then
        log_error "Prefill 探活超时"
        return 1
    fi
    if [ $rc_wd -ne 0 ]; then
        log_error "Decode 探活超时"
        return 1
    fi

    log_ok "所有服务已就绪"
    echo ""
    echo "  Prefill: http://${PREFILL_IP}:${PREFILL_PORT}"
    echo "  Decode:  http://${DECODE_IP}:${DECODE_PORT}"
    echo "  Bootstrap: ${PREFILL_IP}:${BOOTSTRAP_PORT}"
}

cmd_stop() {
    log_step "停止所有服务"
    _stop_remote "${PREFILL_HOST}" "Prefill"
    _stop_remote "${DECODE_HOST}" "Decode"
    log_ok "所有服务已停止"
}

cmd_status() {
    log_step "服务状态"
    local all_ok=true

    _check_service "Prefill" "${PREFILL_IP}" "${PREFILL_PORT}" || all_ok=false
    _check_service "Decode" "${DECODE_IP}" "${DECODE_PORT}" || all_ok=false

    echo ""
    if $all_ok; then
        log_ok "所有服务正常"
        return 0
    else
        log_fail "部分服务异常"
        return 1
    fi
}

cmd_test() {
    log_step "推理测试 (pd_coordinator.py)"

    local test_args="${*:-}"

    # 检查 Prefill 和 Decode 是否可达
    if ! ssh ${SSH_OPTS} "${PREFILL_HOST}" "curl -s --max-time 5 http://${PREFILL_IP}:${PREFILL_PORT}/health" > /dev/null 2>&1; then
        log_error "Prefill 不可达，请先启动服务"
        return 1
    fi
    if ! ssh ${SSH_OPTS} "${PREFILL_HOST}" "curl -s --max-time 5 http://${DECODE_IP}:${DECODE_PORT}/health" > /dev/null 2>&1; then
        log_error "Decode 不可达，请先启动服务"
        return 1
    fi

    # 部署 coordinator 到 Prefill 节点并执行
    log_info "部署 pd_coordinator.py → ${PREFILL_HOST}"
    scp ${SSH_OPTS} "${SCRIPT_DIR}/pd_coordinator.py" "${PREFILL_HOST}:${SGLANG_REPO}/pd_coordinator.py"

    log_info "执行推理测试..."
    ssh ${SSH_OPTS} "${PREFILL_HOST}" \
        "${VENV_DIR}/bin/python ${SGLANG_REPO}/pd_coordinator.py \
            --prefill-host ${PREFILL_IP} --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --no-wait \
            --timeout ${TEST_TIMEOUT} \
            ${test_args}" 2>&1
    local rc=${PIPESTATUS[0]}

    if [ $rc -eq 0 ]; then
        log_ok "推理测试通过"
        return 0
    else
        log_fail "推理测试失败"
        return 1
    fi
}

cmd_logs() {
    log_step "拉取日志到本地"

    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local target_dir="${LOCAL_LOG_DIR}/${ts}"
    mkdir -p "${target_dir}"

    log_info "日志目录: ${target_dir}"

    # 拉取 Prefill 节点日志
    for logfile in prefill; do
        scp ${SSH_OPTS} "${PREFILL_HOST}:${REMOTE_LOG_DIR}/${logfile}.log" \
            "${target_dir}/${logfile}.log" 2>/dev/null && \
            log_info "  ${logfile}.log → ${target_dir}/${logfile}.log" || \
            log_warn "  ${logfile}.log 未找到"
    done

    # 拉取 Decode 节点日志
    for logfile in decode; do
        scp ${SSH_OPTS} "${DECODE_HOST}:${REMOTE_LOG_DIR}/${logfile}.log" \
            "${target_dir}/${logfile}.log" 2>/dev/null && \
            log_info "  ${logfile}.log → ${target_dir}/${logfile}.log" || \
            log_warn "  ${logfile}.log 未找到"
    done

    # 输出日志尾部
    echo ""
    for logfile in prefill decode; do
        if [ -f "${target_dir}/${logfile}.log" ]; then
            echo "--- ${logfile}.log (tail -20) ---"
            tail -20 "${target_dir}/${logfile}.log"
            echo ""
        fi
    done

    log_ok "日志已保存到 ${target_dir}"
}

cmd_clean() {
    log_step "清理"
    cmd_stop
    rm -rf "${LOCAL_LOG_DIR}"
    log_ok "清理完成"
}

cmd_full() {
    local start_time
    start_time=$(date +%s)

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║        P/D Disaggregation 全自动测试                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Prefill: ${PREFILL_HOST} (${PREFILL_IP}:${PREFILL_PORT})"
    echo "  Decode:  ${DECODE_HOST} (${DECODE_IP}:${DECODE_PORT})"
    echo "  Bootstrap: ${PREFILL_IP}:${BOOTSTRAP_PORT}"
    echo "  Model:   ${MODEL_PATH}"
    echo "  Backend: ${TRANSFER_BACKEND}"
    echo "  Overlap: ${DISABLE_OVERLAP}"
    echo ""

    # 1. 停旧服务
    log_step "[1/5] 停止旧服务"
    cmd_stop
    echo ""

    # 2. 启动服务
    log_step "[2/5] 启动服务"
    cmd_start
    echo ""

    # 3. 探活
    log_step "[3/5] 探活检查"
    cmd_status
    echo ""

    # 4. 推理测试
    log_step "[4/5] 推理测试"
    cmd_test
    local test_rc=$?
    echo ""

    # 5. 拉取日志
    log_step "[5/5] 拉取日志"
    cmd_logs
    echo ""

    # 汇总
    local end_time elapsed
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo "╔══════════════════════════════════════════════════════════╗"
    if [ $test_rc -eq 0 ]; then
        echo "║  结果: ${GREEN}PASS${NC}                                           ║"
    else
        echo "║  结果: ${RED}FAIL${NC}                                           ║"
    fi
    echo "║  耗时: ${elapsed}s                                            ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    return $test_rc
}

# ---- 用法 ----

usage() {
    echo "P/D Disaggregation 一键测试"
    echo ""
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  full       全自动一条龙 (停旧→启动→探活→测试→拉日志)"
    echo "  setup      检查两台机器环境"
    echo "  sync       同步代码到两台机器"
    echo "  start      启动 Prefill + Decode"
    echo "  stop       停止所有服务"
    echo "  status     探活检查"
    echo "  test       推理测试 (pd_coordinator.py)"
    echo "  test --batch-tests              运行多种输入/输出长度组合测试"
    echo "  test --num-requests N           并发 N 个请求"
    echo "  logs       拉取日志到本地 ./logs/"
    echo "  clean      停服务 + 清理日志"
    echo ""
    echo "配置:"
    echo "  CONFIG_FILE=path/to/config  覆盖默认配置文件"
    echo "  也可通过环境变量覆盖任意配置项，例如:"
    echo "    TRANSFER_BACKEND=tcp MODEL_PATH=/other/model $0 full"
    echo ""
    echo "当前配置:"
    echo "  PREFILL_HOST=${PREFILL_HOST}  DECODE_HOST=${DECODE_HOST}"
    echo "  PREFILL_IP=${PREFILL_IP}:${PREFILL_PORT}  DECODE_IP=${DECODE_IP}:${DECODE_PORT}"
    echo "  BOOTSTRAP_PORT=${BOOTSTRAP_PORT}"
    echo "  MODEL_PATH=${MODEL_PATH}"
    echo "  TRANSFER_BACKEND=${TRANSFER_BACKEND}"
    echo "  VENV_DIR=${VENV_DIR}"
    echo "  DISABLE_OVERLAP=${DISABLE_OVERLAP}"
}

# ---- 入口 ----

case "${1:-}" in
    full)   cmd_full ;;
    setup)  cmd_setup ;;
    sync)   cmd_sync ;;
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    test)   shift; cmd_test "${@:-}" ;;
    logs)   cmd_logs ;;
    clean)  cmd_clean ;;
    -h|--help|help|"")
        usage ;;
    *)
        log_error "未知命令: $1"
        echo ""
        usage
        exit 1
        ;;
esac
