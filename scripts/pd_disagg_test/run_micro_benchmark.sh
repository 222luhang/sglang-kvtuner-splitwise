#!/bin/bash
# ============================================================================
# 微基准测试 — TTFT 分解 + Triton vs PyTorch 内核耗时对比
#
# 3 部分:
#   1. 内核级微基准: 单独运行 quant/dequant 各路径
#   2. 单请求端到端: 每个配置发 1 个请求，捕获 TIMING 日志
#   3. TTFT 分解: 解析日志，输出各组件耗时
#
# 用法:
#   bash run_micro_benchmark.sh
#   NUM_TOKENS=100 bash run_micro_benchmark.sh  # 长序列
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/micro_benchmark"
mkdir -p "${RESULTS_DIR}"

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"
source "${SCRIPT_DIR}/configs/default.sh"

NUM_TOKENS="${NUM_TOKENS:-30}"
DRY_RUN="${DRY_RUN:-false}"

CONFIG_PATH="/tmp/sglang_ablation.cfg"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

write_config() {
    local content=$1
    if [ "$DRY_RUN" = "true" ]; then return 0; fi
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh ${SSH_OPTS} "${host}" "echo '${content}' > ${CONFIG_PATH}"
    done
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        微基准测试 — TTFT 分解 + 内核对比               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Tokens: ${NUM_TOKENS}"
echo "  Results: ${RESULTS_DIR}"
echo ""

# ==== Part 1: 内核级微基准 ====
log_step "===== Part 1: 内核级微基准 (Triton vs PyTorch) ====="

if [ "$DRY_RUN" = "false" ]; then
    scp ${SSH_OPTS} "${SCRIPT_DIR}/bench_quant_kernels.py" "${PREFILL_HOST}:/tmp/bench_quant_kernels.py"

    log_info "运行内核基准测试..."
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "
        cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        LOG_LEVEL=info python3 /tmp/bench_quant_kernels.py \
            --num-tokens ${NUM_TOKENS} --num-layers 28 --iterations 20
    " 2>&1 | tee "${RESULTS_DIR}/kernel_benchmark.log"

    scp ${SSH_OPTS} "${PREFILL_HOST}:/tmp/kernel_benchmark_results.json" "${RESULTS_DIR}/" 2>/dev/null || true
fi

echo ""

# ==== Part 2: 单请求端到端 (4 configs) ====
log_step "===== Part 2: 单请求 TTFT 分解 ====="

# Configs: config_id|label|runtime_config_content
# Runtime config is written to /tmp/sglang_ablation.cfg on both hosts
A1_CONFIG="PIPELINE_SEND=0
DISABLE_TRITON=0"
A4_CONFIG="PIPELINE_SEND=0
DISABLE_TRITON=1"
A5_CONFIG="PIPELINE_SEND=0
DISABLE_TRITON=0"
A6_CONFIG="PIPELINE_SEND=1
DISABLE_TRITON=0"

# Start 1: baseline
log_step "--- 启动无量化服务 (A1) ---"
cd "${SCRIPT_DIR}"
export CONFIG_FILE="configs/default.sh"
export ENABLE_TRANSFER_QUANT=""
export TRANSFER_QUANT_BITS=""
export LOG_LEVEL="info"

if [ "$DRY_RUN" = "false" ]; then
    ./pd_test.sh stop || true
    sleep 2
    ./pd_test.sh start || { log_error "服务启动失败"; exit 1; }
    sleep 5

    # Run A1
    write_config "PIPELINE_SEND=0
DISABLE_TRITON=0"
    log_info "运行 A1 (baseline) ..."
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "
        cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --concurrency 1 \
            --num-requests 1 \
            --prompt-type medium \
            --max-new-tokens 1 \
            --config-name micro_A1 \
            --output /tmp/micro_A1.json
    " 2>&1 | tail -15

    # Save logs
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "cat /tmp/prefill.log" > "${RESULTS_DIR}/A1_prefill.log" 2>/dev/null || true
    ssh ${SSH_OPTS} "${DECODE_HOST}" "cat /tmp/decode.log" > "${RESULTS_DIR}/A1_decode.log" 2>/dev/null || true
    scp ${SSH_OPTS} "${PREFILL_HOST}:/tmp/micro_A1.json" "${RESULTS_DIR}/A1_bench.json" 2>/dev/null || true

    ./pd_test.sh stop || true
    sleep 2
fi

# Start 2: 4bit quant (A4, A5, A6)
log_step "--- 启动 4bit 量化服务 (A4/A5/A6) ---"
export ENABLE_TRANSFER_QUANT="true"
export TRANSFER_QUANT_BITS="4"

if [ "$DRY_RUN" = "false" ]; then
    ./pd_test.sh start || { log_error "服务启动失败"; exit 1; }
    sleep 5
fi

run_single_config() {
    local cfg_id=$1
    local cfg_label=$2
    local cfg_content=$3

    log_step "--- 运行 ${cfg_id} (${cfg_label}) ---"
    write_config "${cfg_content}"

    if [ "$DRY_RUN" = "false" ]; then
        ssh ${SSH_OPTS} "${PREFILL_HOST}" "
            cd ${SGLANG_REPO} && \
            source ${VENV_DIR}/bin/activate && \
            python3 eval/bench_throughput.py \
                --prefill-host ${PREFILL_IP} \
                --prefill-port ${PREFILL_PORT} \
                --decode-host ${DECODE_IP} \
                --decode-port ${DECODE_PORT} \
                --bootstrap-port ${BOOTSTRAP_PORT} \
                --concurrency 1 \
                --num-requests 1 \
                --prompt-type medium \
                --max-new-tokens 1 \
                --config-name micro_${cfg_id} \
                --output /tmp/micro_${cfg_id}.json
        " 2>&1 | tail -15

        ssh ${SSH_OPTS} "${PREFILL_HOST}" "cat /tmp/prefill.log" > "${RESULTS_DIR}/${cfg_id}_prefill.log" 2>/dev/null || true
        ssh ${SSH_OPTS} "${DECODE_HOST}" "cat /tmp/decode.log" > "${RESULTS_DIR}/${cfg_id}_decode.log" 2>/dev/null || true
        scp ${SSH_OPTS} "${PREFILL_HOST}:/tmp/micro_${cfg_id}.json" "${RESULTS_DIR}/${cfg_id}_bench.json" 2>/dev/null || true
    fi
    echo ""
}

if [ "$DRY_RUN" = "false" ]; then
    ./pd_test.sh start || { log_error "服务启动失败"; exit 1; }
    sleep 5
fi

run_single_config "A4" "4bit PyTorch, no pipe" "${A4_CONFIG}"
run_single_config "A5" "4bit Triton, no pipe" "${A5_CONFIG}"
run_single_config "A6" "4bit Triton, pipe" "${A6_CONFIG}"

if [ "$DRY_RUN" = "false" ]; then
    ./pd_test.sh stop || true
fi

# ==== Part 3: 解析 TTFT 分解 ====
log_step "===== Part 3: TTFT 分解 ====="

if [ "$DRY_RUN" = "false" ]; then
    for cfg_id in A1 A4 A5 A6; do
        prefill_log="${RESULTS_DIR}/${cfg_id}_prefill.log"
        decode_log="${RESULTS_DIR}/${cfg_id}_decode.log"

        if [ ! -f "$prefill_log" ] || [ ! -f "$decode_log" ]; then
            log_warn "跳过 ${cfg_id} (日志缺失)"
            continue
        fi

        echo ""
        log_step "--- ${cfg_id} TTFT 分解 ---"
        python3 "${SCRIPT_DIR}/parse_micro_benchmark.py" \
            --prefill-log "$prefill_log" \
            --decode-log "$decode_log" \
            --output "${RESULTS_DIR}/${cfg_id}_decomposition.json" \
            2>&1

        # Copy to remote for reference
        scp ${SSH_OPTS} "${RESULTS_DIR}/${cfg_id}_decomposition.json" \
            "${PREFILL_HOST}:/tmp/${cfg_id}_decomposition.json" 2>/dev/null || true
    done
fi

# ==== 清理 ====
if [ "$DRY_RUN" = "false" ]; then
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
    ssh ${SSH_OPTS} "${DECODE_HOST}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
fi

echo ""
log_info "所有结果保存在: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/" 2>/dev/null
