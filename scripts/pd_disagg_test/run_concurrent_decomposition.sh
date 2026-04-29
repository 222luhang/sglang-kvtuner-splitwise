#!/bin/bash
# ============================================================================
# 并发 TTFT 分解实验 — 对比 conc=1 vs conc=8 下的各组件耗时
#
# 复用微基准框架，运行 A5 (4bit Triton, no pipe) 和 A6 (4bit Triton, pipe)
# 在 conc=1 和 conc=8 下的 TTFT 分解，定位并发延迟根因。
#
# 用法:
#   bash run_concurrent_decomposition.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/concurrent_decomposition"
mkdir -p "${RESULTS_DIR}"

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"
source "${SCRIPT_DIR}/configs/default.sh"

CONCURRENCY="${CONCURRENCY:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
CONFIG_PATH="/tmp/sglang_ablation.cfg"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

write_config() {
    local content=$1
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh ${SSH_OPTS} "${host}" "echo '${content}' > ${CONFIG_PATH}"
    done
}

run_config() {
    local cfg_id=$1
    local cfg_label=$2
    local cfg_content=$3
    local conc=$4
    local nreq=$5

    local out_dir="${RESULTS_DIR}/${cfg_id}_c${conc}"
    mkdir -p "${out_dir}"

    log_step "--- ${cfg_id} (${cfg_label}) conc=${conc} reqs=${nreq} ---"
    write_config "${cfg_content}"

    # 清空远程日志
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "> /tmp/prefill.log" 2>/dev/null || true
    ssh ${SSH_OPTS} "${DECODE_HOST}" "> /tmp/decode.log" 2>/dev/null || true

    ssh ${SSH_OPTS} "${PREFILL_HOST}" "
        cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --concurrency ${conc} \
            --num-requests ${nreq} \
            --prompt-type medium \
            --max-new-tokens 1 \
            --config-name ${cfg_id}_c${conc} \
            --output /tmp/${cfg_id}_c${conc}.json
    " 2>&1 | tail -15

    # 拉取日志和结果
    ssh ${SSH_OPTS} "${PREFILL_HOST}" "cat /tmp/prefill.log" > "${out_dir}/prefill.log" 2>/dev/null || true
    ssh ${SSH_OPTS} "${DECODE_HOST}" "cat /tmp/decode.log" > "${out_dir}/decode.log" 2>/dev/null || true
    scp ${SSH_OPTS} "${PREFILL_HOST}:/tmp/${cfg_id}_c${conc}.json" "${out_dir}/bench.json" 2>/dev/null || true

    echo ""
}

# 同步解析脚本
scp ${SSH_OPTS} "${SCRIPT_DIR}/parse_micro_benchmark.py" "${PREFILL_HOST}:/tmp/parse_micro_benchmark.py" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     并发 TTFT 分解实验 (conc=${CONCURRENCY})                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "${SCRIPT_DIR}"
export CONFIG_FILE="configs/default.sh"
export ENABLE_TRANSFER_QUANT="true"
export TRANSFER_QUANT_BITS="4"
export LOG_LEVEL="info"

# ---- Start: 4bit quant service ----
log_step "启动 4bit 量化服务..."
./pd_test.sh stop || true
sleep 2
./pd_test.sh start || { log_error "服务启动失败"; exit 1; }
sleep 5

# A5_CONFIG: 4bit Triton, no pipe
A5_CONFIG="PIPELINE_SEND=0
DISABLE_TRITON=0"
A6_CONFIG="PIPELINE_SEND=1
DISABLE_TRITON=0"

# ---- Run all configs ----
# conc=1 (baseline reference)
run_config "A5" "4bit Triton, no pipe" "${A5_CONFIG}" 1 1
run_config "A6" "4bit Triton, pipe"   "${A6_CONFIG}" 1 1

# conc=8 (concurrent)
run_config "A5" "4bit Triton, no pipe" "${A5_CONFIG}" ${CONCURRENCY} ${NUM_REQUESTS}
run_config "A6" "4bit Triton, pipe"   "${A6_CONFIG}" ${CONCURRENCY} ${NUM_REQUESTS}

# ---- Stop ----
./pd_test.sh stop || true

# ---- Cleanup ----
ssh ${SSH_OPTS} "${PREFILL_HOST}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
ssh ${SSH_OPTS} "${DECODE_HOST}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true

# ==== Parse all logs ====
log_step "===== 解析 TTFT 分解 ====="
echo ""

for cfg_dir in "${RESULTS_DIR}"/A*_c*; do
    [ -d "$cfg_dir" ] || continue
    cfg_name=$(basename "$cfg_dir")
    prefill_log="${cfg_dir}/prefill.log"
    decode_log="${cfg_dir}/decode.log"

    if [ ! -s "$prefill_log" ] || [ ! -s "$decode_log" ]; then
        log_warn "跳过 ${cfg_name} (日志为空)"
        continue
    fi

    log_step "--- ${cfg_name} ---"
    python3 "${SCRIPT_DIR}/parse_micro_benchmark.py" \
        --prefill-log "$prefill_log" \
        --decode-log "$decode_log" \
        --output "${cfg_dir}/decomposition.json" \
        2>&1
    echo ""
done

# ==== Cross-config summary ====
log_step "===== 汇总对比 ====="
echo ""

printf "%-20s %8s %8s %8s %8s %8s %8s %8s\n" \
    "配置" "TTFT" "recv_tot" "recv_ms" "deq_ms" "write_ms" "send_tot" "rooms"
echo "----------------------------------------------------------------------------------------------"

for cfg_dir in "${RESULTS_DIR}"/A*_c*; do
    [ -d "$cfg_dir" ] || continue
    cfg_name=$(basename "$cfg_dir")
    json_file="${cfg_dir}/decomposition.json"
    bench_file="${cfg_dir}/bench.json"

    if [ ! -f "$json_file" ]; then continue; fi

    # Extract TTFT from bench result
    ttft=$(jq -r '.results.mean_ttft_ms // "N/A"' "$bench_file" 2>/dev/null || echo "N/A")

    # Count rooms (transfers) in receiver summaries
    rooms=$(jq '[.receiver_summaries[] | .room] | unique | length' "$json_file" 2>/dev/null || echo "?")

    # Average receiver metrics across rooms
    avg_recv=$(jq '[.receiver_summaries[] | .total_ms] | add / length' "$json_file" 2>/dev/null || echo "N/A")
    avg_recv_ms=$(jq '[.receiver_summaries[] | .total_recv_ms] | add / length' "$json_file" 2>/dev/null || echo "N/A")
    avg_dequant=$(jq '[.receiver_summaries[] | .total_dequant_ms] | add / length' "$json_file" 2>/dev/null || echo "N/A")
    avg_write=$(jq '[.receiver_summaries[] | .total_write_ms] | add / length' "$json_file" 2>/dev/null || echo "N/A")

    # Average sender metrics
    avg_send=$(jq '[.sender_summaries[] | .total_ms] | add / length' "$json_file" 2>/dev/null || echo "N/A")

    # Format
    if [ "$ttft" != "N/A" ]; then
        ttft_str=$(printf "%.0f" "$ttft" 2>/dev/null || echo "$ttft")
    else
        ttft_str="N/A"
    fi

    for val in avg_recv avg_recv_ms avg_dequant avg_write avg_send; do
        eval "${val}=\$(printf '%.1f' \$$val 2>/dev/null || echo N/A)"
    done

    printf "%-20s %8s %8s %8s %8s %8s %8s %8s\n" \
        "${cfg_name}" "${ttft_str}" "${avg_recv}" "${avg_recv_ms}" "${avg_dequant}" "${avg_write}" "${avg_send}" "${rooms}"
done

echo ""
log_info "所有结果保存在: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/" 2>/dev/null
