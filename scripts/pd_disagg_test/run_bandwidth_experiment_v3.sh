#!/bin/bash
# ============================================================================
# 带宽限制吞吐量实验 V3
# 使用 tc tbf 精确限速，每个配置重启服务
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/bandwidth_experiment_v3"

mkdir -p "${RESULTS_DIR}"

PREFILL_HOST="${PREFILL_HOST:-ubuntu@10.60.23.70}"
DECODE_HOST="${DECODE_HOST:-ubuntu@10.60.30.66}"
PREFILL_IP="${PREFILL_IP:-10.60.23.70}"
DECODE_IP="${DECODE_IP:-10.60.30.66}"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

CONCURRENCY="${CONCURRENCY:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PROMPT_TYPE="${PROMPT_TYPE:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%H:%M:%S') $1"; }
log_warn()  { echo -e "\033[1;33m[WARN]\033[0m $(date '+%H:%M:%S') $1"; }

# ---- 带宽控制 (tc tbf) ----

set_bandwidth() {
    local bw_mbit=$1
    # burst = rate * 10ms, 最小 32KB
    local burst_kb=$(( bw_mbit * 10 / 8 ))
    [ "$burst_kb" -lt 32 ] && burst_kb=32

    log_info "Setting bandwidth limit: ${bw_mbit}Mbps (burst=${burst_kb}KB) on both machines"
    ssh ${SSH_OPTS} ${PREFILL_HOST} "sudo tc qdisc replace dev eth0 root tbf rate ${bw_mbit}mbit burst ${burst_kb}kb latency 50ms" 2>&1
    ssh ${SSH_OPTS} ${DECODE_HOST}  "sudo tc qdisc replace dev eth0 root tbf rate ${bw_mbit}mbit burst ${burst_kb}kb latency 50ms" 2>&1
}

remove_bandwidth() {
    log_info "Removing bandwidth limits"
    ssh ${SSH_OPTS} ${PREFILL_HOST} "sudo tc qdisc del dev eth0 root 2>/dev/null" 2>&1 || true
    ssh ${SSH_OPTS} ${DECODE_HOST}  "sudo tc qdisc del dev eth0 root 2>/dev/null" 2>&1 || true
}

verify_bandwidth() {
    local expected=$1
    log_info "Verifying tc on prefill:"
    ssh ${SSH_OPTS} ${PREFILL_HOST} "tc qdisc show dev eth0" 2>&1
}

# ---- 服务管理 ----

stop_services() {
    log_info "Stopping services..."
    ssh ${SSH_OPTS} ${PREFILL_HOST} "pkill -9 -f 'python.*sglang' 2>/dev/null" || true
    ssh ${SSH_OPTS} ${DECODE_HOST}  "pkill -9 -f 'python.*sglang' 2>/dev/null" || true
    sleep 3
}

start_services() {
    local extra_env="$1"
    log_info "Starting services (env: ${extra_env:-none})..."

    stop_services

    # 临时移除带宽限制以便快速启动
    local current_bw="$2"
    if [ -n "$current_bw" ] && [ "$current_bw" != "0" ]; then
        remove_bandwidth
    fi

    cd "${SCRIPT_DIR}"
    if [ -n "${extra_env}" ]; then
        eval "export ${extra_env}"
    fi
    PREFILL_HOST="${PREFILL_HOST}" DECODE_HOST="${DECODE_HOST}" \
        CONFIG_FILE="configs/default.sh" ./pd_test.sh start
    sleep 3

    # 恢复带宽限制
    if [ -n "$current_bw" ] && [ "$current_bw" != "0" ]; then
        set_bandwidth "$current_bw"
    fi
}

# ---- 单次 benchmark ----

run_one() {
    local label=$1
    local output="${RESULTS_DIR}/${label}.json"

    log_step "Benchmark: ${label}"

    # 同步脚本
    scp ${SSH_OPTS} "${PROJECT_ROOT}/eval/bench_throughput.py" \
        "${PREFILL_HOST}:/home/ubuntu/sglang-kvtuner-splitwise/eval/" 2>/dev/null

    ssh ${SSH_OPTS} ${PREFILL_HOST} \
        "/home/ubuntu/sglang-env/bin/python /home/ubuntu/sglang-kvtuner-splitwise/eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} --prefill-port 30000 \
            --decode-host ${DECODE_IP}  --decode-port 30001 \
            --concurrency ${CONCURRENCY} \
            --num-requests ${NUM_REQUESTS} \
            --prompt-type ${PROMPT_TYPE} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${label} \
            --output /tmp/${label}.json" 2>&1

    scp ${SSH_OPTS} "${PREFILL_HOST}:/tmp/${label}.json" "${output}" 2>/dev/null
    log_info "Saved: ${output}"
}

# ---- 主流程 ----

log_step "============================================"
log_step "带宽限制吞吐量实验 V3"
log_step "concurrency=${CONCURRENCY}, requests=${NUM_REQUESTS}"
log_step "prompt=${PROMPT_TYPE}, max_new_tokens=${MAX_NEW_TOKENS}"
log_step "============================================"

trap remove_bandwidth EXIT

# 带宽从小到大，最后 unlimited
BANDWIDTHS=("100" "500" "1000" "2000" "5000" "0")

CONFIGS=(
    "baseline:"
    "quant-8bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
    "quant-4bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
)

for bw in "${BANDWIDTHS[@]}"; do
    if [ "$bw" = "0" ]; then
        bw_label="unlimited"
        log_step "===== Bandwidth: UNLIMITED ====="
        remove_bandwidth
    else
        bw_label="${bw}mbps"
        log_step "===== Bandwidth: ${bw}Mbps ====="
    fi

    for entry in "${CONFIGS[@]}"; do
        cfg_name="${entry%%:*}"
        cfg_env="${entry#*:}"
        [ "$cfg_name" = "baseline" ] && cfg_env=""

        label="bw_${bw_label}_${cfg_name}"

        # 每个配置重启服务
        start_services "$cfg_env" "$bw"

        # 设置带宽（如果需要）
        if [ "$bw" != "0" ]; then
            set_bandwidth "$bw"
            verify_bandwidth "$bw"
        fi

        run_one "$label"
    done
done

remove_bandwidth

# ---- 汇总 ----

log_step "============================================"
log_step "Results Summary"
log_step "============================================"

printf "%-30s  %8s  %8s  %8s  %8s\n" "Config" "req/s" "tok/s" "TTFT" "E2E"
echo "-----------------------------------------------------------------------"
for f in "${RESULTS_DIR}"/bw_*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    rps=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['requests_per_sec']:.2f}\")")
    tps=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['output_tokens_per_sec']:.0f}\")")
    ttft=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['mean_ttft_ms']:.0f}\")")
    e2e=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['mean_e2e_ms']:.0f}\")")
    printf "%-30s  %7s  %7s  %6sms  %6sms\n" "$name" "$rps" "$tps" "$ttft" "$e2e"
done
