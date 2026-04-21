#!/bin/bash
# ============================================================================
# 带宽限制 + 层级混合量化实验
# 在 V3 基础上补充 mixed-A/B/C 策略
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

set_bandwidth() {
    local bw_mbit=$1
    local burst_kb=$(( bw_mbit * 10 / 8 ))
    [ "$burst_kb" -lt 32 ] && burst_kb=32
    log_info "Setting bandwidth: ${bw_mbit}Mbps"
    ssh ${SSH_OPTS} ${PREFILL_HOST} "sudo tc qdisc replace dev eth0 root tbf rate ${bw_mbit}mbit burst ${burst_kb}kb latency 50ms" 2>&1
    ssh ${SSH_OPTS} ${DECODE_HOST}  "sudo tc qdisc replace dev eth0 root tbf rate ${bw_mbit}mbit burst ${burst_kb}kb latency 50ms" 2>&1
}

remove_bandwidth() {
    log_info "Removing bandwidth limits"
    ssh ${SSH_OPTS} ${PREFILL_HOST} "sudo tc qdisc del dev eth0 root 2>/dev/null" 2>&1 || true
    ssh ${SSH_OPTS} ${DECODE_HOST}  "sudo tc qdisc del dev eth0 root 2>/dev/null" 2>&1 || true
}

stop_services() {
    ssh ${SSH_OPTS} ${PREFILL_HOST} "pkill -9 -f 'python.*sglang' 2>/dev/null" || true
    ssh ${SSH_OPTS} ${DECODE_HOST}  "pkill -9 -f 'python.*sglang' 2>/dev/null" || true
    sleep 3
}

start_services() {
    local extra_env="$1"
    local current_bw="$2"
    stop_services

    # 临时移除限速以便快速启动
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

    if [ -n "$current_bw" ] && [ "$current_bw" != "0" ]; then
        set_bandwidth "$current_bw"
    fi
}

run_one() {
    local label=$1
    local output="${RESULTS_DIR}/${label}.json"

    log_step "Benchmark: ${label}"

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

# ---- Mixed 策略配置 ----

MIXED_A='[8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8]'
MIXED_B='[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]'
MIXED_C='[8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8]'

# ---- 主流程 ----

log_step "============================================"
log_step "层级混合量化带宽实验"
log_step "configs: mixed-A (avg 5.4bit), mixed-B (avg 4.7bit), mixed-C (avg 5.4bit)"
log_step "bandwidths: 500Mbps, 1Gbps, 2Gbps, unlimited"
log_step "concurrency=${CONCURRENCY}, requests=${NUM_REQUESTS}"
log_step "============================================"

trap remove_bandwidth EXIT

BANDWIDTHS=("500" "1000" "2000" "0")

CONFIGS=(
    "mixed-A:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8 KVTUNER_LAYER_BITS=${MIXED_A}"
    "mixed-B:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8 KVTUNER_LAYER_BITS=${MIXED_B}"
    "mixed-C:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8 KVTUNER_LAYER_BITS=${MIXED_C}"
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

        label="bw_${bw_label}_${cfg_name}"

        # 跳过已有结果
        if [ -f "${RESULTS_DIR}/${label}.json" ]; then
            log_info "Skipping ${label} (already exists)"
            continue
        fi

        start_services "$cfg_env" "$bw"

        if [ "$bw" != "0" ]; then
            set_bandwidth "$bw"
        fi

        run_one "$label"
    done
done

remove_bandwidth

# ---- 汇总 ----

log_step "============================================"
log_step "Results Summary"
log_step "============================================"

printf "%-35s  %8s  %8s  %8s\n" "Config" "req/s" "tok/s" "TTFT"
echo "---------------------------------------------------------------------"
for f in "${RESULTS_DIR}"/bw_*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    rps=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['requests_per_sec']:.2f}\")")
    tps=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['output_tokens_per_sec']:.0f}\")")
    ttft=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['results']['mean_ttft_ms']:.0f}\")")
    printf "%-35s  %7s  %7s  %6sms\n" "$name" "$rps" "$tps" "$ttft"
done