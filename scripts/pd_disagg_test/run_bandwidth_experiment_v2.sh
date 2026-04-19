#!/bin/bash
# ============================================================================
# 带宽限制吞吐量实验 V2
# 实际带宽约 15Gbps，测试 100Mbps/500Mbps/1Gbps/5Gbps/10Gbps/无限
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/bandwidth_experiment_v2"

mkdir -p "${RESULTS_DIR}"

CONCURRENCY="${CONCURRENCY:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 设置带宽限制 (使用 tc netem rate)
set_bandwidth() {
    local bandwidth_mbps=$1
    
    log_info "Setting bandwidth limit to ${bandwidth_mbps}Mbps..."
    
    # 使用 netem rate 限制，更可靠
    ssh gpu1 "sudo tc qdisc replace dev eth0 root netem rate ${bandwidth_mbps}mbit" 2>/dev/null
    ssh gpu2 "sudo tc qdisc replace dev eth0 root netem rate ${bandwidth_mbps}mbit" 2>/dev/null
}

# 移除带宽限制
remove_bandwidth() {
    log_info "Removing bandwidth limits..."
    ssh gpu1 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
    ssh gpu2 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
}

# 验证带宽设置
verify_bandwidth() {
    local expected_mbps=$1
    log_info "Verifying bandwidth limit (${expected_mbps}Mbps)..."
    
    # 简单验证：测试实际传输速度
    ssh gpu1 "timeout 5 python3 -c \"
import socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 9005))
s.listen(1)
s.settimeout(3)
try:
    conn, addr = s.accept()
    total = 0
    start = time.time()
    while time.time() - start < 3:
        d = conn.recv(65536)
        if not d: break
        total += len(d)
    elapsed = time.time() - start
    mbps = total * 8 / elapsed / 1e6
    print(f'Actual bandwidth: {mbps:.1f} Mbps (expected: ${expected_mbps} Mbps)')
except: pass
\"" &
    sleep 1
    
    ssh gpu2 "python3 -c \"
import socket
s = socket.socket()
s.connect(('10.60.23.70', 9005))
data = b'x' * 65536
for _ in range(500):
    s.sendall(data)
s.close()
\"" 2>&1
    
    sleep 2
}

run_benchmark() {
    local config_name=$1
    local exp_env=$2
    local output_json="${RESULTS_DIR}/${config_name}.json"

    log_step "Running benchmark: ${config_name}"

    (cd "${SCRIPT_DIR}" && ./pd_test.sh stop) || true
    sleep 3

    cd "${SCRIPT_DIR}"
    if [ -n "${exp_env}" ]; then
        eval "export ${exp_env}"
    fi
    CONFIG_FILE="configs/default.sh" ./pd_test.sh start
    sleep 5

    source "${SCRIPT_DIR}/configs/default.sh"

    scp "${PROJECT_ROOT}/eval/bench_throughput.py" gpu1:"${SGLANG_REPO}/eval/" 2>/dev/null

    ssh gpu1 "cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --concurrency ${CONCURRENCY} \
            --num-requests ${NUM_REQUESTS} \
            --config-name ${config_name} \
            --output /tmp/${config_name}.json" 2>&1

    scp gpu1:"/tmp/${config_name}.json" "${output_json}" 2>/dev/null

    log_info "Results saved to ${output_json}"
}

# 主流程
log_step "Bandwidth Experiment V2"
log_step "Actual bandwidth: ~15 Gbps"
log_step "Testing: 100Mbps, 500Mbps, 1Gbps, 5Gbps, 10Gbps, unlimited"
log_step "Concurrency: ${CONCURRENCY}, Requests: ${NUM_REQUESTS}"

trap remove_bandwidth EXIT

# 带宽配置 (Mbps)
BANDWIDTH_VALUES=("100" "500" "1000" "5000" "10000" "0")
CONFIGS=(
    "baseline:"
    "quant-8bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
)

for bw in "${BANDWIDTH_VALUES[@]}"; do
    if [ "${bw}" = "0" ]; then
        log_step "=========================================="
        log_step "Bandwidth: Unlimited"
        log_step "=========================================="
        remove_bandwidth
    else
        log_step "=========================================="
        log_step "Bandwidth: ${bw}Mbps"
        log_step "=========================================="
        set_bandwidth ${bw}
        verify_bandwidth ${bw}
    fi

    for config_entry in "${CONFIGS[@]}"; do
        config_name="${config_entry%%:*}"
        config_env="${config_entry#*:}"

        if [ "${config_name}" = "baseline" ]; then
            config_env=""
        fi

        run_benchmark "bw${bw}mbps_${config_name}" "${config_env}"
    done
done

remove_bandwidth

log_step "All experiments completed!"
log_step "Summary:"
for json in "${RESULTS_DIR}"/*.json; do
    if [ -f "$json" ]; then
        name=$(basename "$json" .json)
        req=$(jq -r '.results.requests_per_sec // 0' "$json")
        tok=$(jq -r '.results.tokens_per_sec // 0' "$json")
        echo "  ${name}: ${req} req/s, ${tok} tok/s"
    fi
done
