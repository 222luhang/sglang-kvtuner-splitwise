#!/bin/bash
# ============================================================================
# 网络延迟测试 - 测试不同RTT下传输量化的效果
#
# 用法: bash run_latency_experiment.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/latency_experiment"

mkdir -p "${RESULTS_DIR}"

NUM_RUNS="${NUM_RUNS:-3}"  # 减少运行次数以加快测试

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 添加网络延迟 (RTT = 2 * delay)
add_latency() {
    local delay_ms=$1  # 单程延迟 (ms), RTT = 2 * delay_ms
    log_info "Adding ${delay_ms}ms one-way latency (${delay_ms}ms RTT)..."

    # 在两台机器上都添加延迟
    # gpu1: 对发往 gpu2 的流量添加延迟
    ssh gpu1 "sudo tc qdisc add dev eth0 root netem delay ${delay_ms}ms 5ms distribution normal" 2>/dev/null || \
    ssh gpu1 "sudo tc qdisc replace dev eth0 root netem delay ${delay_ms}ms 5ms distribution normal"

    # gpu2: 对发往 gpu1 的流量添加延迟
    ssh gpu2 "sudo tc qdisc add dev eth0 root netem delay ${delay_ms}ms 5ms distribution normal" 2>/dev/null || \
    ssh gpu2 "sudo tc qdisc replace dev eth0 root netem delay ${delay_ms}ms 5ms distribution normal"
}

# 移除网络延迟
remove_latency() {
    log_info "Removing network latency..."
    ssh gpu1 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
    ssh gpu2 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
}

# 运行单个配置的benchmark
run_benchmark() {
    local config_name=$1
    local exp_env=$2
    local output_csv="${RESULTS_DIR}/${config_name}.csv"

    log_step "Running benchmark: ${config_name}"

    # 停止旧服务
    (cd "${SCRIPT_DIR}" && ./pd_test.sh stop) || true
    sleep 3

    # 启动服务
    cd "${SCRIPT_DIR}"
    if [ -n "${exp_env}" ]; then
        eval "export ${exp_env}"
    fi
    CONFIG_FILE="configs/default.sh" ./pd_test.sh start
    sleep 5

    # 运行benchmark
    source "${SCRIPT_DIR}/configs/default.sh"

    scp "${PROJECT_ROOT}/eval/bench_ttft.py" gpu1:"${SGLANG_REPO}/eval/" 2>/dev/null

    ssh gpu1 "cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_ttft.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --config-name ${config_name} \
            --num-runs ${NUM_RUNS} \
            --input-types short,medium,long \
            --output /tmp/${config_name}.csv" 2>&1

    # 拉取结果
    scp gpu1:"/tmp/${config_name}.csv" "${output_csv}" 2>/dev/null

    log_info "Results saved to ${output_csv}"
}

# 主流程
log_step "Network Latency Experiment"
log_step "Testing RTT values: 10ms, 50ms, 100ms, 200ms"
log_step "Configs: baseline, quant-8bit, quant-4bit"
log_step "Runs per input type: ${NUM_RUNS}"

# 确保清理延迟设置
trap remove_latency EXIT

# RTT值 (单程延迟 = RTT/2)
RTT_VALUES=("10" "50" "100" "200")
CONFIGS=(
    "baseline:"
    "quant-8bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
    "quant-4bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
)

# 只测试较大的输入类型 (short, medium, long)
# 因为 tiny 的传输时间太短,延迟影响不明显
# 通过 bench_ttft.py 的 --input-types 参数指定

for rtt in "${RTT_VALUES[@]}"; do
    delay=$((rtt / 2))

    log_step "=========================================="
    log_step "RTT: ${rtt}ms (one-way: ${delay}ms)"
    log_step "=========================================="

    # 添加延迟
    add_latency ${delay}

    # 验证延迟 (从gpu1 ping gpu2)
    source "${SCRIPT_DIR}/configs/default.sh"
    log_info "Verifying latency (gpu1 -> gpu2)..."
    ssh gpu1 "ping -c 3 ${DECODE_IP} | tail -1" 2>/dev/null || true

    # 运行各配置
    for config_entry in "${CONFIGS[@]}"; do
        config_name="${config_entry%%:*}"
        config_env="${config_entry#*:}"

        if [ "${config_name}" = "baseline" ]; then
            config_env=""
        fi

        run_benchmark "rtt${rtt}_${config_name}" "${config_env}"
    done
done

# 移除延迟
remove_latency

log_step "=========================================="
log_step "All experiments completed!"
log_step "=========================================="

# 汇总结果
log_step "Summary:"
for csv in "${RESULTS_DIR}"/*.csv; do
    if [ -f "$csv" ]; then
        name=$(basename "$csv" .csv)
        mean_ttft=$(tail -n +2 "$csv" | awk -F',' '{sum+=$6} END {print sum/NR}')
        log_info "  ${name}: mean TTFT=${mean_ttft}ms"
    fi
done
