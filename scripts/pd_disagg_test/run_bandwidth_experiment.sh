#!/bin/bash
# ============================================================================
# 带宽限制吞吐量实验
# 使用 tc tbf 限制带宽,测试不同带宽下传输量化的吞吐量效果
#
# 用法: bash run_bandwidth_experiment.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/bandwidth_experiment"

mkdir -p "${RESULTS_DIR}"

# 实验参数
CONCURRENCY="${CONCURRENCY:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PROMPT_TYPE="${PROMPT_TYPE:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 设置带宽限制
# 使用 tc tbf (Token Bucket Filter) 限制出口带宽
set_bandwidth() {
    local bandwidth_mbps=$1
    local burst_kb=$((bandwidth_mbps * 10))  # burst = 10x bandwidth in KB
    local latency_ms=50  # max latency through queue

    log_info "Setting bandwidth limit to ${bandwidth_mbps}Mbps..."

    # 在两台机器上都设置出口带宽限制
    # TBF 参数: rate=带宽, burst=突发大小, latency=最大队列延迟
    ssh gpu1 "sudo tc qdisc replace dev eth0 root handle 1: tbf rate ${bandwidth_mbps}mbit burst ${burst_kb}kbit latency ${latency_ms}ms" 2>/dev/null
    ssh gpu2 "sudo tc qdisc replace dev eth0 root handle 1: tbf rate ${bandwidth_mbps}mbit burst ${burst_kb}kbit latency ${latency_ms}ms" 2>/dev/null
}

# 移除带宽限制
remove_bandwidth() {
    log_info "Removing bandwidth limits..."
    ssh gpu1 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
    ssh gpu2 "sudo tc qdisc del dev eth0 root 2>/dev/null" || true
}

# 运行单个配置的 benchmark
run_benchmark() {
    local config_name=$1
    local exp_env=$2
    local output_json="${RESULTS_DIR}/${config_name}.json"

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

    # 运行 benchmark
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
            --prompt-type ${PROMPT_TYPE} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${config_name} \
            --output /tmp/${config_name}.json" 2>&1

    # 拉取结果
    scp gpu1:"/tmp/${config_name}.json" "${output_json}" 2>/dev/null

    log_info "Results saved to ${output_json}"
}

# 主流程
log_step "Bandwidth Limit Experiment"
log_step "Testing bandwidth limits: 100Mbps, 500Mbps, 1Gbps, unlimited"
log_step "Concurrency: ${CONCURRENCY}, Requests: ${NUM_REQUESTS}"
log_step "Prompt type: ${PROMPT_TYPE}, Max new tokens: ${MAX_NEW_TOKENS}"

# 确保清理
trap remove_bandwidth EXIT

# 带宽配置 (Mbps, 0 = unlimited)
BANDWIDTH_VALUES=("100" "500" "1000" "0")
CONFIGS=(
    "baseline:"
    "quant-8bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
    "quant-4bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
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
    fi

    # 验证带宽设置
    log_info "Verifying bandwidth..."
    ssh gpu1 "tc qdisc show dev eth0 | head -2" || true

    # 运行各配置
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

log_step "=========================================="
log_step "All experiments completed!"
log_step "=========================================="

# 汇总结果
log_step "Summary:"
echo ""
printf "%-20s %-12s %12s %12s %12s\n" "Config" "BW(Mbps)" "Req/s" "Tok/s" "TTFT(ms)"
echo "------------------------------------------------------------------------"

for json in "${RESULTS_DIR}"/*.json; do
    if [ -f "$json" ]; then
        name=$(basename "$json" .json)
        req_per_sec=$(jq -r '.results.requests_per_sec // 0' "$json")
        tok_per_sec=$(jq -r '.results.tokens_per_sec // 0' "$json")
        mean_ttft=$(jq -r '.results.mean_ttft_ms // 0' "$json")
        printf "%-20s %-12s %12.2f %12.2f %12.1f\n" "$name" "-" "$req_per_sec" "$tok_per_sec" "$mean_ttft"
    fi
done

# 生成对比报告
log_step "Generating comparison report..."
python3 << 'PYTHON'
import json
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
results = []

for json_file in sorted(results_dir.glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)
    results.append(data)

# 按带宽分组
print("\n" + "=" * 80)
print("Throughput Comparison by Bandwidth")
print("=" * 80)

for bw in ["100", "500", "1000", "0"]:
    bw_label = "Unlimited" if bw == "0" else f"{bw}Mbps"
    print(f"\n{bw_label}:")
    print("-" * 60)

    for config in ["baseline", "quant-8bit", "quant-4bit"]:
        name = f"bw{bw}mbps_{config}"
        matching = [r for r in results if r["config"] == name]
        if matching:
            r = matching[0]["results"]
            print(f"  {config:12s}: {r['requests_per_sec']:6.2f} req/s, "
                  f"{r['tokens_per_sec']:7.2f} tok/s, TTFT={r['mean_ttft_ms']:.1f}ms")

# 计算带宽限制下量化的收益
print("\n" + "=" * 80)
print("Throughput Improvement from Transfer Quantization")
print("=" * 80)

for bw in ["100", "500", "1000"]:
    baseline_name = f"bw{bw}mbps_baseline"
    baseline = [r for r in results if r["config"] == baseline_name]
    if not baseline:
        continue

    baseline_tps = baseline[0]["results"]["tokens_per_sec"]
    print(f"\n{bw}Mbps:")

    for config in ["quant-8bit", "quant-4bit"]:
        name = f"bw{bw}mbps_{config}"
        matching = [r for r in results if r["config"] == name]
        if matching:
            tps = matching[0]["results"]["tokens_per_sec"]
            improvement = (tps - baseline_tps) / baseline_tps * 100
            print(f"  {config}: {improvement:+.1f}% throughput change")
PYTHON
