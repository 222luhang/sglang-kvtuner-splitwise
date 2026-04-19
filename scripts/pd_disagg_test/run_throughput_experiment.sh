#!/bin/bash
# ============================================================================
# 吞吐量实验 - 不同并发度和层级量化配置
# 测试不同并发场景下传输量化的吞吐量效果
#
# 用法: bash run_throughput_experiment.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/throughput_experiment"

mkdir -p "${RESULTS_DIR}"

# 实验参数
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PROMPT_TYPE="${PROMPT_TYPE:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16,32}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# 层级量化配置 (使用函数代替关联数组以兼容旧版 bash)
get_layer_bits() {
    local strategy=$1
    case "$strategy" in
        "uniform-8bit") echo "[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]" ;;
        "uniform-4bit") echo "[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]" ;;
        "mixed-A") echo "[8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8]" ;;
        "mixed-B") echo "[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]" ;;
        "mixed-C") echo "[4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4]" ;;
        "mixed-D") echo "[8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8]" ;;
        *) echo "" ;;
    esac
}

run_benchmark() {
    local config_name=$1
    local exp_env=$2
    local concurrency=$3
    local output_json="${RESULTS_DIR}/${config_name}_c${concurrency}.json"

    log_step "Running benchmark: ${config_name} (concurrency=${concurrency})"

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
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --concurrency ${concurrency} \
            --num-requests ${NUM_REQUESTS} \
            --prompt-type ${PROMPT_TYPE} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${config_name} \
            --output /tmp/${config_name}_c${concurrency}.json" 2>&1

    # 拉取结果
    scp gpu1:"/tmp/${config_name}_c${concurrency}.json" "${output_json}" 2>/dev/null

    log_info "Results saved to ${output_json}"
}

# 主流程
log_step "Throughput Experiment"
log_step "Concurrency levels: ${CONCURRENCY_LEVELS}"
log_step "Requests: ${NUM_REQUESTS}, Prompt type: ${PROMPT_TYPE}"
log_step "Configs: baseline, uniform-8bit, uniform-4bit, mixed-A/B/C/D"

# 配置列表
CONFIGS=(
    "baseline:"
    "uniform-8bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
    "uniform-4bit:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
)

# 添加层级混合精度配置
for strategy in mixed-A mixed-B mixed-C mixed-D; do
    layer_bits=$(get_layer_bits "$strategy")
    CONFIGS+=("${strategy}:ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4 KVTUNER_LAYER_BITS=${layer_bits}")
done

# 解析并发度列表
IFS=',' read -ra CONCURRENCY_ARRAY <<< "${CONCURRENCY_LEVELS}"

# 运行实验
for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    log_step "=========================================="
    log_step "Concurrency: ${concurrency}"
    log_step "=========================================="

    for config_entry in "${CONFIGS[@]}"; do
        config_name="${config_entry%%:*}"
        config_env="${config_entry#*:}"

        if [ "${config_name}" = "baseline" ]; then
            config_env=""
        fi

        run_benchmark "${config_name}" "${config_env}" "${concurrency}"
    done
done

log_step "=========================================="
log_step "All throughput experiments completed!"
log_step "=========================================="

# 汇总结果
log_step "Summary:"
echo ""
printf "%-15s %-8s %10s %12s %12s\n" "Config" "Conc" "Req/s" "Tok/s" "TTFT(ms)"
echo "----------------------------------------------------------------"

for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    for config_name in baseline uniform-8bit uniform-4bit mixed-A mixed-B mixed-C mixed-D; do
        json="${RESULTS_DIR}/${config_name}_c${concurrency}.json"
        if [ -f "$json" ]; then
            req=$(jq -r '.results.requests_per_sec // 0' "$json" 2>/dev/null || echo "0")
            tok=$(jq -r '.results.tokens_per_sec // 0' "$json" 2>/dev/null || echo "0")
            ttft=$(jq -r '.results.mean_ttft_ms // 0' "$json" 2>/dev/null || echo "0")
            printf "%-15s %-8s %10.2f %12.2f %12.1f\n" "$config_name" "c${concurrency}" "$req" "$tok" "$ttft"
        fi
    done
    echo ""
done

# 生成对比分析
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

print("\n" + "=" * 80)
print("Throughput Comparison by Concurrency")
print("=" * 80)

configs = ["baseline", "uniform-8bit", "uniform-4bit", "mixed-A", "mixed-B", "mixed-C", "mixed-D"]
concurrency_levels = sorted(set(
    int(r["params"]["concurrency"]) for r in results if "params" in r and "concurrency" in r["params"]
))

for conc in concurrency_levels:
    print(f"\nConcurrency={conc}:")
    print("-" * 60)

    for config in configs:
        matching = [r for r in results if r.get("config", "").startswith(config) and
                   r.get("params", {}).get("concurrency") == conc]
        if matching:
            r = matching[0]["results"]
            print(f"  {config:15s}: {r['requests_per_sec']:6.2f} req/s, "
                  f"{r['tokens_per_sec']:7.2f} tok/s, TTFT={r['mean_ttft_ms']:.1f}ms")

# 计算量化相对 baseline 的提升
print("\n" + "=" * 80)
print("Throughput Improvement vs Baseline")
print("=" * 80)

for conc in concurrency_levels:
    baseline = [r for r in results if r.get("config", "").startswith("baseline") and
               r.get("params", {}).get("concurrency") == conc]
    if not baseline:
        continue

    baseline_tps = baseline[0]["results"]["tokens_per_sec"]
    print(f"\nConcurrency={conc}:")

    for config in ["uniform-8bit", "uniform-4bit", "mixed-A", "mixed-B", "mixed-C", "mixed-D"]:
        matching = [r for r in results if r.get("config", "").startswith(config) and
                   r.get("params", {}).get("concurrency") == conc]
        if matching:
            tps = matching[0]["results"]["tokens_per_sec"]
            improvement = (tps - baseline_tps) / baseline_tps * 100
            print(f"  {config:15s}: {improvement:+.1f}%")
PYTHON
