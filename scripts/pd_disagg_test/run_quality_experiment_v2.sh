#!/bin/bash
# ============================================================================
# 质量评估实验 V2
# 测试传输量化对模型输出质量的影响，包括层级混合精度配置
#
# 用法: bash run_quality_experiment_v2.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/quality_experiment_v2"

mkdir -p "${RESULTS_DIR}"

# 实验参数
NUM_SAMPLES="${NUM_SAMPLES:-50}"
BENCHMARKS="${BENCHMARKS:-gsm8k,mmlu,hellaswag}"

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
    local benchmark=$3
    local output_json="${RESULTS_DIR}/${benchmark}_${config_name}.json"

    log_step "Running ${benchmark} benchmark: ${config_name}"

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

    scp "${PROJECT_ROOT}/eval/run_benchmark.py" gpu1:"${SGLANG_REPO}/eval/" 2>/dev/null

    ssh gpu1 "cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/run_benchmark.py \
            --benchmark ${benchmark} \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --num-samples ${NUM_SAMPLES} \
            --config-name ${config_name} \
            --output /tmp/${benchmark}_${config_name}.json" 2>&1

    # 拉取结果
    scp gpu1:"/tmp/${benchmark}_${config_name}.json" "${output_json}" 2>/dev/null

    log_info "Results saved to ${output_json}"
}

# 主流程
log_step "Quality Evaluation Experiment V2"
log_step "Benchmarks: ${BENCHMARKS}"
log_step "Samples per benchmark: ${NUM_SAMPLES}"
log_step "Configs: baseline, uniform-8bit, uniform-4bit, mixed-A/B/C/D"

# 配置列表
# 格式: "config_name:env_vars"
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

# 运行实验
IFS=',' read -ra BENCHMARK_LIST <<< "${BENCHMARKS}"

for benchmark in "${BENCHMARK_LIST[@]}"; do
    log_step "=========================================="
    log_step "Benchmark: ${benchmark}"
    log_step "=========================================="

    for config_entry in "${CONFIGS[@]}"; do
        config_name="${config_entry%%:*}"
        config_env="${config_entry#*:}"

        if [ "${config_name}" = "baseline" ]; then
            config_env=""
        fi

        run_benchmark "${config_name}" "${config_env}" "${benchmark}"
    done
done

log_step "=========================================="
log_step "All quality experiments completed!"
log_step "=========================================="

# 汇总结果
log_step "Summary:"
echo ""
printf "%-15s %-12s %10s %10s\n" "Benchmark" "Config" "Accuracy" "Correct"
echo "--------------------------------------------------------"

for benchmark in "${BENCHMARK_LIST[@]}"; do
    for config_name in baseline uniform-8bit uniform-4bit mixed-A mixed-B mixed-C mixed-D; do
        json="${RESULTS_DIR}/${benchmark}_${config_name}.json"
        if [ -f "$json" ]; then
            accuracy=$(jq -r '.accuracy * 100 // 0' "$json" 2>/dev/null || echo "0")
            correct=$(jq -r '.correct // 0' "$json" 2>/dev/null || echo "0")
            total=$(jq -r '.total // 0' "$json" 2>/dev/null || echo "0")
            printf "%-15s %-12s %9.1f%% %6s/%s\n" "$benchmark" "$config_name" "$accuracy" "$correct" "$total"
        fi
    done
    echo ""
done
