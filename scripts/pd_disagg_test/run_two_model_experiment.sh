#!/bin/bash
# ============================================================================
# 两种模型对比实验 (Qwen2.5-7B vs Gemma2-27B)
# 在 GPU1 上运行 benchmark，通过 SSH 执行
#
# 用法: bash run_two_model_experiment.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/two_model_comparison"

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

NUM_RUNS="${NUM_RUNS:-5}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

run_experiment() {
    local model_name=$1
    local config_file=$2
    local exp_name=$3
    local exp_env=$4

    log_step "=========================================="
    log_step "Model: ${model_name}, Config: ${exp_name}"
    log_step "=========================================="

    # 停止旧服务
    log_info "Stopping old services..."
    (cd "${SCRIPT_DIR}" && ./pd_test.sh stop) || true
    sleep 3

    # 启动新服务
    log_info "Starting services with ${exp_name} config..."
    cd "${SCRIPT_DIR}"
    CONFIG_FILE="${config_file}" ${exp_env} ./pd_test.sh start

    # 等待服务稳定
    log_info "Waiting for services to stabilize..."
    sleep 5

    # 在 GPU1 上运行 TTFT benchmark
    local output_csv="${RESULTS_DIR}/${model_name}_${exp_name}.csv"
    log_info "Running TTFT benchmark (${NUM_RUNS} runs per input type)..."

    # 从 config_file 获取 PREFILL_IP 和 DECODE_IP
    source "${SCRIPT_DIR}/${config_file}"

    # 同步 bench_ttft.py 到 GPU1
    scp "${PROJECT_ROOT}/eval/bench_ttft.py" gpu1:"${SGLANG_REPO}/eval/"

    # 在 GPU1 上执行 benchmark
    ssh gpu1 "cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_ttft.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --config-name ${model_name}_${exp_name} \
            --num-runs ${NUM_RUNS} \
            --output /tmp/${model_name}_${exp_name}.csv" 2>&1

    # 拉取结果文件
    scp gpu1:"/tmp/${model_name}_${exp_name}.csv" "${output_csv}"

    log_info "Results saved to ${output_csv}"
    echo ""
}

# 主流程
log_step "Starting two-model comparison experiment"
log_step "Models: qwen2.5-7b, gemma2-27b"
log_step "Configs per model: baseline, quant-8bit, quant-4bit, mixed-b"
log_step "Runs per input type: ${NUM_RUNS}"
echo ""

# Qwen2.5-7B 实验
run_experiment "qwen2.5-7b" "configs/default.sh" "baseline" ""
run_experiment "qwen2.5-7b" "configs/default.sh" "quant-8bit" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
run_experiment "qwen2.5-7b" "configs/default.sh" "quant-4bit" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
run_experiment "qwen2.5-7b" "configs/default.sh" "mixed-b" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4 KVTUNER_LAYER_BITS=[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]"

# Gemma2-27B 实验
run_experiment "gemma2-27b" "configs/gemma2-27b.sh" "baseline" ""
run_experiment "gemma2-27b" "configs/gemma2-27b.sh" "quant-8bit" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=8"
run_experiment "gemma2-27b" "configs/gemma2-27b.sh" "quant-4bit" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4"
run_experiment "gemma2-27b" "configs/gemma2-27b.sh" "mixed-b" "ENABLE_TRANSFER_QUANT=true TRANSFER_QUANT_BITS=4 KVTUNER_LAYER_BITS=[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]"

# 汇总结果
log_step "=========================================="
log_step "All experiments completed!"
log_step "=========================================="
echo ""
log_info "Results saved in: ${RESULTS_DIR}"
ls -la "${RESULTS_DIR}"/*.csv

# 生成汇总统计
log_step "Summary:"
for csv in "${RESULTS_DIR}"/*.csv; do
    if [ -f "$csv" ]; then
        name=$(basename "$csv" .csv)
        success=$(tail -n +2 "$csv" | grep -c ",true," || echo 0)
        total=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
        log_info "  ${name}: ${success}/${total} successful"
    fi
done
