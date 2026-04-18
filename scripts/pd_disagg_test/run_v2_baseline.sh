#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/two_model_comparison_v2"

NUM_RUNS="${NUM_RUNS:-5}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

log_step "V2 Baseline Experiment (sync optimization, no transfer quant)"
log_step "Runs per input type: ${NUM_RUNS}"

# Stop old services
log_info "Stopping old services..."
(cd "${SCRIPT_DIR}" && ./pd_test.sh stop) || true
sleep 3

# Start services with baseline config
log_info "Starting services with baseline config..."
cd "${SCRIPT_DIR}"
CONFIG_FILE="configs/default.sh" ./pd_test.sh start

# Wait for services
log_info "Waiting for services to stabilize..."
sleep 5

# Run benchmark
log_info "Running TTFT benchmark (${NUM_RUNS} runs per input type)..."

source "${SCRIPT_DIR}/configs/default.sh"

# Sync bench_ttft.py to GPU1
scp "${PROJECT_ROOT}/eval/bench_ttft.py" gpu1:"${SGLANG_REPO}/eval/"

# Run benchmark
ssh gpu1 "cd ${SGLANG_REPO} && \
    source ${VENV_DIR}/bin/activate && \
    python3 eval/bench_ttft.py \
        --prefill-host ${PREFILL_IP} \
        --prefill-port ${PREFILL_PORT} \
        --decode-host ${DECODE_IP} \
        --decode-port ${DECODE_PORT} \
        --bootstrap-port ${BOOTSTRAP_PORT} \
        --config-name qwen2.5-7b_baseline_v2 \
        --num-runs ${NUM_RUNS} \
        --output /tmp/qwen2.5-7b_baseline_v2.csv" 2>&1

# Pull results
scp gpu1:"/tmp/qwen2.5-7b_baseline_v2.csv" "${RESULTS_DIR}/qwen2.5-7b_baseline_v2.csv"

log_info "Results saved to ${RESULTS_DIR}/qwen2.5-7b_baseline_v2.csv"
