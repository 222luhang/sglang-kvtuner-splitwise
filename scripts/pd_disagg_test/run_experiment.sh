#!/bin/bash
# ============================================================================
# run_experiment.sh — 一键执行实验 3.1 (TTFT) 和 3.2 (质量评估)
#
# 用法:
#   # 实验 3.1: TTFT 测试 (baseline)
#   bash scripts/pd_disagg_test/run_experiment.sh ttft --config baseline
#
#   # 实验 3.1: TTFT 测试 (8-bit)
#   bash scripts/pd_disagg_test/run_experiment.sh ttft --config quant-8bit
#
#   # 实验 3.1: TTFT 测试 (所有配置)
#   bash scripts/pd_disagg_test/run_experiment.sh ttft --all
#
#   # 实验 3.2: 质量评估 (GSM8K, baseline)
#   bash scripts/pd_disagg_test/run_experiment.sh quality --benchmark gsm8k --config baseline
#
#   # 实验 3.2: 质量评估 (所有基准, quant-8bit)
#   bash scripts/pd_disagg_test/run_experiment.sh quality --all-benchmarks --config quant-8bit
#
# 环境变量覆盖:
#   PREFILL_HOST / DECODE_HOST  — 服务 IP (默认: 10.60.23.70 / 10.60.30.66)
#   NUM_RUNS                    — TTFT 每组重复次数 (默认: 5)
#   NUM_SAMPLES                 — 质量评估样本数 (默认: 50 快速验证)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
EVAL_DIR="${PROJECT_ROOT}/eval"

PREFILL_HOST="${PREFILL_HOST:-10.60.23.70}"
DECODE_HOST="${DECODE_HOST:-10.60.30.66}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
NUM_RUNS="${NUM_RUNS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

# 配置矩阵: 配置名 → 服务启动的环境变量
CONFIG_MATRIX="baseline:  quant-8bit:ENABLE_TRANSFER_QUANT=true,TRANSFER_QUANT_BITS=8  quant-4bit:ENABLE_TRANSFER_QUANT=true,TRANSFER_QUANT_BITS=4  mixed-B:ENABLE_TRANSFER_QUANT=true,TRANSFER_QUANT_BITS=4,KVTUNER_LAYER_BITS=[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
log_step()  { echo -e "\033[0;34m==>\033[0m $1"; }

# ---------------------------------------------------------------------------
# 解析实验配置
# ---------------------------------------------------------------------------

get_config_env() {
    local config_name=$1
    echo "$CONFIG_MATRIX" | tr ' ' '\n' | grep "^${config_name}:" | cut -d: -f2
}

get_all_configs() {
    echo "$CONFIG_MATRIX" | tr ' ' '\n' | cut -d: -f1
}

# ---------------------------------------------------------------------------
# 实验 3.1: TTFT
# ---------------------------------------------------------------------------

run_ttft() {
    local config_name=$1
    local output_dir="${RESULTS_DIR}/exp1_ttft"
    local output_csv="${output_dir}/${config_name}.csv"

    local config_env
    config_env=$(get_config_env "$config_name")
    if [ -z "$config_env" ]; then
        log_error "Unknown config: $config_name"
        return 1
    fi

    log_step "Starting TTFT experiment: config=${config_name}"
    log_info "Output: ${output_csv}"
    log_info "Runs per input: ${NUM_RUNS}"
    log_info ""

    python3 "${EVAL_DIR}/bench_ttft.py" \
        --prefill-host "${PREFILL_HOST}" \
        --prefill-port "${PREFILL_PORT}" \
        --decode-host "${DECODE_HOST}" \
        --decode-port "${DECODE_PORT}" \
        --bootstrap-port "${BOOTSTRAP_PORT}" \
        --config-name "${config_name}" \
        --num-runs "${NUM_RUNS}" \
        --output "${output_csv}"
}

# ---------------------------------------------------------------------------
# 实验 3.2: 质量评估
# ---------------------------------------------------------------------------

run_quality() {
    local benchmark=$1
    local config_name=$2
    local output_dir="${RESULTS_DIR}/exp2_quality"
    local output_json="${output_dir}/${config_name}_${benchmark}.json"

    log_step "Starting quality experiment: benchmark=${benchmark} config=${config_name}"
    log_info "Output: ${output_json}"
    log_info "Samples: ${NUM_SAMPLES}"
    log_info ""

    python3 "${EVAL_DIR}/run_benchmark.py" \
        --benchmark "${benchmark}" \
        --prefill-host "${PREFILL_HOST}" \
        --prefill-port "${PREFILL_PORT}" \
        --decode-host "${DECODE_HOST}" \
        --decode-port "${DECODE_PORT}" \
        --bootstrap-port "${BOOTSTRAP_PORT}" \
        --config-name "${config_name}" \
        --num-samples "${NUM_SAMPLES}" \
        --output "${output_json}"
}

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $0 <experiment> [options]

Experiments:
  ttft     Run TTFT benchmark (Experiment 3.1)
  quality  Run quality evaluation (Experiment 3.2)

TTFT Options:
  --config NAME      Run specific config (baseline, quant-8bit, quant-4bit, mixed-B)
  --all              Run all configs
  --input-types LIST  Input types (e.g., "short,medium,long")
  --no-wait          Skip health check

Quality Options:
  --benchmark NAME   Run specific benchmark (gsm8k, mmlu, hellaswag)
  --config NAME      Config name for labeling output
  --all-benchmarks   Run all benchmarks (gsm8k, mmlu, hellaswag)
  --num-samples N    Number of samples (default: 50)

Common Options:
  --dry-run          Show what would be run without executing
EOF
}

# Parse arguments
EXPERIMENT="${1:-}"
shift || true

if [ -z "$EXPERIMENT" ]; then
    usage
    exit 1
fi

case "$EXPERIMENT" in
    ttft)
        CONFIG_ARG=""
        INPUT_TYPES=""
        NO_WAIT=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --config) CONFIG_ARG="$2"; shift 2 ;;
                --all) CONFIG_ARG="__ALL__"; shift ;;
                --input-types) INPUT_TYPES="$2"; shift 2 ;;
                --no-wait) NO_WAIT="--no-wait"; shift ;;
                --dry-run) echo "[DRY RUN] Would execute TTFT experiment"; exit 0 ;;
                *) log_error "Unknown option: $1"; usage; exit 1 ;;
            esac
        done

        if [ "$CONFIG_ARG" = "__ALL__" ]; then
            for cfg in $(get_all_configs); do
                echo ""
                run_ttft "$cfg"
            done
        elif [ -n "$CONFIG_ARG" ]; then
            if [ -n "$INPUT_TYPES" ]; then
                NO_WAIT="$NO_WAIT" python3 "${EVAL_DIR}/bench_ttft.py" \
                    --prefill-host "${PREFILL_HOST}" --prefill-port "${PREFILL_PORT}" \
                    --decode-host "${DECODE_HOST}" --decode-port "${DECODE_PORT}" \
                    --bootstrap-port "${BOOTSTRAP_PORT}" \
                    --config-name "${CONFIG_ARG}" --num-runs "${NUM_RUNS}" \
                    --input-types "${INPUT_TYPES}" \
                    --output "${RESULTS_DIR}/exp1_ttft/${CONFIG_ARG}.csv"
            else
                run_ttft "$CONFIG_ARG"
            fi
        else
            usage
            exit 1
        fi
        ;;

    quality)
        BENCH_ARG=""
        CONFIG_ARG="baseline"
        while [ $# -gt 0 ]; do
            case "$1" in
                --benchmark) BENCH_ARG="$2"; shift 2 ;;
                --all-benchmarks) BENCH_ARG="__ALL__"; shift ;;
                --config) CONFIG_ARG="$2"; shift 2 ;;
                --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
                --dry-run) echo "[DRY RUN] Would execute quality experiment"; exit 0 ;;
                *) log_error "Unknown option: $1"; usage; exit 1 ;;
            esac
        done

        if [ "$BENCH_ARG" = "__ALL__" ]; then
            for bench in gsm8k mmlu hellaswag; do
                echo ""
                run_quality "$bench" "$CONFIG_ARG"
            done
        elif [ -n "$BENCH_ARG" ]; then
            run_quality "$BENCH_ARG" "$CONFIG_ARG"
        else
            usage
            exit 1
        fi
        ;;

    *)
        usage
        exit 1
        ;;
esac
