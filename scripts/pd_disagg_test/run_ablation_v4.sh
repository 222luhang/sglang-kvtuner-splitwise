#!/bin/bash
# ============================================================================
# 消融实验 v4 — 交替 A/B 测试，消除冷启动差异
#
# 2 次服务启动:
#   Start 1: 无量化 (QUANT_DISABLE=1) — 仅跑 baseline (A1)
#   Start 2: 4bit 量化 — A1(baseline模拟)/A5/A6/A2/A3 交替
#
# Start 2 中通过 QUANT_DISABLE 切换 baseline vs 量化，确保公平对比。
# 所有配置在同一服务启动内交替运行，消除 GPU 漂移。
#
# 每轮 6 runs 交替:
#   quant_pipe → quant_off → quant_pipe → quant_off_triton → quant_off_pytorch → quant_pipe
#
# 配置:
#   A1: baseline (QUANT_DISABLE=1, 无量化)
#   A5: 4bit Triton, 无 pipeline
#   A6: 4bit Triton, 有 pipeline
#   A2: 4bit PyTorch, 无 pipeline (DISABLE_TRITON=1)
#   A3: 4bit PyTorch, 有 pipeline (DISABLE_TRITON=1)
#
# 用法:
#   bash run_ablation_v4.sh
#   PROMPT_TOKENS=512 CONCURRENCY=4 bash run_ablation_v4.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NUM_RUNS="${NUM_RUNS:-5}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-8}"
PROMPT_TYPE="${PROMPT_TYPE:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PROMPT_TOKENS="${PROMPT_TOKENS:-0}"
TRANSFER_QUANT_BITS="${TRANSFER_QUANT_BITS:-4}"

SEQ_TAG="medium"
if [ "${PROMPT_TOKENS}" -gt 0 ] 2>/dev/null; then
    SEQ_TAG="${PROMPT_TOKENS}tok"
fi
RESULTS_DIR="${PROJECT_ROOT}/results/ablation_v4_${SEQ_TAG}"
mkdir -p "${RESULTS_DIR}"

source "${SCRIPT_DIR}/configs/default.sh"

CONFIG_PATH="/tmp/sglang_ablation.cfg"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

write_config() {
    local content=$1
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${host}" "printf '%s' '${content}' > ${CONFIG_PATH}"
    done
}

run_benchmark() {
    local config_id=$1
    local run_idx=$2
    local output_json="${RESULTS_DIR}/${config_id}_run${run_idx}.json"

    if [ -f "$output_json" ] && [ -s "$output_json" ]; then
        local existing
        existing=$(jq -r '.results.successful // 0' "$output_json" 2>/dev/null || echo "0")
        if [ "$existing" -gt 0 ]; then
            log_info "跳过 ${config_id} run${run_idx} (已有结果)"
            return 0
        fi
    fi

    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${PREFILL_HOST}" "
        cd ${SGLANG_REPO} && source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --concurrency ${CONCURRENCY} --num-requests ${NUM_REQUESTS} \
            ${PROMPT_TOKENS:+--prompt-tokens ${PROMPT_TOKENS}} \
            ${PROMPT_TYPE:+--prompt-type ${PROMPT_TYPE}} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${config_id}_run${run_idx} --output /tmp/${config_id}_run${run_idx}.json
    " 2>&1 | tail -10

    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${PREFILL_HOST}:/tmp/${config_id}_run${run_idx}.json" "${output_json}" 2>/dev/null || true
}

# Config presets (written to /tmp/sglang_ablation.cfg)
# QUANT_DISABLE=1 → 无量化 (baseline)
# QUANT_DISABLE=0 + PIPELINE_SEND=0 + DISABLE_TRITON=0 → 4bit Triton no pipe (A5)
# QUANT_DISABLE=0 + PIPELINE_SEND=1 + DISABLE_TRITON=0 → 4bit Triton pipe (A6)
# QUANT_DISABLE=0 + PIPELINE_SEND=0 + DISABLE_TRITON=1 → 4bit PyTorch no pipe (A2)
# QUANT_DISABLE=0 + PIPELINE_SEND=1 + DISABLE_TRITON=1 → 4bit PyTorch pipe (A3)

A1_CFG="QUANT_DISABLE=1
PIPELINE_SEND=0
DISABLE_TRITON=0"
A5_CFG="QUANT_DISABLE=0
PIPELINE_SEND=0
DISABLE_TRITON=0"
A6_CFG="QUANT_DISABLE=0
PIPELINE_SEND=1
DISABLE_TRITON=0"
A2_CFG="QUANT_DISABLE=0
PIPELINE_SEND=0
DISABLE_TRITON=1"
A3_CFG="QUANT_DISABLE=0
PIPELINE_SEND=1
DISABLE_TRITON=1"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     消融实验 v4 — 交替 A/B 测试 (含 baseline)          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  量化: ${TRANSFER_QUANT_BITS}bit (服务启动时设定)"
echo "  Baseline: 通过 QUANT_DISABLE=1 运行时切换"
echo "  重复次数: ${NUM_RUNS} 轮 × 5 runs/轮 = $((NUM_RUNS * 5)) 总 runs"
echo "  并发: ${CONCURRENCY}, 请求数: ${NUM_REQUESTS}"
echo "  Prompt: ${PROMPT_TYPE} ${PROMPT_TOKENS:+(${PROMPT_TOKENS} tokens)}"
echo "  结果目录: ${RESULTS_DIR}"
echo ""

cd "${SCRIPT_DIR}"
export CONFIG_FILE="configs/default.sh"
export LOG_LEVEL="info"

# ==== 单次服务启动: 量化服务 + 运行时切换 baseline/量化 ====
log_step "===== 启动 ${TRANSFER_QUANT_BITS}bit 量化服务 (运行时切换 baseline) ====="
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="${TRANSFER_QUANT_BITS}"
KVTUNER_LAYER_BITS=""
./pd_test.sh stop || true; sleep 2
./pd_test.sh start || { log_error "启动失败"; exit 1; }
sleep 5

# Warmup: 各配置跑一次
log_info "Warmup: 各配置一次..."
write_config "${A1_CFG}"
run_benchmark "A1" "warmup" || true
write_config "${A5_CFG}"
run_benchmark "A5" "warmup" || true
write_config "${A6_CFG}"
run_benchmark "A6" "warmup" || true
write_config "${A2_CFG}"
run_benchmark "A2" "warmup" || true
write_config "${A3_CFG}"
run_benchmark "A3" "warmup" || true

log_step "=========================================="
log_step "交替模式: A6→A1→A6→A5→A3→A2 (5 runs/轮)"
log_step "=========================================="

for run_idx in $(seq 1 "${NUM_RUNS}"); do
    log_info "--- 轮 ${run_idx} ---"

    # A6: pipe first
    write_config "${A6_CFG}"
    run_benchmark "A6" "${run_idx}"

    # A1: baseline (no quant)
    write_config "${A1_CFG}"
    run_benchmark "A1" "${run_idx}"

    # A5: no pipe triton
    write_config "${A5_CFG}"
    run_benchmark "A5" "${run_idx}"

    # A3: pipe pytorch
    write_config "${A3_CFG}"
    run_benchmark "A3" "${run_idx}"

    # A2: no pipe pytorch
    write_config "${A2_CFG}"
    run_benchmark "A2" "${run_idx}"

    echo ""
done

# 清理
./pd_test.sh stop || true
for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${host}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
done

# ==== 汇总 ====
echo ""
log_step "===== 汇总 (交替平均) ====="
echo ""

python3 << PYTHON_SUMMARY
import json, math
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
num_runs = ${NUM_RUNS}
quant_bits = '${TRANSFER_QUANT_BITS}'

configs = {}
for cfg_id in ['A1', 'A5', 'A6', 'A2', 'A3']:
    times = []
    for run in range(1, num_runs + 1):
        f = results_dir / f'{cfg_id}_run{run}.json'
        if f.exists():
            d = json.load(open(f))
            ttft = d['results']['mean_ttft_ms']
            times.append(ttft)
    if times:
        configs[cfg_id] = times

labels = {
    'A1': f'baseline (no quant, fp16)',
    'A5': f'{quant_bits}bit Triton no pipe',
    'A6': f'{quant_bits}bit Triton pipe',
    'A2': f'{quant_bits}bit PyTorch no pipe',
    'A3': f'{quant_bits}bit PyTorch pipe',
}

baseline_avg = sum(configs['A1']) / len(configs['A1']) if 'A1' in configs else None

print(f"{'Config':<30} {'Avg TTFT':>12} {'vs A1':>12} {'Runs':>5}")
print('-' * 61)
for cfg_id in ['A1', 'A5', 'A6', 'A2', 'A3']:
    if cfg_id not in configs:
        continue
    times = configs[cfg_id]
    avg = sum(times) / len(times)
    std = math.sqrt(sum((t - avg)**2 for t in times) / len(times))
    if baseline_avg and cfg_id != 'A1':
        diff = avg - baseline_avg
        pct = (diff / baseline_avg * 100) if baseline_avg > 0 else 0
        sign = '+' if diff >= 0 else ''
        print(f'{labels[cfg_id]:<30} {avg:>7.1f}+/-{std:.1f} {sign}{diff:>8.1f} ({sign}{pct:.1f}%) {len(times):>5}')
    else:
        print(f'{labels[cfg_id]:<30} {avg:>7.1f}+/-{std:.1f} {"":>12} {len(times):>5}')

print()

# Per-round detail
print(f"{'Config':<30} {'Run':>4} {'TTFT':>10}")
print('-' * 46)
for cfg_id in ['A1', 'A5', 'A6', 'A2', 'A3']:
    if cfg_id not in configs:
        continue
    for i, t in enumerate(configs[cfg_id], 1):
        print(f'{labels[cfg_id]:<30} {i:>4} {t:>10.1f}')
    print()

# Pipeline comparison
print(f"{'No Pipe':<30} {'Pipe':<30} {'Pipe-NoPipe':>12}")
print('-' * 74)
for no_pipe, pipe in [('A5', 'A6'), ('A2', 'A3')]:
    if no_pipe in configs and pipe in configs:
        avg_np = sum(configs[no_pipe]) / len(configs[no_pipe])
        avg_p = sum(configs[pipe]) / len(configs[pipe])
        diff = avg_p - avg_np
        sign = '+' if diff >= 0 else ''
        print(f'{labels[no_pipe]:<30} {labels[pipe]:<30} {sign}{diff:>10.1f}ms')

print()
print(f'所有结果保存在: {results_dir}/')
PYTHON_SUMMARY
