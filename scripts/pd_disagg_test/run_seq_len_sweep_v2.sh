#!/bin/bash
# ============================================================================
# 序列长度扫描实验 v2 — 消除热漂移
#
# 每个序列长度独立重启服务 + warmup, 确保每个 tok 的数据在 GPU
# 热平衡态下采集。
#
# 实验配置 (6 组):
#   baseline: 无量化 (QUANT_DISABLE=1)
#   8bit:     8bit 均匀量化
#   4bit:     4bit 均匀量化
#   mixed-A:  首尾 5 层 8bit, 中间 18 层 4bit
#   mixed-B:  首尾 3+2 层 8bit, 中间 23 层 4bit
#   mixed-C:  首尾 5 层 4bit, 中间 18 层 8bit
#
# 实验流程 (每个 tok):
#   Phase 1: 启动 4bit 均匀量化服务
#            → 3 轮 warmup → baseline/4bit 交替 3 轮
#   Phase 2: 启动 8bit 均匀量化服务
#            → 3 轮 warmup → 8bit/baseline 交替 3 轮
#   Phase 3: 启动 mixed-A 服务
#            → 3 轮 warmup → mixed-A 交替 3 轮
#   Phase 4: 启动 mixed-B 服务
#            → 3 轮 warmup → mixed-B 交替 3 轮
#   Phase 5: 启动 mixed-C 服务
#            → 3 轮 warmup → mixed-C 交替 3 轮
#
# 用法:
#   bash run_seq_len_sweep_v2.sh
#   SEQ_LENGTHS="128 512 1024" bash run_seq_len_sweep_v2.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/seq_len_sweep_v2"
mkdir -p "${RESULTS_DIR}"

source "${SCRIPT_DIR}/configs/default.sh"

NUM_RUNS="${NUM_RUNS:-3}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEQ_LENGTHS="${SEQ_LENGTHS:-128 256 512 1024 2048 3072 4096}"

CONFIG_PATH="/tmp/sglang_ablation.cfg"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---- 层级混合量化配置 ----

get_layer_bits() {
    local strategy=$1
    case "$strategy" in
        "mixed-A") echo "[8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8]" ;;
        "mixed-B") echo "[8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,8]" ;;
        "mixed-C") echo "[4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4]" ;;
        *) echo "" ;;
    esac
}

# ---- 运行时配置文件 ----

write_config() {
    local content=$1
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${host}" "printf '%s' '${content}' > ${CONFIG_PATH}"
    done
}

BASELINE_CFG="QUANT_DISABLE=1"

QUANT_CFG="QUANT_DISABLE=0"

# ---- Benchmark ----

run_benchmark() {
    local config_id=$1
    local run_idx=$2
    local prompt_tokens=$3
    local output_json="${RESULTS_DIR}/${config_id}_tok${prompt_tokens}_run${run_idx}.json"

    if [ -f "$output_json" ] && [ -s "$output_json" ]; then
        local existing
        existing=$(jq -r '.results.successful // 0' "$output_json" 2>/dev/null || echo "0")
        if [ "$existing" -gt 0 ]; then
            log_info "  跳过 ${config_id} tok=${prompt_tokens} ${run_idx}"
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
            --prompt-tokens ${prompt_tokens} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${config_id}_tok${prompt_tokens}_${run_idx} \
            --output /tmp/${config_id}_tok${prompt_tokens}_${run_idx}.json
    " 2>&1 | tail -5

    scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${PREFILL_HOST}:/tmp/${config_id}_tok${prompt_tokens}_${run_idx}.json" \
        "${output_json}" 2>/dev/null || true
}

# ---- 启动服务 (带环境变量) ----

start_service() {
    local quant_bits=${1:-}
    local layer_bits=${2:-}

    cd "${SCRIPT_DIR}"
    export CONFIG_FILE="configs/default.sh"
    export LOG_LEVEL="info"
    export ENABLE_TRANSFER_QUANT="true"
    export TRANSFER_QUANT_BITS="${quant_bits:-4}"
    export KVTUNER_LAYER_BITS="${layer_bits:-}"
    export SGLANG_TCP_PIPELINE_SEND="1"
    export SGLANG_DISABLE_TRITON="0"

    ./pd_test.sh stop || true
    sleep 2
    ./pd_test.sh start || { log_error "启动失败 (bits=${quant_bits}, layers=${layer_bits})"; return 1; }
    sleep 5
}

# ---- Warmup ----

run_warmup() {
    local phase_id=$1
    local quant_cfg=$2
    local tok=$3

    log_info "  Warmup (3 轮, phase=${phase_id})..."
    for i in 1 2 3; do
        write_config "${quant_cfg}"
        run_benchmark "warmup_${phase_id}_q" "w${i}" "${tok}" || true
        write_config "${BASELINE_CFG}"
        run_benchmark "warmup_${phase_id}_bl" "w${i}" "${tok}" || true
    done
    log_info "  Warmup 完成"
}

# ---- 交替记录 (3 轮 B-A-B-A) ----

run_alternating() {
    local quant_id=$1
    local quant_cfg=$2
    local tok=$3

    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        log_info "  --- ${quant_id} tok=${tok} 轮 ${run_idx} ---"

        # B (quant)
        write_config "${quant_cfg}"
        run_benchmark "${quant_id}" "${run_idx}_b1" "${tok}"

        # A (baseline)
        write_config "${BASELINE_CFG}"
        run_benchmark "baseline" "${run_idx}_a1" "${tok}"

        # B (quant)
        write_config "${quant_cfg}"
        run_benchmark "${quant_id}" "${run_idx}_b2" "${tok}"

        # A (baseline)
        write_config "${BASELINE_CFG}"
        run_benchmark "baseline" "${run_idx}_a2" "${tok}"
    done
}

# ---- 仅量化记录 (用于 mixed configs, 无 baseline 对照) ----

run_quant_only() {
    local quant_id=$1
    local quant_cfg=$2
    local tok=$3

    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        log_info "  --- ${quant_id} tok=${tok} 轮 ${run_idx} ---"
        write_config "${quant_cfg}"
        run_benchmark "${quant_id}" "${run_idx}_b1" "${tok}"
        write_config "${BASELINE_CFG}"
        run_benchmark "baseline" "${run_idx}_a1" "${tok}"
        write_config "${quant_cfg}"
        run_benchmark "${quant_id}" "${run_idx}_b2" "${tok}"
        write_config "${BASELINE_CFG}"
        run_benchmark "baseline" "${run_idx}_a2" "${tok}"
    done
}

# ==== 主流程 ====

echo ""
echo "================================================================"
echo "  序列长度扫描实验 v2 — 消除热漂移"
echo "  配置: baseline / 8bit / 4bit / mixed-A / mixed-B / mixed-C"
echo "  序列长度: ${SEQ_LENGTHS}"
echo "  重复次数: ${NUM_RUNS} 轮 x 4 runs/轮"
echo "  并发: ${CONCURRENCY}, 请求数: ${NUM_REQUESTS}"
echo "  结果目录: ${RESULTS_DIR}"
echo "================================================================"
echo ""

cd "${SCRIPT_DIR}"
TOTAL_START=$(date +%s)

# Clean old results
rm -rf "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

for tok in ${SEQ_LENGTHS}; do
    TOK_START=$(date +%s)

    echo ""
    log_step "========================================================"
    log_step "tok=${tok} (共 5 个服务启动)"
    log_step "========================================================"
    echo ""

    # ---- Phase 1: 4bit 均匀量化 (含 baseline 对照) ----
    log_step "[Phase 1/5] 4bit 均匀量化 + baseline 交替"
    start_service "4" ""
    run_warmup "p1_4bit" "${QUANT_CFG}" "${tok}"
    run_alternating "4bit" "${QUANT_CFG}" "${tok}"

    # ---- Phase 2: 8bit 均匀量化 ----
    log_step "[Phase 2/5] 8bit 均匀量化 + baseline 交替"
    start_service "8" ""
    run_warmup "p2_8bit" "${QUANT_CFG}" "${tok}"
    run_alternating "8bit" "${QUANT_CFG}" "${tok}"

    # ---- Phase 3: mixed-A ----
    log_step "[Phase 3/5] mixed-A (首尾5层8bit, 中间18层4bit)"
    start_service "4" "$(get_layer_bits 'mixed-A')"
    run_warmup "p3_mixedA" "${QUANT_CFG}" "${tok}"
    run_quant_only "mixed-A" "${QUANT_CFG}" "${tok}"

    # ---- Phase 4: mixed-B ----
    log_step "[Phase 4/5] mixed-B (首尾3+2层8bit, 中间23层4bit)"
    start_service "4" "$(get_layer_bits 'mixed-B')"
    run_warmup "p4_mixedB" "${QUANT_CFG}" "${tok}"
    run_quant_only "mixed-B" "${QUANT_CFG}" "${tok}"

    # ---- Phase 5: mixed-C ----
    log_step "[Phase 5/5] mixed-C (首尾5层4bit, 中间18层8bit)"
    start_service "4" "$(get_layer_bits 'mixed-C')"
    run_warmup "p5_mixedC" "${QUANT_CFG}" "${tok}"
    run_quant_only "mixed-C" "${QUANT_CFG}" "${tok}"

    # 清理
    ./pd_test.sh stop || true

    TOK_END=$(date +%s)
    TOK_ELAPSED=$(( TOK_END - TOK_START ))
    log_info "tok=${tok} 完成 (耗时 ${TOK_ELAPSED}s)"
done

# 清理远程配置文件
for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${host}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
done

./pd_test.sh stop || true

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

# ==== 汇总 ====

echo ""
log_step "===== 汇总 ====="
echo ""

export RESULTS_DIR TOTAL_ELAPSED
python3 << 'PYTHON_SUMMARY'
import json, math, os
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results/seq_len_sweep_v2"))
seq_lengths = [128, 256, 512, 1024, 2048, 3072, 4096]
num_runs = int(os.environ.get("NUM_RUNS", "3"))

def load(name):
    f = results_dir / name
    if not f.exists() or f.stat().st_size == 0:
        return None
    try:
        return json.load(open(f))['results']['mean_ttft_ms']
    except:
        return None

def stats(vals):
    if not vals:
        return None, 0, 0
    a = sum(vals) / len(vals)
    s = math.sqrt(sum((x - a) ** 2 for x in vals) / len(vals))
    return a, s, len(vals)

print("=" * 100)
print("  序列长度扫描实验 v2 结果 (prefix caching disabled, 每 tok 独立重启 + warmup)")
print("=" * 100)
print()

configs = ['baseline', '4bit', '8bit', 'mixed-A', 'mixed-B', 'mixed-C']

for tok in seq_lengths:
    print(f"--- tok={tok} ---")
    print(f"{'Config':<12} {'Avg TTFT':>12} {'Std':>8} {'vs Baseline':>14} {'Runs':>6}")
    print("-" * 56)

    all_data = {}
    for cfg in configs:
        vals = []
        if cfg == 'baseline':
            # Collect from 4bit and 8bit phases
            for phase in ['4bit', '8bit']:
                for run in range(1, num_runs + 1):
                    for s in ['_a1', '_a2']:
                        v = load(f'baseline_tok{tok}_run{run}{s}.json')
                        if v is not None:
                            vals.append(v)
        else:
            for run in range(1, num_runs + 1):
                for s in ['_b1', '_b2']:
                    v = load(f'{cfg}_tok{tok}_run{run}{s}.json')
                    if v is not None:
                        vals.append(v)

        if vals:
            all_data[cfg] = vals

    baseline_avg = sum(all_data.get('baseline', [0])) / len(all_data.get('baseline', [1])) if 'baseline' in all_data else None

    for cfg in configs:
        if cfg not in all_data:
            print(f"{cfg:<12} {'N/A':>12}")
            continue
        avg, std, n = stats(all_data[cfg])
        if baseline_avg and cfg != 'baseline':
            diff = avg - baseline_avg
            pct = diff / baseline_avg * 100
            sign = '+' if diff >= 0 else ''
            print(f"{cfg:<12} {avg:>8.1f}ms {std:>6.1f} {sign}{diff:>8.1f} ({sign}{pct:.1f}%) {n:>6}")
        else:
            print(f"{cfg:<12} {avg:>8.1f}ms {std:>6.1f} {'':>14} {n:>6}")

    print()

# Summary table
print("=" * 100)
print("  汇总: 各 tok 下各配置 vs baseline 差值")
print("=" * 100)
print()
header = f"{'Tok':>6}"
for cfg in configs:
    if cfg != 'baseline':
        header += f" | {cfg + ' Δ':>14}"
print(header)
print("-" * (6 + 16 * (len(configs) - 1)))

for tok in seq_lengths:
    bl_vals = []
    for run in range(1, num_runs + 1):
        for s in ['_a1', '_a2']:
            v = load(f'baseline_tok{tok}_run{run}{s}.json')
            if v is not None:
                bl_vals.append(v)

    if not bl_vals:
        continue
    avg_bl = sum(bl_vals) / len(bl_vals)

    line = f"{tok:>6}"
    for cfg in configs:
        if cfg == 'baseline':
            continue
        vals = []
        for run in range(1, num_runs + 1):
            for s in ['_b1', '_b2']:
                v = load(f'{cfg}_tok{tok}_run{run}{s}.json')
                if v is not None:
                    vals.append(v)
        if vals:
            avg = sum(vals) / len(vals)
            diff = avg - avg_bl
            pct = diff / avg_bl * 100
            sign = '+' if diff >= 0 else ''
            line += f" | {sign}{diff:>6.0f}ms({sign}{pct:.1f}%)"
        else:
            line += f" | {'N/A':>14}"
    print(line)

print()
print(f"总耗时: {int(os.environ.get('TOTAL_ELAPSED', 0))}s")
print(f"结果保存在: {results_dir}/")
PYTHON_SUMMARY
