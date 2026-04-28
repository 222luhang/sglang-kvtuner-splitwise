#!/bin/bash
# ============================================================================
# 消融实验 v2 — 运行时配置切换，最小化服务重启
#
# 设计: 仅 2 次服务启动
#   Start 1: 无量化 baseline → A1 (1 config)
#   Start 2: 4bit 量化 → A4/A5/A6/A7 (4 configs, 运行时切换 Triton/Pipeline)
#
# 运行时切换通过 /tmp/sglang_ablation.cfg 实现 (SSH 写入两台机器)
#
# 7 配置 × 3 次重复, 固定 conc=8, 32 requests, medium prompt
#
# 配置矩阵:
#   A1: baseline (无量化)
#   A2: 8bit 量化, PyTorch, 无异步 pipeline   ← 需要 8bit 量化启动
#   A3: 8bit 量化, PyTorch, 有异步 pipeline   ← 需要 8bit 量化启动
#   A4: 4bit 量化, PyTorch, 无异步 pipeline
#   A5: 4bit 量化, Triton,  无异步 pipeline
#   A6: 4bit 量化, Triton,  有异步 pipeline (当前默认)
#   A7: mixed-C,   Triton,  有异步 pipeline
#
# 用法:
#   bash run_ablation_experiment.sh          # 运行全部
#   bash run_ablation_experiment.sh --dry-run # 只打印配置不执行
#   NUM_RUNS=1 bash run_ablation_experiment.sh  # 只跑 1 次
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/ablation_experiment"

mkdir -p "${RESULTS_DIR}"

# ---- 参数 ----
NUM_RUNS="${NUM_RUNS:-3}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
CONCURRENCY="${CONCURRENCY:-8}"
PROMPT_TYPE="${PROMPT_TYPE:-medium}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DRY_RUN="${DRY_RUN:-false}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# ---- 颜色 ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${BLUE}${BOLD}==>${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ---- 加载集群配置 ----
source "${SCRIPT_DIR}/configs/default.sh"

# 远程配置文件路径 (与 conn.py 中的 _CONFIG_PATH 一致)
CONFIG_PATH="/tmp/sglang_ablation.cfg"

# ---- 配置文件写入 ----
write_config_to_hosts() {
    local content=$1
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[dry-run] 写入配置文件到两台机器:"
        echo "  ${content}"
        return 0
    fi
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh ${SSH_OPTS} "${host}" "echo '${content}' > ${CONFIG_PATH}"
        log_info "已写入 ${host}:${CONFIG_PATH}"
    done
}

clear_config_on_hosts() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[dry-run] 清除远程配置文件"
        return 0
    fi
    for host in "${PREFILL_HOST}" "${DECODE_HOST}"; do
        ssh ${SSH_OPTS} "${host}" "rm -f ${CONFIG_PATH}" 2>/dev/null || true
    done
}

# ---- 服务启动/停止 ----
start_services() {
    cd "${SCRIPT_DIR}"
    export CONFIG_FILE="configs/default.sh"

    if [ "$DRY_RUN" = "true" ]; then
        log_info "[dry-run] 启动服务 (env: ENABLE_TRANSFER_QUANT=${ENABLE_TRANSFER_QUANT:-<unset>}, TRANSFER_QUANT_BITS=${TRANSFER_QUANT_BITS:-<unset>})"
        return 0
    fi

    ./pd_test.sh stop || true
    sleep 3
    ./pd_test.sh start || { log_error "服务启动失败"; return 1; }
    sleep 5
}

stop_services() {
    if [ "$DRY_RUN" = "false" ]; then
        (cd "${SCRIPT_DIR}" && ./pd_test.sh stop) || true
    fi
}

# ---- 单次 benchmark 运行 ----
run_benchmark() {
    local config_id=$1
    local run_idx=$2
    local output_json="${RESULTS_DIR}/${config_id}_run${run_idx}.json"

    # 断点续跑
    if [ -f "$output_json" ] && [ -s "$output_json" ]; then
        local existing_success
        existing_success=$(jq -r '.results.successful // 0' "$output_json" 2>/dev/null || echo "0")
        if [ "$existing_success" -gt 0 ]; then
            log_info "跳过 ${config_id} run${run_idx} (已有 ${existing_success} 成功结果)"
            return 0
        fi
    fi

    # 同步 benchmark 脚本到远程
    if [ "$DRY_RUN" = "false" ]; then
        scp ${SSH_OPTS} "${PROJECT_ROOT}/eval/bench_throughput.py" \
            "${PREFILL_HOST}:${SGLANG_REPO}/eval/" 2>/dev/null || true
    fi

    local remote_output="/tmp/${config_id}_run${run_idx}.json"
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[dry-run] 运行 benchmark: ${config_id} run${run_idx}"
        return 0
    fi

    ssh ${SSH_OPTS} "${PREFILL_HOST}" "
        cd ${SGLANG_REPO} && \
        source ${VENV_DIR}/bin/activate && \
        python3 eval/bench_throughput.py \
            --prefill-host ${PREFILL_IP} \
            --prefill-port ${PREFILL_PORT} \
            --decode-host ${DECODE_IP} \
            --decode-port ${DECODE_PORT} \
            --bootstrap-port ${BOOTSTRAP_PORT} \
            --concurrency ${CONCURRENCY} \
            --num-requests ${NUM_REQUESTS} \
            --prompt-type ${PROMPT_TYPE} \
            --max-new-tokens ${MAX_NEW_TOKENS} \
            --config-name ${config_id}_run${run_idx} \
            --output ${remote_output}
    " 2>&1 | tail -20

    scp ${SSH_OPTS} "${PREFILL_HOST}:${remote_output}" "${output_json}" 2>/dev/null || {
        log_error "结果拉取失败: ${config_id} run${run_idx}"
        return 1
    }

    log_info "结果: ${output_json}"
}

# ---- 运行一个配置 (warmup + N runs) ----
run_config() {
    local config_id=$1
    local label=$2
    local cfg_content=$3  # 配置文件内容

    log_step "=========================================="
    log_step "配置 ${config_id}: ${label}"
    log_step "=========================================="

    # 写入运行时配置
    write_config_to_hosts "${cfg_content}"

    # Warmup run (discarded)
    log_info "Warmup: ${config_id} ..."
    run_benchmark "${config_id}" "warmup" || log_warn "Warmup 失败 (非致命)"

    # 正式运行
    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        run_benchmark "${config_id}" "${run_idx}"
    done
    echo ""
}

# ---- 主流程 ----

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     消融实验 v2 — 运行时配置切换 (2 次服务启动)          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  配置数: 7 (A1-A7)"
echo "  重复次数: ${NUM_RUNS}"
echo "  并发: ${CONCURRENCY}, 请求数: ${NUM_REQUESTS}"
echo "  Prompt: ${PROMPT_TYPE}, max_new_tokens: ${MAX_NEW_TOKENS}"
echo "  结果目录: ${RESULTS_DIR}"
echo "  Dry run: ${DRY_RUN}"
echo ""

# ==== Start 1: 无量化 baseline (A1) ====
log_step "===== Start 1/2: 无量化 Baseline ====="
ENABLE_TRANSFER_QUANT=""
TRANSFER_QUANT_BITS=""
KVTUNER_LAYER_BITS=""
start_services
run_config "A1" "baseline (无量化)" "PIPELINE_SEND=0
DISABLE_TRITON=0"

# ==== Start 2: 4bit 量化 (A4-A7) ====
log_step "===== Start 2/2: 4bit 量化 (A4→A5→A6→A7) ====="
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="4"
KVTUNER_LAYER_BITS=""
start_services

# A4: 4bit PyTorch, 无 pipeline
run_config "A4" "4bit PyTorch, 无 pipeline" "PIPELINE_SEND=0
DISABLE_TRITON=1"

# A5: 4bit Triton, 无 pipeline
run_config "A5" "4bit Triton, 无 pipeline" "PIPELINE_SEND=0
DISABLE_TRITON=0"

# A6: 4bit Triton, 有 pipeline (当前默认)
run_config "A6" "4bit Triton, 有 pipeline" "PIPELINE_SEND=1
DISABLE_TRITON=0"

# A7: mixed-C Triton, 有 pipeline
MIXED_C_BITS="[4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4]"
run_config "A7" "mixed-C Triton, 有 pipeline" "PIPELINE_SEND=1
DISABLE_TRITON=0"

# ==== Start 3: 8bit 量化 (A2-A3) ====
log_step "===== Start 3/3: 8bit 量化 (A2→A3) ====="
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="8"
KVTUNER_LAYER_BITS=""
start_services

# A2: 8bit PyTorch, 无 pipeline
run_config "A2" "8bit PyTorch, 无 pipeline" "PIPELINE_SEND=0
DISABLE_TRITON=0"

# A3: 8bit PyTorch, 有 pipeline
run_config "A3" "8bit PyTorch, 有 pipeline" "PIPELINE_SEND=1
DISABLE_TRITON=0"

# 清理
stop_services
clear_config_on_hosts

# ---- 汇总结果 ----
echo ""
log_step "=========================================="
log_step "汇总结果"
log_step "=========================================="
echo ""

printf "%-8s %-30s %10s %10s %10s %10s %10s\n" \
    "配置" "描述" "req/s" "tok/s" "TTFT(ms)" "p99TTFT" "e2e(ms)"
echo "----------------------------------------------------------------------------"

CONFIG_SUMMARY=(
    "A1:baseline (无量化)"
    "A2:8bit PyTorch, 无 pipeline"
    "A3:8bit PyTorch, 有 pipeline"
    "A4:4bit PyTorch, 无 pipeline"
    "A5:4bit Triton, 无 pipeline"
    "A6:4bit Triton, 有 pipeline"
    "A7:mixed-C Triton, 有 pipeline"
)

for entry in "${CONFIG_SUMMARY[@]}"; do
    IFS=':' read -r config_id label <<< "${entry}"

    total_rps=0; total_tps=0; total_ttft=0; total_p99=0; total_e2e=0; count=0

    for run_idx in $(seq 1 "${NUM_RUNS}"); do
        json="${RESULTS_DIR}/${config_id}_run${run_idx}.json"
        if [ -f "$json" ] && [ -s "$json" ]; then
            rps=$(jq -r '.results.requests_per_sec // 0' "$json" 2>/dev/null)
            tps=$(jq -r '.results.tokens_per_sec // 0' "$json" 2>/dev/null)
            ttft=$(jq -r '.results.mean_ttft_ms // 0' "$json" 2>/dev/null)
            p99=$(jq -r '.results.p99_ttft_ms // 0' "$json" 2>/dev/null)
            e2e=$(jq -r '.results.mean_e2e_ms // 0' "$json" 2>/dev/null)
            total_rps=$(echo "${total_rps} + ${rps}" | bc)
            total_tps=$(echo "${total_tps} + ${tps}" | bc)
            total_ttft=$(echo "${total_ttft} + ${ttft}" | bc)
            total_p99=$(echo "${total_p99} + ${p99}" | bc)
            total_e2e=$(echo "${total_e2e} + ${e2e}" | bc)
            count=$((count + 1))
        fi
    done

    if [ "$count" -gt 0 ]; then
        avg_rps=$(echo "scale=2; ${total_rps} / ${count}" | bc)
        avg_tps=$(echo "scale=2; ${total_tps} / ${count}" | bc)
        avg_ttft=$(echo "scale=1; ${total_ttft} / ${count}" | bc)
        avg_p99=$(echo "scale=1; ${total_p99} / ${count}" | bc)
        avg_e2e=$(echo "scale=1; ${total_e2e} / ${count}" | bc)
        printf "%-8s %-30s %10s %10s %10s %10s %10s\n" \
            "${config_id}" "${label}" "${avg_rps}" "${avg_tps}" "${avg_ttft}" "${avg_p99}" "${avg_e2e}"
    else
        printf "%-8s %-30s %10s %10s %10s %10s %10s\n" \
            "${config_id}" "${label}" "N/A" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo ""
log_info "所有结果保存在: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/" 2>/dev/null

# ---- 生成 JSON 汇总 ----
RESULTS_DIR="${RESULTS_DIR}" NUM_RUNS="${NUM_RUNS}" python3 << 'PYTHON_SUMMARY'
import json, os, sys, math
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results/ablation_experiment"))
if not results_dir.exists():
    sys.exit(0)

configs = [
    ("A1", "baseline"),
    ("A2", "8bit_pytorch_no_pipe"),
    ("A3", "8bit_pytorch_pipe"),
    ("A4", "4bit_pytorch_no_pipe"),
    ("A5", "4bit_triton_no_pipe"),
    ("A6", "4bit_triton_pipe"),
    ("A7", "mixed_C_triton_pipe"),
]

num_runs = int(os.environ.get("NUM_RUNS", "3"))

summary = {
    "experiment": "ablation_v2",
    "design": "runtime_config_toggle_3_starts",
    "params": {"concurrency": 8, "num_requests": 32, "prompt_type": "medium", "num_runs": num_runs},
    "configs": []
}

for config_id, label in configs:
    runs = []
    for run_idx in range(1, num_runs + 1):
        json_file = results_dir / f"{config_id}_run{run_idx}.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            runs.append(data["results"])

    if runs:
        avg = {}
        for key in ["requests_per_sec", "tokens_per_sec", "output_tokens_per_sec",
                     "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "mean_e2e_ms"]:
            values = [r.get(key, 0) for r in runs]
            avg[key] = round(sum(values) / len(values), 2)
        avg["successful"] = min(r.get("successful", 0) for r in runs)
        avg["failed"] = max(r.get("failed", 0) for r in runs)
        avg["num_runs"] = len(runs)

        for key in ["requests_per_sec", "mean_ttft_ms"]:
            values = [r.get(key, 0) for r in runs]
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                avg[f"{key}_std"] = round(math.sqrt(variance), 2)

        summary["configs"].append({
            "id": config_id, "label": label, "results": avg, "runs": runs
        })

output_file = results_dir / "ablation_v2_summary.json"
with open(output_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nJSON 汇总已保存: {output_file}")
PYTHON_SUMMARY
