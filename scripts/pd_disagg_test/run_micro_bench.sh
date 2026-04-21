#!/bin/bash
# ============================================================================
# KV Cache Transfer Micro-Benchmark
# Measures quantize/dequantize/transfer timing at different sequence lengths
#
# Usage: bash run_micro_bench.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/micro_benchmark"

mkdir -p "${RESULTS_DIR}"

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step()  { echo -e "\033[0;34m[STEP]\033[0m $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# Source config
source "${SCRIPT_DIR}/configs/default.sh"

log_step "KV Cache Transfer Micro-Benchmark"
log_step "Testing sequence lengths: 16, 32, 64, 128, 256, 512, 1024, 2048"
echo ""

# Deploy benchmark script to gpu1
log_info "Deploying benchmark script to gpu1..."
scp "${PROJECT_ROOT}/eval/bench_quant_micro.py" gpu1:"${SGLANG_REPO}/eval/"

# Run benchmark on gpu1 (has GPU)
log_info "Running micro-benchmark on gpu1..."
ssh gpu1 "cd ${SGLANG_REPO} && \
    source ${VENV_DIR}/bin/activate && \
    python3 eval/bench_quant_micro.py \
        --output /tmp/quant_micro_bench.json" 2>&1 | tee "${RESULTS_DIR}/benchmark.log"

# Pull results
log_info "Pulling results from gpu1..."
scp gpu1:"/tmp/quant_micro_bench.json" "${RESULTS_DIR}/results.json"

log_info "Results saved to ${RESULTS_DIR}/results.json"

# Generate plots
log_step "Generating visualization plots..."
python3 << 'PYTHON'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("${RESULTS_DIR}")
with open(results_dir / "results.json") as f:
    data = json.load(f)

results = data["results"]

# Extract data
seq_lens = [r["seq_len"] for r in results]
quant_8bit = [r["quant_8bit_mean"] for r in results]
quant_4bit = [r["quant_4bit_mean"] for r in results]
dequant_8bit = [r["dequant_8bit_mean"] for r in results]
dequant_4bit = [r["dequant_4bit_mean"] for r in results]
transfer_1gbps = [r["transfer_1gbps_baseline"] for r in results]
transfer_10gbps = [r["transfer_10gbps_baseline"] for r in results]

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Quantization timing
ax1 = axes[0, 0]
ax1.plot(seq_lens, quant_8bit, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='8-bit')
ax1.plot(seq_lens, quant_4bit, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='4-bit')
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Quantization Timing')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Plot 2: Dequantization timing
ax2 = axes[0, 1]
ax2.plot(seq_lens, dequant_8bit, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='8-bit')
ax2.plot(seq_lens, dequant_4bit, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='4-bit')
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Dequantization Timing')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Plot 3: Total processing time (quant + dequant)
ax3 = axes[1, 0]
total_8bit = [q + d for q, d in zip(quant_8bit, dequant_8bit)]
total_4bit = [q + d for q, d in zip(quant_4bit, dequant_4bit)]
ax3.plot(seq_lens, total_8bit, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='8-bit total')
ax3.plot(seq_lens, total_4bit, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='4-bit total')
ax3.set_xlabel('Sequence Length')
ax3.set_ylabel('Time (ms)')
ax3.set_title('Total Quant+Dequant Timing')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)

# Plot 4: Processing time vs Transfer time
ax4 = axes[1, 1]
ax4.plot(seq_lens, total_8bit, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Quant+Dequant (8-bit)')
ax4.plot(seq_lens, total_4bit, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='Quant+Dequant (4-bit)')
ax4.plot(seq_lens, transfer_1gbps, '^--', color='#2ca02c', linewidth=2, markersize=8, label='Transfer @1Gbps')
ax4.plot(seq_lens, transfer_10gbps, 'v--', color='#d62728', linewidth=2, markersize=8, label='Transfer @10Gbps')
ax4.set_xlabel('Sequence Length')
ax4.set_ylabel('Time (ms)')
ax4.set_title('Processing vs Transfer Time')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log', base=2)

plt.suptitle('KV Cache Transfer Micro-Benchmark Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(results_dir / 'micro_benchmark.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to {results_dir}/micro_benchmark.png")

# Print summary
print("\n" + "=" * 70)
print("Summary: Processing Time vs Transfer Time")
print("=" * 70)
print(f"{'SeqLen':>8} {'Quant8':>10} {'Dequant8':>10} {'Total8':>10} {'Xfer@1G':>10} {'Xfer@10G':>10}")
print("-" * 70)
for i, s in enumerate(seq_lens):
    print(f"{s:>8} {quant_8bit[i]:>10.3f} {dequant_8bit[i]:>10.3f} {total_8bit[i]:>10.3f} "
          f"{transfer_1gbps[i]:>10.3f} {transfer_10gbps[i]:>10.3f}")
PYTHON

log_step "Micro-benchmark complete!"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
