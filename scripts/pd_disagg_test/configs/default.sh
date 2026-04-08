# ============================================================================
# P/D Disaggregation 默认集群配置
# 可通过环境变量覆盖，例如: PREFILL_HOST=1.2.3.4 ./pd_test.sh full
# ============================================================================

# SSH 配置（对应 ~/.ssh/config 中的 Host 别名）
PREFILL_HOST="${PREFILL_HOST:-ubuntu@10.60.23.70}"
DECODE_HOST="${DECODE_HOST:-ubuntu@10.60.30.66}"

# 内网 IP（用于服务 --host 和 --dist-init-addr）
PREFILL_IP="${PREFILL_IP:-10.60.23.70}"
DECODE_IP="${DECODE_IP:-10.60.30.66}"

# 端口
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

# 模型
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"

# 远程机器路径
SGLANG_REPO="${SGLANG_REPO:-/home/ubuntu/sglang-kvtuner-splitwise}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/sglang-env}"

# 服务参数
TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"
DISABLE_OVERLAP="${DISABLE_OVERLAP:-false}"
LOG_LEVEL="${LOG_LEVEL:-warning}"

# KVTuner 量化参数
ENABLE_KVTUNER="${ENABLE_KVTUNER:-false}"
KVTUNER_LAYER_CONFIG="${KVTUNER_LAYER_CONFIG:-/home/ubuntu/qwen2.5-7b_layer_quant.json}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-false}"

# 超时（秒）
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"

# 日志路径（远程）
REMOTE_LOG_DIR="/tmp"
# 日志路径（本地拉取后存放位置）
LOCAL_LOG_DIR="./logs"

# 远程辅助脚本在远端的部署路径
REMOTE_WORKER_PATH="${SGLANG_REPO}/remote_worker.sh"
