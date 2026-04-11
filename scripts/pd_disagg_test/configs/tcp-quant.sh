# TCP + Transfer Quantization 配置
# 用法: CONFIG_FILE=configs/tcp-quant.sh ./pd_test.sh full

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/default.sh"

TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/sglang-env}"

# 启用传输量化 (8-bit)
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="${TRANSFER_QUANT_BITS:-8}"
