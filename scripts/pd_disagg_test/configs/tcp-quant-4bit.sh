# TCP + Transfer Quantization 4-bit 配置
# 用法: CONFIG_FILE=configs/tcp-quant-4bit.sh ./pd_test.sh full

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/default.sh"

TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"

# 启用传输量化 (4-bit)
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="4"
