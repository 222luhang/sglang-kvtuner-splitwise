# TCP + Transfer Quantization Mixed (逐层混合精度) 配置
# 用法: CONFIG_FILE=configs/tcp-quant-mixed.sh ./pd_test.sh full
#
# 可通过 MIXED_STRATEGY 选择策略:
#   mixed-A: 前/后 5 层 8-bit + 中间 4-bit
#   mixed-B: 敏感度 Top-5 层 8-bit + 其余 4-bit (默认)
#   mixed-C: 敏感度 Top-10 层 8-bit + 其余 4-bit
#   mixed-D: Top-5 层 8-bit + 中间 4-bit + 底部 2-bit
#
# 用法:
#   MIXED_STRATEGY=mixed-A ./pd_test.sh full
#   MIXED_STRATEGY=mixed-B ./pd_test.sh full

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/default.sh"

TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"

# 启用传输量化，全局默认 4-bit（被逐层配置覆盖）
ENABLE_TRANSFER_QUANT="true"
TRANSFER_QUANT_BITS="4"

# 逐层混合精度配置
# 选择策略
MIXED_STRATEGY="${MIXED_STRATEGY:-mixed-B}"

# 加载对应的逐层配置
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/quant"
KVTUNER_LAYER_BITS="$(cat "${CONFIG_DIR}/${MIXED_STRATEGY}.json")"
