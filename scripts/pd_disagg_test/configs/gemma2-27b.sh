# Gemma2-27B 模型配置
# 用法: CONFIG_FILE=configs/gemma2-27b.sh ./pd_test.sh full

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/default.sh"

# Gemma2-27B 模型路径
MODEL_PATH="/data/Gemma/gemma2-27B"

# 启用 overlap schedule (DISABLE_OVERLAP=false)
DISABLE_OVERLAP="false"

# 传输量化配置 (可通过环境变量覆盖)
# ENABLE_TRANSFER_QUANT="${ENABLE_TRANSFER_QUANT:-false}"
# TRANSFER_QUANT_BITS="${TRANSFER_QUANT_BITS:-8}"
