# TCP backend 配置
# 基于 default.sh，覆盖 backend 和 venv 路径

source "$(dirname "$0")/default.sh"

TRANSFER_BACKEND="${TRANSFER_BACKEND:-tcp}"
VENV_DIR="${VENV_DIR:-/home/ubuntu/sglang-env}"
