#!/bin/bash
# deploy_to_nodes.sh - 部署 KVTuner P/D 分离验证环境到 4 台机器

set -e

# 机器配置
NODES=(
    "10.60.6.75"
    "10.60.19.152"
    "10.60.9.62"
    "10.60.176.217"
)

USERNAME="ubuntu"
PASSWORD="luhang222"
WORKSPACE="/home/ubuntu/.openclaw/workspace/sglang-kvtuner"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 sshpass
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        log_error "sshpass 未安装，请运行：sudo apt-get install -y sshpass"
        exit 1
    fi
}

# 测试 SSH 连接
test_ssh_connection() {
    local node=$1
    log_info "测试连接到 $node ..."
    
    if sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
         "$USERNAME@$node" "echo '连接成功'" 2>/dev/null; then
        log_info "✓ $node 连接成功"
        return 0
    else
        log_error "✗ $node 连接失败"
        return 1
    fi
}

# 部署到单个节点
deploy_to_node() {
    local node=$1
    local node_idx=$2
    
    log_info "正在部署到 $node (节点 $node_idx) ..."
    
    # 创建远程目录
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
        "mkdir -p $WORKSPACE/scripts/pd_quant_validation"
    
    # 复制验证脚本
    sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
        ./start_prefill.sh \
        ./start_decode.sh \
        ./start_router.sh \
        ./run_validation.sh \
        ./check_status.sh \
        "$USERNAME@$node:$WORKSPACE/scripts/pd_quant_validation/"
    
    # 设置执行权限
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
        "chmod +x $WORKSPACE/scripts/pd_quant_validation/*.sh"
    
    log_info "✓ $node 部署完成"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查 Python 环境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 uv
    if ! command -v uv &> /dev/null; then
        log_warn "uv 未安装，将使用 pip"
    fi
}

# 主函数
main() {
    log_info "=========================================="
    log_info "KVTuner P/D 分离验证环境部署"
    log_info "=========================================="
    
    check_dependencies
    check_sshpass
    
    log_info ""
    log_info "步骤 1: 测试所有节点连接..."
    local failed_nodes=()
    for i in "${!NODES[@]}"; do
        if ! test_ssh_connection "${NODES[$i]}"; then
            failed_nodes+=("${NODES[$i]}")
        fi
    done
    
    if [ ${#failed_nodes[@]} -ne 0 ]; then
        log_error "以下节点连接失败：${failed_nodes[*]}"
        log_warn "请检查网络连接和凭据"
        exit 1
    fi
    
    log_info ""
    log_info "步骤 2: 部署验证脚本..."
    for i in "${!NODES[@]}"; do
        deploy_to_node "${NODES[$i]}" $((i+1))
    done
    
    log_info ""
    log_info "=========================================="
    log_info "部署完成！"
    log_info "=========================================="
    log_info ""
    log_info "下一步:"
    log_info "1. 在 Node 1 & 3 运行：./start_prefill.sh"
    log_info "2. 在 Node 2 & 4 运行：./start_decode.sh"
    log_info "3. 在任一节点运行：./start_router.sh"
    log_info "4. 运行验证测试：./run_validation.sh"
    log_info ""
}

# 脚本目录
cd "$(dirname "$0")"

main "$@"
