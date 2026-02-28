#!/bin/bash
# deploy_and_run.sh - 一键部署并启动 P/D 分离验证环境

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
REMOTE_DIR="/home/ubuntu/pd_quant_validation"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 测试所有节点连接
test_connections() {
    log_step "测试所有节点连接..."
    for node in "${NODES[@]}"; do
        if sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
             "$USERNAME@$node" "echo 'OK'" >/dev/null 2>&1; then
            log_info "✓ $node 连接成功"
        else
            log_error "✗ $node 连接失败"
            return 1
        fi
    done
    log_info "所有节点连接正常"
}

# 部署脚本到节点
deploy_scripts() {
    log_step "部署脚本到所有节点..."
    
    for node in "${NODES[@]}"; do
        log_info "部署到 $node ..."
        
        # 创建远程目录
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "mkdir -p $REMOTE_DIR"
        
        # 复制脚本
        sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
            "$SCRIPT_DIR/start_prefill.sh" \
            "$SCRIPT_DIR/start_decode.sh" \
            "$SCRIPT_DIR/start_router.sh" \
            "$SCRIPT_DIR/check_status.sh" \
            "$SCRIPT_DIR/run_validation.sh" \
            "$SCRIPT_DIR/validate_pd_quant.py" \
            "$USERNAME@$node:$REMOTE_DIR/"
        
        # 设置执行权限
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/*.py"
        
        log_info "✓ $node 脚本部署完成"
    done
}

# 同步 sglang-kvtuner 代码
sync_code() {
    log_step "同步 sglang-kvtuner 代码到所有节点..."
    
    local local_code="/home/ubuntu/.openclaw/workspace/sglang-kvtuner"
    local remote_code="/home/ubuntu/sglang-kvtuner"
    
    for node in "${NODES[@]}"; do
        log_info "同步代码到 $node ..."
        
        # 使用 rsync 或 scp 同步代码
        if command -v rsync &> /dev/null; then
            rsync -avz -e "sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=no" \
                "$local_code/" "$USERNAME@$node:$remote_code/" 2>/dev/null || \
            sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
                "$local_code/" "$USERNAME@$node:$remote_code/"
        else
            sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
                "$local_code/" "$USERNAME@$node:$remote_code/"
        fi
        
        log_info "✓ $node 代码同步完成"
    done
    
    # 更新脚本中的路径
    for node in "${NODES[@]}"; do
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "sed -i 's|/home/ubuntu/.openclaw/workspace/sglang-kvtuner|/home/ubuntu/sglang-kvtuner|g' $REMOTE_DIR/*.sh 2>/dev/null || true"
    done
}

# 安装依赖
install_deps() {
    log_step "检查并安装依赖..."
    
    for node in "${NODES[@]}"; do
        log_info "检查 $node 依赖..."
        
        # 检查 sglang 是否已安装
        local has_sglang=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "~/.venv/bin/python3 -c 'import sglang; print(\"ok\")' 2>/dev/null || echo "not_found")
        
        if [ "$has_sglang" != "ok" ]; then
            log_info "  安装 sglang 到 $node ..."
            
            # 在后台安装
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
                "cd /home/ubuntu/sglang-kvtuner && ~/.venv/bin/pip install -e . --quiet" &
        else
            log_info "  ✓ sglang 已安装"
        fi
    done
    
    # 等待安装完成
    log_info "等待依赖安装 (最多 120 秒)..."
    sleep 120
}

# 在节点上启动服务
start_services() {
    log_step "启动服务..."
    
    # Node 1 & 3: Prefill
    for node in "10.60.6.75" "10.60.9.62"; do
        log_info "在 $node 启动 Prefill 服务..."
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "cd $REMOTE_DIR && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_prefill.sh" &
    done
    
    # Node 2 & 4: Decode
    for node in "10.60.19.152" "10.60.176.217"; do
        log_info "在 $node 启动 Decode 服务..."
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$node" \
            "cd $REMOTE_DIR && MODEL_PATH=/data/Qwen/Qwen2.5-7B TP_SIZE=2 ./start_decode.sh" &
    done
    
    # 等待服务启动
    log_info "等待服务启动 (60 秒)..."
    sleep 60
    
    # 启动 Router (在 Node 1)
    log_info "在 10.60.6.75 启动 Router..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ubuntu@10.60.6.75 \
        "cd $REMOTE_DIR && PREFILL_NODES='10.60.6.75:30000,10.60.9.62:30000' DECODE_NODES='10.60.19.152:30001,10.60.176.217:30001' ./start_router.sh" &
    
    # 等待 Router 启动
    sleep 30
}

# 运行验证
run_validation() {
    log_step "运行验证测试..."
    
    # 在 Node 1 运行验证
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ubuntu@10.60.6.75 \
        "cd $REMOTE_DIR && python3 validate_pd_quant.py --output validation_report.json"
}

# 主函数
main() {
    echo "=========================================="
    echo "P/D 分离与量化验证 - 一键部署"
    echo "=========================================="
    echo ""
    echo "配置:"
    echo "  模型：/data/Qwen/Qwen2.5-7B"
    echo "  GPU: 2×RTX 3090 per node"
    echo "  TP 大小：2"
    echo "  传输后端：nixl (TCP)"
    echo "  量化：KVTuner (Prefill K4/V4, Decode K8/V8)"
    echo ""
    
    test_connections
    echo ""
    
    sync_code
    echo ""
    
    install_deps
    echo ""
    
    deploy_scripts
    echo ""
    
    start_services
    echo ""
    
    run_validation
    echo ""
    
    log_info "=========================================="
    log_info "部署和验证完成!"
    log_info "=========================================="
    log_info ""
    log_info "查看验证报告:"
    log_info "  ssh ubuntu@10.60.6.75 'cat $REMOTE_DIR/validation_report.json'"
    log_info ""
    log_info "检查服务状态:"
    log_info "  ssh ubuntu@10.60.6.75 'cd $REMOTE_DIR && ./check_status.sh'"
    log_info ""
}

main "$@"
