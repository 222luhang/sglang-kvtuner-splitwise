#!/bin/bash
# check_status.sh - 检查 P/D 分离服务状态

set -e

# 配置
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_service() {
    local name=$1
    local port=$2
    local url="http://localhost:$port/health"
    
    printf "%-20s " "$name (:$port)"
    
    if curl -s --connect-timeout 2 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}● 运行中${NC}"
        
        # 获取详细信息
        local info=$(curl -s "http://localhost:$port/get_server_info" 2>/dev/null)
        if [ -n "$info" ]; then
            echo "$info" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f\"  模型：{d.get('model_path', 'N/A')[:50]}...\")
    print(f\"  模式：{d.get('disaggregation_mode', 'N/A')}\")
    print(f\"  KV 量化：{d.get('kv_cache_dtype', 'N/A')}\")
except:
    pass
" 2>/dev/null || true
        fi
        return 0
    else
        echo -e "${RED}○ 未运行${NC}"
        return 1
    fi
}

check_processes() {
    echo ""
    echo "进程状态:"
    echo "----------------------------------------"
    
    local sglang_procs=$(ps aux | grep -E "sglang.*launch_server|sglang_router" | grep -v grep || true)
    
    if [ -n "$sglang_procs" ]; then
        echo "$sglang_procs" | awk '{print $2, $11, $12, $13}' | while read pid cmd arg1 arg2; do
            printf "  PID %-8s %s %s %s\n" "$pid" "$cmd" "$arg1" "$arg2"
        done
    else
        echo "  未找到 SGLang 相关进程"
    fi
}

check_ports() {
    echo ""
    echo "端口监听状态:"
    echo "----------------------------------------"
    
    for port in $PREFILL_PORT $DECODE_PORT $ROUTER_PORT; do
        if ss -tlnp 2>/dev/null | grep -q ":$port "; then
            local proc=$(ss -tlnp 2>/dev/null | grep ":$port " | awk -F'"' '{print $2}' | head -1)
            echo -e "  ${GREEN}●${NC} :$port - $proc"
        else
            echo -e "  ${RED}○${NC} :$port - 未监听"
        fi
    done
}

check_logs() {
    echo ""
    echo "最近日志:"
    echo "----------------------------------------"
    
    local log_dir="/tmp/sglang_logs"
    if [ -d "$log_dir" ]; then
        local latest_log=$(ls -t "$log_dir"/*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "最新日志：$latest_log"
            echo ""
            tail -20 "$latest_log" 2>/dev/null | sed 's/^/  /'
        else
            echo "  未找到日志文件"
        fi
    else
        echo "  日志目录不存在：$log_dir"
    fi
}

main() {
    echo "=========================================="
    echo "P/D 分离服务状态检查"
    echo "=========================================="
    echo "时间：$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    echo "服务状态:"
    echo "----------------------------------------"
    check_service "Prefill" $PREFILL_PORT
    check_service "Decode" $DECODE_PORT
    check_service "Router" $ROUTER_PORT
    
    check_processes
    check_ports
    check_logs
    
    echo ""
    echo "=========================================="
}

main "$@"
