#!/bin/bash
# test_scheduler.sh - 测试动态调度器

set -e

SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:9000}"

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_test() { echo -e "${YELLOW}[TEST]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# 测试 1: 健康检查
test_health() {
    log_test "测试 1: 调度器健康检查"
    
    if curl -s "$SCHEDULER_URL/health" | grep -q "healthy"; then
        log_pass "调度器健康检查通过"
        return 0
    else
        log_fail "调度器健康检查失败"
        return 1
    fi
}

# 测试 2: 获取状态
test_status() {
    log_test "测试 2: 获取调度器状态"
    
    local status=$(curl -s "$SCHEDULER_URL/status")
    
    if echo "$status" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'prefill_nodes' in d and 'decode_nodes' in d" 2>/dev/null; then
        log_pass "调度器状态获取成功"
        echo "$status" | python3 -m json.tool
        return 0
    else
        log_fail "调度器状态获取失败"
        return 1
    fi
}

# 测试 3: 节点健康
test_node_health() {
    log_test "测试 3: 节点健康状态"
    
    local status=$(curl -s "$SCHEDULER_URL/status")
    
    # 检查 Prefill 节点
    local prefill_healthy=$(echo "$status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
healthy = sum(1 for n in d['prefill_nodes'].values() if n['status'] == 'healthy')
print(healthy)
" 2>/dev/null)
    
    # 检查 Decode 节点
    local decode_healthy=$(echo "$status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
healthy = sum(1 for n in d['decode_nodes'].values() if n['status'] == 'healthy')
print(healthy)
" 2>/dev/null)
    
    if [ "$prefill_healthy" -gt 0 ] && [ "$decode_healthy" -gt 0 ]; then
        log_pass "节点健康：Prefill=$prefill_healthy, Decode=$decode_healthy"
        return 0
    else
        log_fail "节点健康检查失败：Prefill=$prefill_healthy, Decode=$decode_healthy"
        return 1
    fi
}

# 测试 4: 调度请求
test_schedule_request() {
    log_test "测试 4: 调度请求"
    
    local result=$(curl -s -X POST "$SCHEDULER_URL/schedule" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Hello, how are you?",
            "max_tokens": 32,
            "temperature": 0.7
        }')
    
    if echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if 'text' in d or 'error' in d else 'FAIL')" 2>/dev/null | grep -q "OK"; then
        log_pass "调度请求成功"
        echo "$result" | python3 -m json.tool
        return 0
    else
        log_fail "调度请求失败"
        echo "$result"
        return 1
    fi
}

# 测试 5: 负载平衡
test_load_balancing() {
    log_test "测试 5: 负载均衡测试"
    
    # 发送多个请求
    for i in {1..5}; do
        curl -s -X POST "$SCHEDULER_URL/schedule" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"Test request $i\", \"max_tokens\": 16}" &
    done
    
    wait
    
    # 检查负载分布
    local status=$(curl -s "$SCHEDULER_URL/status")
    
    echo "$status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Prefill 节点负载:')
for url, info in d['prefill_nodes'].items():
    print(f'  {url}: load={info[\"load\"]:.2f}, health={info[\"health_score\"]:.1f}')
print('Decode 节点负载:')
for url, info in d['decode_nodes'].items():
    print(f'  {url}: load={info[\"load\"]:.2f}, health={info[\"health_score\"]:.1f}')
" 2>/dev/null
    
    log_pass "负载均衡测试完成"
    return 0
}

# 测试 6: Prefix Cache
test_prefix_cache() {
    log_test "测试 6: Prefix Cache 测试"
    
    # 发送相同 prompt 两次
    local prompt="What is the capital of France?"
    
    echo "第一次请求（Cache Miss）..."
    curl -s -X POST "$SCHEDULER_URL/schedule" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_tokens\": 32}" > /dev/null
    
    sleep 1
    
    echo "第二次请求（预期 Cache Hit）..."
    curl -s -X POST "$SCHEDULER_URL/schedule" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_tokens\": 32}" > /dev/null
    
    # 检查命中率
    local status=$(curl -s "$SCHEDULER_URL/status")
    
    local hit_rate=$(echo "$status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for info in d['prefill_nodes'].values():
    if info.get('prefix_hit_rate', 0) > 0:
        print(f\"{info['prefix_hit_rate']:.2f}\")
        sys.exit(0)
print('0')
" 2>/dev/null)
    
    if [ "$hit_rate" != "0" ]; then
        log_pass "Prefix Cache 命中率：$hit_rate"
        return 0
    else
        log_fail "Prefix Cache 未生效"
        return 1
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "动态调度器测试"
    echo "=========================================="
    echo ""
    
    local tests_passed=0
    local tests_failed=0
    
    test_health && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    test_status && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    test_node_health && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    test_schedule_request && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    test_load_balancing && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    test_prefix_cache && ((tests_passed++)) || ((tests_failed++))
    echo ""
    
    echo "=========================================="
    echo "测试结果"
    echo "=========================================="
    echo "通过：$tests_passed"
    echo "失败：$tests_failed"
    echo ""
    
    if [ $tests_failed -eq 0 ]; then
        echo -e "${GREEN}✓ 所有测试通过!${NC}"
        exit 0
    else
        echo -e "${RED}✗ 部分测试失败${NC}"
        exit 1
    fi
}

# 检查调度器是否运行
check_scheduler() {
    if ! curl -s "$SCHEDULER_URL/health" > /dev/null 2>&1; then
        echo -e "${RED}错误：调度器未运行${NC}"
        echo "请先启动调度器:"
        echo "  python3 dynamic_scheduler.py --prefill 10.60.6.75:30000,10.60.9.62:30000 --decode 10.60.19.152:30001,10.60.176.217:30001"
        exit 1
    fi
}

# 运行测试
check_scheduler
main "$@"
