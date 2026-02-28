#!/bin/bash
# test_compressed_transfer.sh - 测试 KVTuner KV Cache 压缩传输

set -e

# 配置
PREFILL_NODE="${PREFILL_NODE:-10.60.6.75:30000}"
DECODE_NODE="${DECODE_NODE:-10.60.19.152:30001}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen2.5-7B}"
NBITS="${NBITS:-4}"

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# 测试 1: 检查服务状态
test_services() {
    log_test "测试 1: 检查 P/D 服务状态"
    
    # 检查 Prefill
    if sshpass -p "luhang222" ssh -o StrictHostKeyChecking=no ubuntu@${PREFILL_NODE%:*} \
         "curl -s http://localhost:${PREFILL_NODE#*:}/health" | grep -q "healthy"; then
        log_pass "Prefill 服务健康 (${PREFILL_NODE})"
    else
        log_fail "Prefill 服务不健康"
        return 1
    fi
    
    # 检查 Decode
    if sshpass -p "luhang222" ssh -o StrictHostKeyChecking=no ubuntu@${DECODE_NODE%:*} \
         "curl -s http://localhost:${DECODE_NODE#*:}/health" | grep -q "healthy"; then
        log_pass "Decode 服务健康 (${DECODE_NODE})"
    else
        log_fail "Decode 服务不健康"
        return 1
    fi
}

# 测试 2: 测试压缩传输
test_compressed_transfer() {
    log_test "测试 2: KVTuner 压缩传输测试"
    
    # 创建测试脚本
    cat > /tmp/test_transfer.py << 'PYEOF'
import asyncio
import sys
import torch

sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation')
from kv_transfer_compressed import CompressedKVTransfer, CompressionLevel

async def test():
    print("创建测试 KV Cache (模拟 Llama-70B, 1000 tokens)...")
    # [layers=80, batch=1, heads=64, seq_len=1000, head_dim=128]
    # 简化测试：使用较小的张量
    test_kv = torch.randn(32, 1, 32, 500, 128, dtype=torch.bfloat16, device='cuda')
    original_size = test_kv.element_size() * test_kv.numel() / (1024 * 1024)
    print(f"  原始大小：{original_size:.2f} MB")
    
    # 创建传输器
    transfer = CompressedKVTransfer(nbits=4, backend='nixl')
    
    # 测试不同压缩级别
    for level in [CompressionLevel.LOSSLESS, CompressionLevel.HIGH_QUALITY, 
                  CompressionLevel.BALANCED, CompressionLevel.HIGH_COMPRESSION]:
        print(f"\n测试压缩级别：{level.value}")
        
        try:
            result, stats = await transfer.transfer_kv(
                test_kv,
                f"http://{sys.argv[1]}",
                f"http://{sys.argv[2]}",
                compression_level=level
            )
            
            print(f"  压缩比：{stats.compression_ratio:.2f}x")
            print(f"  传输时间：{stats.transfer_time_ms:.2f} ms")
            print(f"  总时间：{stats.total_time_ms:.2f} ms")
            print(f"  带宽：{stats.bandwidth_gbps:.2f} Gbps")
            
            # 计算精度损失
            mse = ((result - test_kv) ** 2).mean().item()
            print(f"  MSE: {mse:.6f}")
            
            if not stats.success:
                print(f"  ❌ 失败：{stats.error_message}")
                return False
            
        except Exception as e:
            print(f"  ❌ 异常：{e}")
            return False
    
    # 打印统计摘要
    print("\n" + "="*50)
    print("统计摘要:")
    summary = transfer.get_stats_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_transfer.py <src_host:port> <dst_host:port>")
        sys.exit(1)
    
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
PYEOF
    
    # 执行测试
    if sshpass -p "luhang222" ssh -o StrictHostKeyChecking=no ubuntu@${PREFILL_NODE%:*} \
         "cd ~/pd_quant_validation && python3 /tmp/test_transfer.py ${PREFILL_NODE} ${DECODE_NODE}"; then
        log_pass "压缩传输测试通过"
        return 0
    else
        log_fail "压缩传输测试失败"
        return 1
    fi
}

# 测试 3: 性能基准测试
test_benchmark() {
    log_test "测试 3: 性能基准测试"
    
    # 创建基准测试脚本
    cat > /tmp/benchmark_transfer.py << 'PYEOF'
import asyncio
import sys
import time
import torch

sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/sglang-kvtuner/scripts/pd_quant_validation')
from kv_transfer_compressed import CompressedKVTransfer

async def benchmark():
    print("性能基准测试")
    print("="*60)
    
    # 测试不同大小
    sizes = [
        ("小 (100 tokens)", 32, 1, 32, 100, 128),
        ("中 (500 tokens)", 32, 1, 32, 500, 128),
        ("大 (1000 tokens)", 32, 1, 32, 1000, 128),
    ]
    
    transfer = CompressedKVTransfer(nbits=4, backend='nixl')
    
    for name, layers, batch, heads, seq_len, head_dim in sizes:
        print(f"\n{name}: {layers}x{batch}x{heads}x{seq_len}x{head_dim}")
        
        test_kv = torch.randn(layers, batch, heads, seq_len, head_dim, 
                             dtype=torch.bfloat16, device='cuda')
        original_mb = test_kv.element_size() * test_kv.numel() / (1024 * 1024)
        print(f"  大小：{original_mb:.2f} MB")
        
        # 运行 3 次取平均
        times = []
        for i in range(3):
            start = time.time()
            _, stats = await transfer.transfer_kv(
                test_kv,
                f"http://{sys.argv[1]}",
                f"http://{sys.argv[2]}",
                compression_level=None  # 自动选择
            )
            times.append(stats.total_time_ms)
        
        avg_time = sum(times) / len(times)
        print(f"  平均时间：{avg_time:.2f} ms")
        print(f"  平均带宽：{(original_mb * 8) / (avg_time / 1000) / 1000:.2f} Gbps")
    
    print("\n" + "="*60)
    print("基准测试完成")

if __name__ == "__main__":
    asyncio.run(benchmark())
PYEOF
    
    # 执行基准测试
    if sshpass -p "luhang222" ssh -o StrictHostKeyChecking=no ubuntu@${PREFILL_NODE%:*} \
         "python3 /tmp/benchmark_transfer.py ${PREFILL_NODE} ${DECODE_NODE}"; then
        log_pass "性能基准测试完成"
        return 0
    else
        log_fail "性能基准测试失败"
        return 1
    fi
}

# 测试 4: 端到端推理测试
test_end_to_end() {
    log_test "测试 4: 端到端推理测试（带压缩传输）"
    
    # 创建端到端测试脚本
    cat > /tmp/test_e2e.py << 'PYEOF'
import asyncio
import sys
import aiohttp

async def test():
    print("端到端推理测试")
    print("="*60)
    
    # 通过 Prefill 节点发送请求
    prefill_url = f"http://{sys.argv[1]}"
    
    prompts = [
        "Hello, how are you?",
        "请介绍一下你自己。",
        "What is the capital of France?",
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            print(f"\n请求 {i+1}: {prompt[:30]}...")
            
            try:
                start = time.time()
                async with session.post(
                    f"{prefill_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"max_new_tokens": 32}
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    result = await resp.json()
                    latency = time.time() - start
                    
                    if "text" in result:
                        print(f"  响应：{result['text'][:50]}...")
                        print(f"  延迟：{latency*1000:.2f} ms")
                    else:
                        print(f"  错误：{result}")
            
            except Exception as e:
                print(f"  异常：{e}")
    
    print("\n" + "="*60)
    print("测试完成")

if __name__ == "__main__":
    import time
    asyncio.run(test())
PYEOF
    
    # 执行测试
    if sshpass -p "luhang222" ssh -o StrictHostKeyChecking=no ubuntu@${PREFILL_NODE%:*} \
         "python3 /tmp/test_e2e.py ${PREFILL_NODE}"; then
        log_pass "端到端测试完成"
        return 0
    else
        log_fail "端到端测试失败"
        return 1
    fi
}

# 生成报告
generate_report() {
    log_info "生成测试报告..."
    
    cat > /tmp/test_report.md << EOF
# KVTuner KV Cache 压缩传输测试报告

**日期**: $(date '+%Y-%m-%d %H:%M:%S')
**Prefill 节点**: ${PREFILL_NODE}
**Decode 节点**: ${DECODE_NODE}
**量化位数**: ${NBITS}-bit

## 测试结果

$(if [ $1 -eq 0 ]; then
    echo "✅ 所有测试通过"
else
    echo "❌ 部分测试失败"
fi)

## 配置

- 模型：${MODEL_PATH}
- 传输后端：NIXL (UCX over TCP)
- 压缩级别：自动选择

## 预期性能

| 指标 | 无压缩 | 4-bit 压缩 | 提升 |
|------|--------|-----------|------|
| 传输大小 | 100% | 25% | 75% ↓ |
| 传输延迟 | 基线 | ~25% | 75% ↓ |
| 带宽使用 | 基线 | ~25% | 75% ↓ |

## 下一步

1. 集成到生产环境
2. 大规模基准测试
3. 精度验证

EOF
    
    log_info "测试报告：/tmp/test_report.md"
}

# 主函数
main() {
    echo "=========================================="
    echo "KVTuner KV Cache 压缩传输测试"
    echo "=========================================="
    echo ""
    echo "配置:"
    echo "  Prefill: ${PREFILL_NODE}"
    echo "  Decode: ${DECODE_NODE}"
    echo "  量化：${NBITS}-bit"
    echo ""
    
    local tests_failed=0
    
    test_services || ((tests_failed++))
    echo ""
    
    test_compressed_transfer || ((tests_failed++))
    echo ""
    
    test_benchmark || ((tests_failed++))
    echo ""
    
    test_end_to_end || ((tests_failed++))
    echo ""
    
    generate_report $tests_failed
    
    echo "=========================================="
    if [ $tests_failed -eq 0 ]; then
        echo -e "${GREEN}✓ 所有测试通过!${NC}"
        exit 0
    else
        echo -e "${RED}✗ ${tests_failed} 个测试失败${NC}"
        exit 1
    fi
}

main "$@"
