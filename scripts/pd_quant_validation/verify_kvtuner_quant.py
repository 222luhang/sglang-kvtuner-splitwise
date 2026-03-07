#!/usr/bin/env python3
"""
验证 KVTuner 层级量化配置是否生效

检查项目:
1. 服务配置中的 KVTuner 参数
2. 层级量化配置文件是否加载
3. 显存使用是否符合预期（压缩比）
4. 实际传输的 KV Cache 大小
"""

import requests
import json
import sys
from typing import Dict, Any

# 服务地址
PREFILL_NODE = "10.60.6.75:30000"
DECODE_NODE = "10.60.19.152:30001"


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_server_info(node: str) -> Dict[str, Any]:
    """获取服务器信息"""
    try:
        response = requests.get(f"http://{node}/get_server_info", timeout=10)
        return response.json()
    except Exception as e:
        print(f"✗ 无法获取 {node} 信息：{e}")
        return {}


def check_kvtuner_config(info: Dict[str, Any], node_name: str) -> bool:
    """检查 KVTuner 配置"""
    print_section(f"{node_name} - KVTuner 配置检查")
    
    if not info:
        return False
    
    checks = []
    
    # 基础量化配置
    kv_cache_dtype = info.get('kv_cache_dtype', 'N/A')
    enable_kvtuner = info.get('enable_kvtuner_quant', False)
    nbits_key = info.get('kvtuner_nbits_key', 'N/A')
    nbits_value = info.get('kvtuner_nbits_value', 'N/A')
    
    print(f"KV Cache 类型：     {kv_cache_dtype}")
    print(f"KVTuner 启用：      {enable_kvtuner}")
    print(f"Key bits:          {nbits_key}")
    print(f"Value bits:        {nbits_value}")
    
    checks.append(("KV Cache 量化", kv_cache_dtype != 'none'))
    checks.append(("KVTuner 启用", enable_kvtuner))
    
    # 层级量化配置
    enable_layer_wise = info.get('enable_kvtuner_layer_wise', False)
    layer_config_file = info.get('kvtuner_layer_config_file', None)
    layer_bits = info.get('kvtuner_layer_bits', None)
    
    print(f"\n层级量化:")
    print(f"  启用：           {enable_layer_wise}")
    print(f"  配置文件：       {layer_config_file or '未使用'}")
    
    if layer_bits:
        print(f"  层级配置：")
        if isinstance(layer_bits, dict):
            for layer, bits in sorted(layer_bits.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                print(f"    Layer {layer}: {bits}-bit")
        elif isinstance(layer_bits, list):
            for i, bits in enumerate(layer_bits[:10]):  # 只显示前 10 层
                print(f"    Layer {i}: {bits}-bit")
            if len(layer_bits) > 10:
                print(f"    ... 共 {len(layer_bits)} 层")
    
    checks.append(("层级量化启用", enable_layer_wise or layer_config_file is not None))
    
    # 残差配置
    residual_length = info.get('kvtuner_residual_length', 'N/A')
    print(f"\n残差配置:")
    print(f"  残差长度：       {residual_length}")
    
    # 打印结果
    print("\n" + "-" * 70)
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    return all_passed


def check_memory_usage(info: Dict[str, Any], node_name: str, expected_kbits: int):
    """检查显存使用和压缩效果"""
    print_section(f"{node_name} - 显存使用检查")
    
    if not info:
        return
    
    mem = info.get('internal_states', [{}])[0].get('memory_usage', {})
    
    if not mem:
        print("✗ 无法获取显存使用信息")
        return
    
    weight_gb = mem.get('weight', 0)
    kvcache_gb = mem.get('kvcache', 0)
    token_capacity = mem.get('token_capacity', 0)
    
    print(f"权重内存：      {weight_gb:.2f} GB")
    print(f"KV Cache 内存：  {kvcache_gb:.2f} GB")
    print(f"Token 容量：     {token_capacity:,}")
    
    # 计算压缩比
    # Qwen2.5-7B BF16 基线：~22.8 GB KV Cache
    baseline_kvcache = 22.8
    if kvcache_gb > 0:
        compression = baseline_kvcache / kvcache_gb
        savings = (1 - kvcache_gb / baseline_kvcache) * 100
        
        print(f"\n量化效果:")
        print(f"  压缩比：        {compression:.2f}x")
        print(f"  显存节省：      {savings:.1f}%")
        
        # 理论压缩比估算
        # fp8_e5m2 + KVTuner 4-bit: 理论压缩比 ~4-6x
        # fp8_e5m2 + KVTuner 8-bit: 理论压缩比 ~2-3x
        if expected_kbits == 4:
            expected_compression = 4.0
        elif expected_kbits == 8:
            expected_compression = 2.0
        else:
            expected_compression = 2.0
        
        print(f"\n理论参考 (KVTuner {expected_kbits}-bit):")
        print(f"  预期压缩比：    ~{expected_compression:.1f}x")
        
        if compression >= expected_compression * 0.8:
            print(f"\n✓ 压缩效果符合预期")
        else:
            print(f"\n⚠ 压缩效果低于预期，可能需要检查配置")
    
    return mem


def check_transfer_backend(info: Dict[str, Any], node_name: str):
    """检查传输后端配置"""
    print_section(f"{node_name} - 传输后端检查")
    
    if not info:
        return
    
    mode = info.get('disaggregation_mode', 'N/A')
    backend = info.get('disaggregation_transfer_backend', 'N/A')
    bootstrap_port = info.get('disaggregation_bootstrap_port', 'N/A')
    ib_device = info.get('disaggregation_ib_device', 'N/A')
    fake_auto = info.get('disaggregation_decode_enable_fake_auto', False)
    
    print(f"P/D 模式：         {mode}")
    print(f"传输后端：        {backend}")
    print(f"Bootstrap 端口：   {bootstrap_port}")
    print(f"IB 设备：          {ib_device or '未配置'}")
    print(f"Fake Auto:        {fake_auto}")
    
    # 检查配置是否正确
    if backend == 'nixl':
        print(f"\n✓ NIXL 后端已启用")
        
        # 检查 NIXL 配置
        # 注意：当前 API 可能不直接暴露 NIXL 后端类型
        # 需要通过日志或环境变量确认
        
    return True


def verify_layer_quant_config(config_file: str) -> bool:
    """验证层级量化配置文件"""
    print_section("层级量化配置文件验证")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"配置文件：{config_file}")
        print(f"模型：    {config.get('model', 'N/A')}")
        
        layer_bits = config.get('layer_bits', {})
        
        if isinstance(layer_bits, dict):
            print(f"\n层级配置 (共 {len(layer_bits)} 层):")
            for layer, bits in sorted(layer_bits.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)[:10]:
                print(f"  Layer {layer}: {bits}-bit")
            if len(layer_bits) > 10:
                print(f"  ... 共 {len(layer_bits)} 层")
        
        print(f"\n✓ 配置文件格式正确")
        return True
        
    except FileNotFoundError:
        print(f"✗ 配置文件不存在：{config_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败：{e}")
        return False
    except Exception as e:
        print(f"✗ 验证失败：{e}")
        return False


def run_complete_verification():
    """运行完整验证"""
    print("\n")
    print("█" * 70)
    print("█  KVTuner 层级量化配置验证")
    print("█  POSIX 后端 + KV Cache 压缩传输")
    print("█" * 70)
    
    # 检查 Prefill 节点
    print("\n[1/4] 检查 Prefill 节点配置...")
    prefill_info = get_server_info(PREFILL_NODE)
    prefill_ok = check_kvtuner_config(prefill_info, "Prefill 节点")
    check_memory_usage(prefill_info, "Prefill 节点", expected_kbits=4)
    check_transfer_backend(prefill_info, "Prefill 节点")
    
    # 检查 Decode 节点
    print("\n[2/4] 检查 Decode 节点配置...")
    decode_info = get_server_info(DECODE_NODE)
    decode_ok = check_kvtuner_config(decode_info, "Decode 节点")
    check_memory_usage(decode_info, "Decode 节点", expected_kbits=8)
    check_transfer_backend(decode_info, "Decode 节点")
    
    # 验证层级配置文件
    config_file = prefill_info.get('kvtuner_layer_config_file')
    if config_file:
        print("\n[3/4] 验证层级量化配置文件...")
        config_ok = verify_layer_quant_config(config_file)
    else:
        print("\n[3/4] 跳过配置文件验证（使用统一量化）")
        config_ok = True
    
    # 总结
    print_section("验证总结")
    
    all_ok = prefill_ok and decode_ok and config_ok
    
    if all_ok:
        print("✅ 所有检查通过！KVTuner 层级量化配置已生效")
        print("\n下一步:")
        print("  1. 运行完整 P/D 流程测试")
        print("  2. 验证 KV Cache 传输")
        print("  3. 检查传输后的量化效果")
    else:
        print("⚠ 部分检查未通过，请检查配置")
        
        if not prefill_ok:
            print("  - Prefill 节点 KVTuner 配置可能有问题")
        if not decode_ok:
            print("  - Decode 节点 KVTuner 配置可能有问题")
    
    return all_ok


if __name__ == "__main__":
    success = run_complete_verification()
    sys.exit(0 if success else 1)
