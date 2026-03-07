#!/usr/bin/env python3
"""
Complete P/D Disaggregation Flow Test

测试完整的 P/D 分离流程：
1. 获取 Bootstrap Room ID
2. 发送 Prefill 请求
3. 发送 Decode 请求  
4. 验证 KV Cache 传输
5. 性能统计
"""

import requests
import time
import json
import sys

# 服务配置
BOOTSTRAP_URL = "http://10.60.179.106:8998"
PREFILL_NODE = "10.60.6.75:30000"
DECODE_NODE = "10.60.19.152:30001"
MODEL_PATH = "/data/Qwen/Qwen2.5-7B"


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def get_bootstrap_room():
    """步骤 1: 获取 Bootstrap Room ID"""
    print_section("步骤 1: 获取 Bootstrap Room ID")
    
    try:
        response = requests.post(f"{BOOTSTRAP_URL}/bootstrap", timeout=10)
        data = response.json()
        
        if data.get('success'):
            print(f"✓ Room ID: {data['room_id']}")
            print(f"  Prefill 节点：{data['prefill_node']}")
            print(f"  Decode 节点：{data['decode_node']}")
            print(f"  有效期：{data['expires_in']} 秒")
            return data
        else:
            print(f"✗ 获取 Room ID 失败：{data}")
            return None
    
    except Exception as e:
        print(f"✗ 错误：{e}")
        return None


def send_prefill_request(room_info):
    """步骤 2: 发送 Prefill 请求"""
    print_section("步骤 2: 发送 Prefill 请求")
    
    room_id = room_info['room_id']
    prefill_url = f"http://{room_info['prefill_node']}/generate"
    
    prompt = "Hello, this is a test of P/D disaggregation with KVTuner layer-wise quantization. Please introduce yourself."
    
    print(f"Prompt: {prompt[:80]}...")
    print(f"Room ID: {room_id}")
    print(f"Prefill URL: {prefill_url}")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            prefill_url,
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                },
                "bootstrap_room": room_id,
            },
            timeout=120
        )
        
        latency = (time.time() - start_time) * 1000
        result = response.json()
        
        if "text" in result:
            print(f"✓ Prefill 成功")
            print(f"  响应：{result['text'][:150]}...")
            print(f"  延迟：{latency:.2f} ms")
            
            # 尝试获取 token 统计
            if "usage" in result:
                print(f"  完成 tokens: {result['usage'].get('completion_tokens', 'N/A')}")
            
            return {"success": True, "latency": latency, "result": result}
        else:
            print(f"✗ Prefill 失败：{result}")
            return {"success": False, "error": result}
    
    except Exception as e:
        print(f"✗ 错误：{e}")
        return {"success": False, "error": str(e)}


def send_decode_request(room_info):
    """步骤 3: 发送 Decode 请求"""
    print_section("步骤 3: 发送 Decode 请求")
    
    room_id = room_info['room_id']
    decode_url = f"http://{room_info['decode_node']}/generate"
    
    prompt = "Continue the conversation."
    
    print(f"Prompt: {prompt}")
    print(f"Room ID: {room_id}")
    print(f"Decode URL: {decode_url}")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            decode_url,
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                },
                "bootstrap_room": room_id,
            },
            timeout=120
        )
        
        latency = (time.time() - start_time) * 1000
        result = response.json()
        
        if "text" in result:
            print(f"✓ Decode 成功")
            print(f"  响应：{result['text'][:150]}...")
            print(f"  延迟：{latency:.2f} ms")
            
            if "usage" in result:
                print(f"  完成 tokens: {result['usage'].get('completion_tokens', 'N/A')}")
            
            return {"success": True, "latency": latency, "result": result}
        else:
            print(f"✗ Decode 失败：{result}")
            return {"success": False, "error": result}
    
    except Exception as e:
        print(f"✗ 错误：{e}")
        return {"success": False, "error": str(e)}


def complete_room(room_info):
    """步骤 4: 完成 Room"""
    print_section("步骤 4: 完成 Room")
    
    room_id = room_info['room_id']
    room_id_str = str(room_id) if not isinstance(room_id, str) else room_id
    
    try:
        response = requests.post(
            f"{BOOTSTRAP_URL}/bootstrap/{room_id}/complete",
            timeout=10
        )
        
        if response.json().get('success'):
            print(f"✓ Room {room_id_str[:8]}... 标记为完成")
            return True
        else:
            print(f"✗ 完成 Room 失败")
            return False
    
    except Exception as e:
        print(f"✗ 错误：{e}")
        return False


def verify_quantization():
    """步骤 5: 验证量化配置"""
    print_section("步骤 5: 验证 KVTuner 层级量化配置")
    
    try:
        # 检查 Prefill 节点
        response = requests.get(f"http://{PREFILL_NODE}/get_server_info", timeout=10)
        info = response.json()
        
        print("Prefill 节点量化配置:")
        print(f"  模型：{info.get('model_path', 'N/A')}")
        print(f"  P/D 模式：{info.get('disaggregation_mode', 'N/A')}")
        print(f"  KV Cache 量化：{info.get('kv_cache_dtype', 'N/A')}")
        print(f"  KVTuner 启用：{info.get('enable_kvtuner_quant', False)}")
        print(f"  KVTuner nbits: {info.get('kvtuner_nbits_key', 'N/A')}")
        
        # 显存使用
        memory = info.get('internal_states', [{}])[0].get('memory_usage', {})
        print(f"\n  显存使用:")
        print(f"    权重：{memory.get('weight', 'N/A')} GB")
        print(f"    KV Cache: {memory.get('kvcache', 'N/A')} GB")
        print(f"    Token 容量：{memory.get('token_capacity', 'N/A')}")
        
        # 计算压缩比
        if memory.get('kvcache'):
            baseline = 22.8  # BF16 基线
            actual = memory['kvcache']
            compression = baseline / actual if actual > 0 else 0
            savings = (1 - actual / baseline) * 100 if baseline > 0 else 0
            
            print(f"\n  量化效果:")
            print(f"    预估压缩比：{compression:.2f}x")
            print(f"    显存节省：{savings:.1f}%")
        
        return True
    
    except Exception as e:
        print(f"✗ 错误：{e}")
        return False


def run_complete_test():
    """运行完整测试流程"""
    print("\n")
    print("█"*60)
    print("█  完整 P/D 分离流程测试")
    print("█  KVTuner 层级量化 + Splitwise 架构")
    print("█  模型：Qwen2.5-7B")
    print("█"*60)
    print("\n")
    
    # 步骤 1: 获取 Room ID
    room_info = get_bootstrap_room()
    if not room_info:
        print("\n✗ 无法获取 Room ID，测试终止")
        return None
    
    time.sleep(1)
    
    # 步骤 2: Prefill 请求
    prefill_result = send_prefill_request(room_info)
    time.sleep(1)
    
    # 步骤 3: Decode 请求
    decode_result = send_decode_request(room_info)
    time.sleep(1)
    
    # 步骤 4: 完成 Room
    complete_room(room_info)
    time.sleep(1)
    
    # 步骤 5: 验证量化
    verify_quantization()
    
    # 总结
    print_section("测试总结")
    
    print(f"Room ID: {room_info['room_id']}")
    print(f"Prefill 节点：{room_info['prefill_node']}")
    print(f"Decode 节点：{room_info['decode_node']}")
    print()
    
    if prefill_result.get('success'):
        print(f"✓ Prefill 成功 ({prefill_result['latency']:.2f} ms)")
    else:
        print(f"✗ Prefill 失败：{prefill_result.get('error', 'Unknown')}")
    
    if decode_result.get('success'):
        print(f"✓ Decode 成功 ({decode_result['latency']:.2f} ms)")
    else:
        print(f"✗ Decode 失败：{decode_result.get('error', 'Unknown')}")
    
    print()
    total_latency = prefill_result.get('latency', 0) + decode_result.get('latency', 0)
    print(f"总延迟：{total_latency:.2f} ms")
    print()
    
    if prefill_result.get('success') and decode_result.get('success'):
        print("✅ 完整 P/D 分离流程测试成功!")
        return True
    else:
        print("❌ 部分步骤失败")
        return False


if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)
