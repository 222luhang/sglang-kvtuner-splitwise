#!/usr/bin/env python3
"""
Test Client for P/D Disaggregation with Bootstrap

测试完整的 P/D 分离流程：
1. 获取 bootstrap room id
2. 发送 Prefill 请求
3. 发送 Decode 请求
4. 验证结果
"""

import aiohttp
import asyncio
import json
import time
import sys

BOOTSTRAP_URL = "http://10.60.6.75:8998"
PREFILL_URL = "http://10.60.6.75:30000"
DECODE_URL = "http://10.60.19.152:30001"
MODEL_PATH = "/data/Qwen/Qwen2.5-7B"


async def get_bootstrap_room():
    """获取 bootstrap room id"""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BOOTSTRAP_URL}/bootstrap") as resp:
            data = await resp.json()
            if data.get('success'):
                print(f"✓ 获取 Room ID: {data['room_id'][:8]}...")
                print(f"  Prefill: {data['prefill_node']}")
                print(f"  Decode: {data['decode_node']}")
                return data
            else:
                print(f"✗ 获取 Room ID 失败：{data}")
                return None


async def test_prefill_with_room(room_id: str, prefill_node: str):
    """使用 room id 发送 Prefill 请求"""
    url = f"http://{prefill_node}/generate"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={
                "text": "Hello, this is a test of P/D disaggregation with KVTuner quantization.",
                "sampling_params": {
                    "max_new_tokens": 32,
                    "temperature": 0.7,
                },
                "bootstrap_room": room_id,  # 传递 room id
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            result = await resp.json()
            return result


async def test_decode_with_room(room_id: str, decode_node: str):
    """使用 room id 发送 Decode 请求"""
    url = f"http://{decode_node}/generate"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json={
                "text": "Continue from prefill.",
                "sampling_params": {
                    "max_new_tokens": 32,
                    "temperature": 0.7,
                },
                "bootstrap_room": room_id,  # 传递 room id
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            result = await resp.json()
            return result


async def complete_room(room_id: str):
    """标记 room 完成"""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BOOTSTRAP_URL}/bootstrap/{room_id}/complete") as resp:
            data = await resp.json()
            if data.get('success'):
                print(f"✓ Room {room_id[:8]}... 标记为完成")


async def main():
    """运行完整测试"""
    print("\n")
    print("="*60)
    print("P/D 分离完整流程测试（带 Bootstrap）")
    print("="*60)
    print("\n")
    
    # 步骤 1: 获取 bootstrap room
    print("步骤 1: 获取 Bootstrap Room ID")
    print("-"*60)
    room_info = await get_bootstrap_room()
    
    if not room_info:
        print("✗ 无法获取 Room ID，测试终止")
        return
    
    room_id = room_info['room_id']
    prefill_node = room_info['prefill_node']
    decode_node = room_info['decode_node']
    print()
    
    # 步骤 2: 发送 Prefill 请求
    print("步骤 2: 发送 Prefill 请求")
    print("-"*60)
    start = time.time()
    prefill_result = await test_prefill_with_room(room_id, prefill_node)
    prefill_latency = (time.time() - start) * 1000
    
    if "text" in prefill_result:
        print(f"✓ Prefill 成功")
        print(f"  响应：{prefill_result['text'][:100]}...")
        print(f"  延迟：{prefill_latency:.2f} ms")
    else:
        print(f"✗ Prefill 失败：{prefill_result}")
    
    print()
    
    # 步骤 3: 发送 Decode 请求
    print("步骤 3: 发送 Decode 请求")
    print("-"*60)
    start = time.time()
    decode_result = await test_decode_with_room(room_id, decode_node)
    decode_latency = (time.time() - start) * 1000
    
    if "text" in decode_result:
        print(f"✓ Decode 成功")
        print(f"  响应：{decode_result['text'][:100]}...")
        print(f"  延迟：{decode_latency:.2f} ms")
    else:
        print(f"✗ Decode 失败：{decode_result}")
    
    print()
    
    # 步骤 4: 标记 room 完成
    print("步骤 4: 完成 Room")
    print("-"*60)
    await complete_room(room_id)
    print()
    
    # 总结
    print("="*60)
    print("测试总结")
    print("="*60)
    print(f"Room ID: {room_id[:8]}...")
    print(f"Prefill 节点：{prefill_node}")
    print(f"Decode 节点：{decode_node}")
    print(f"Prefill 延迟：{prefill_latency:.2f} ms")
    print(f"Decode 延迟：{decode_latency:.2f} ms")
    print(f"总延迟：{prefill_latency + decode_latency:.2f} ms")
    print()
    
    if "text" in prefill_result and "text" in decode_result:
        print("✅ 完整 P/D 分离流程测试成功!")
    else:
        print("❌ 部分步骤失败")
    
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
