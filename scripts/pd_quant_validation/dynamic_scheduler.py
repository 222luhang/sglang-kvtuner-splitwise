#!/usr/bin/env python3
"""
Dynamic Scheduler for P/D Disaggregation

基于 vLLM PR #2809 思想的动态调度器实现：
- 动态负载均衡
- 节点健康检查
- Prefix-Aware 路由
- 弹性节点管理

Usage:
    python3 dynamic_scheduler.py --prefill node1:30000,node2:30000 --decode node3:30001,node4:30001
"""

import argparse
import asyncio
import aiohttp
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    OVERLOADED = "overloaded"
    DRAINING = "draining"


@dataclass
class NodeInfo:
    """节点信息"""
    url: str
    role: str  # "prefill" or "decode"
    status: NodeStatus = NodeStatus.HEALTHY
    load: float = 0.0  # 0.0 - 1.0
    running_requests: int = 0
    kv_cache_usage: float = 0.0  # 0.0 - 1.0
    last_heartbeat: float = 0.0
    total_tokens: int = 0
    avg_latency: float = 0.0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    
    @property
    def health_score(self) -> float:
        """计算健康分数 (0-100)"""
        if self.status == NodeStatus.UNHEALTHY:
            return 0.0
        
        score = 100.0
        
        # 负载惩罚
        score -= self.load * 30
        
        # KV Cache 使用惩罚
        score -= self.kv_cache_usage * 20
        
        # 延迟惩罚
        if self.avg_latency > 1.0:
            score -= min(20, (self.avg_latency - 1.0) * 10)
        
        # 命中率奖励
        total = self.prefix_cache_hits + self.prefix_cache_misses
        if total > 0:
            hit_rate = self.prefix_cache_hits / total
            score += hit_rate * 10
        
        return max(0.0, min(100.0, score))
    
    @property
    def prefix_hit_rate(self) -> float:
        """计算 prefix cache 命中率"""
        total = self.prefix_cache_hits + self.prefix_cache_misses
        if total == 0:
            return 0.0
        return self.prefix_cache_hits / total


@dataclass
class RequestInfo:
    """请求信息"""
    request_id: str
    prompt_hash: str  # prompt 的哈希，用于 prefix cache 匹配
    prompt_length: int
    max_tokens: int
    arrival_time: float
    assigned_prefill: Optional[str] = None
    assigned_decode: Optional[str] = None
    prefill_done_time: Optional[float] = None


class DynamicScheduler:
    """动态调度器"""
    
    def __init__(
        self,
        prefill_nodes: List[str],
        decode_nodes: List[str],
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 15.0,
        load_threshold: float = 0.8,
        enable_prefix_aware: bool = True,
    ):
        self.prefill_nodes: Dict[str, NodeInfo] = {}
        self.decode_nodes: Dict[str, NodeInfo] = {}
        self.requests: Dict[str, RequestInfo] = {}
        
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.load_threshold = load_threshold
        self.enable_prefix_aware = enable_prefix_aware
        
        # Prefix Cache 映射：prompt_hash -> node_url
        self.prefix_cache: Dict[str, str] = {}
        
        # 初始化节点
        for url in prefill_nodes:
            self.prefill_nodes[url] = NodeInfo(url=url, role="prefill")
        for url in decode_nodes:
            self.decode_nodes[url] = NodeInfo(url=url, role="decode")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
    
    async def start(self):
        """启动调度器"""
        self.session = aiohttp.ClientSession()
        self._running = True
        
        # 启动心跳检查任务
        asyncio.create_task(self._heartbeat_loop())
        
        # 启动负载均衡器
        asyncio.create_task(self._load_balancer_loop())
        
        logger.info(f"Dynamic Scheduler started")
        logger.info(f"  Prefill nodes: {list(self.prefill_nodes.keys())}")
        logger.info(f"  Decode nodes: {list(self.decode_nodes.keys())}")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        if self.session:
            await self.session.close()
    
    async def _heartbeat_loop(self):
        """心跳检查循环"""
        while self._running:
            try:
                await asyncio.gather(
                    *[self._check_node_health(node) for node in self.prefill_nodes.values()],
                    *[self._check_node_health(node) for node in self.decode_nodes.values()],
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _check_node_health(self, node: NodeInfo):
        """检查节点健康状态"""
        try:
            async with self.session.get(f"{node.url}/health", timeout=5) as resp:
                if resp.status == 200:
                    node.status = NodeStatus.HEALTHY
                    node.last_heartbeat = time.time()
                    
                    # 获取详细状态
                    try:
                        async with self.session.get(f"{node.url}/get_server_info", timeout=5) as info_resp:
                            info = await info_resp.json()
                            node.running_requests = info.get('internal_states', [{}])[0].get('last_gen_throughput', 0)
                            node.kv_cache_usage = info.get('internal_states', [{}])[0].get('memory_usage', {}).get('kvcache', 0) / 20.0  # 归一化
                    except:
                        pass
                else:
                    node.status = NodeStatus.UNHEALTHY
        except Exception as e:
            if time.time() - node.last_heartbeat > self.heartbeat_timeout:
                node.status = NodeStatus.UNHEALTHY
                logger.warning(f"Node {node.url} is unhealthy: {e}")
    
    async def _load_balancer_loop(self):
        """负载均衡循环"""
        while self._running:
            try:
                # 更新节点负载分数
                for node in list(self.prefill_nodes.values()) + list(self.decode_nodes.values()):
                    if node.status == NodeStatus.HEALTHY:
                        if node.load > self.load_threshold:
                            node.status = NodeStatus.OVERLOADED
                    elif node.status == NodeStatus.OVERLOADED:
                        if node.load < self.load_threshold * 0.5:
                            node.status = NodeStatus.HEALTHY
            except Exception as e:
                logger.error(f"Load balancer error: {e}")
            
            await asyncio.sleep(1.0)
    
    def _compute_prompt_hash(self, prompt: str) -> str:
        """计算 prompt 的哈希"""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]
    
    def _select_prefill_node(self, prompt_hash: str) -> Optional[str]:
        """选择 Prefill 节点"""
        healthy_nodes = [
            n for n in self.prefill_nodes.values()
            if n.status in [NodeStatus.HEALTHY, NodeStatus.OVERLOADED]
        ]
        
        if not healthy_nodes:
            return None
        
        # Prefix-Aware 调度
        if self.enable_prefix_aware and prompt_hash in self.prefix_cache:
            cached_node = self.prefix_cache[prompt_hash]
            if cached_node in self.prefill_nodes:
                node = self.prefill_nodes[cached_node]
                if node.status == NodeStatus.HEALTHY:
                    logger.info(f"Prefix cache hit: routing to {cached_node}")
                    node.prefix_cache_hits += 1
                    return cached_node
        
        # 如果没有缓存或缓存节点不可用，选择负载最低的节点
        if self.enable_prefix_aware and prompt_hash in self.prefix_cache:
            node = self.prefill_nodes.get(self.prefix_cache[prompt_hash])
            if node:
                node.prefix_cache_misses += 1
        
        best_node = min(healthy_nodes, key=lambda n: n.load)
        logger.info(f"Selecting prefill node: {best_node.url} (load={best_node.load:.2f})")
        return best_node.url
    
    def _select_decode_node(self) -> Optional[str]:
        """选择 Decode 节点"""
        healthy_nodes = [
            n for n in self.decode_nodes.values()
            if n.status == NodeStatus.HEALTHY
        ]
        
        if not healthy_nodes:
            return None
        
        # 选择负载最低的节点
        best_node = min(healthy_nodes, key=lambda n: n.load)
        logger.info(f"Selecting decode node: {best_node.url} (load={best_node.load:.2f})")
        return best_node.url
    
    async def schedule_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """调度一个请求"""
        request_id = f"req_{int(time.time() * 1000)}"
        prompt_hash = self._compute_prompt_hash(prompt)
        
        request = RequestInfo(
            request_id=request_id,
            prompt_hash=prompt_hash,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            arrival_time=time.time(),
        )
        
        self.requests[request_id] = request
        
        # 选择节点
        prefill_url = self._select_prefill_node(prompt_hash)
        if not prefill_url:
            return {"error": "No healthy prefill node available"}
        
        decode_url = self._select_decode_node()
        if not decode_url:
            return {"error": "No healthy decode node available"}
        
        request.assigned_prefill = prefill_url
        request.assigned_decode = decode_url
        
        # 更新 prefix cache
        if self.enable_prefix_aware:
            self.prefix_cache[prompt_hash] = prefill_url
        
        logger.info(f"Scheduling request {request_id}: prefill={prefill_url}, decode={decode_url}")
        
        # 执行请求
        result = await self._execute_request(request, prompt, max_tokens, temperature)
        
        # 更新统计
        if "error" not in result:
            prefill_node = self.prefill_nodes[prefill_url]
            decode_node = self.decode_nodes[decode_url]
            prefill_node.total_tokens += request.prompt_length
            decode_node.total_tokens += len(result.get("text", "").split())
        
        return result
    
    async def _execute_request(
        self,
        request: RequestInfo,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict:
        """执行请求（通过 Prefill → Decode 流程）"""
        try:
            # 步骤 1: Prefill 阶段
            prefill_start = time.time()
            
            # 注意：实际的 P/D 分离需要 bootstrap room id
            # 这里使用简化的直接调用
            async with self.session.post(
                f"{request.assigned_prefill}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=60
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    prefill_end = time.time()
                    request.prefill_done_time = prefill_end
                    
                    # 更新延迟统计
                    prefill_node = self.prefill_nodes[request.assigned_prefill]
                    prefill_node.avg_latency = (prefill_end - prefill_start)
                    prefill_node.load = min(1.0, prefill_node.running_requests / 100.0)
                    
                    return result
                else:
                    error = await resp.text()
                    return {"error": f"Prefill error: {error}"}
        
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            "prefill_nodes": {
                url: {
                    "status": node.status.value,
                    "load": node.load,
                    "health_score": node.health_score,
                    "prefix_hit_rate": node.prefix_hit_rate,
                }
                for url, node in self.prefill_nodes.items()
            },
            "decode_nodes": {
                url: {
                    "status": node.status.value,
                    "load": node.load,
                    "health_score": node.health_score,
                }
                for url, node in self.decode_nodes.items()
            },
            "prefix_cache_size": len(self.prefix_cache),
            "active_requests": len(self.requests),
        }
    
    async def add_node(self, url: str, role: str):
        """动态添加节点"""
        if role == "prefill":
            self.prefill_nodes[url] = NodeInfo(url=url, role=role)
        elif role == "decode":
            self.decode_nodes[url] = NodeInfo(url=url, role=role)
        logger.info(f"Added {role} node: {url}")
    
    async def remove_node(self, url: str, role: str):
        """动态移除节点"""
        if role == "prefill" and url in self.prefill_nodes:
            node = self.prefill_nodes[url]
            node.status = NodeStatus.DRAINING
            # 等待节点排空
            while node.running_requests > 0:
                await asyncio.sleep(1)
            del self.prefill_nodes[url]
        elif role == "decode" and url in self.decode_nodes:
            node = self.decode_nodes[url]
            node.status = NodeStatus.DRAINING
            while node.running_requests > 0:
                await asyncio.sleep(1)
            del self.decode_nodes[url]
        logger.info(f"Removed {role} node: {url}")


async def main():
    parser = argparse.ArgumentParser(description="Dynamic P/D Scheduler")
    parser.add_argument("--prefill", type=str, required=True, help="Prefill nodes (comma-separated)")
    parser.add_argument("--decode", type=str, required=True, help="Decode nodes (comma-separated)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Scheduler host")
    parser.add_argument("--port", type=int, default=9000, help="Scheduler port")
    
    args = parser.parse_args()
    
    prefill_nodes = [n.strip() for n in args.prefill.split(",")]
    decode_nodes = [n.strip() for n in args.decode.split(",")]
    
    scheduler = DynamicScheduler(
        prefill_nodes=[f"http://{n}" for n in prefill_nodes],
        decode_nodes=[f"http://{n}" for n in decode_nodes],
    )
    
    await scheduler.start()
    
    # 创建 HTTP 服务器
    async def handle_request(request):
        if request.path == "/schedule":
            data = await request.json()
            result = await scheduler.schedule_request(
                prompt=data.get("prompt", ""),
                max_tokens=data.get("max_tokens", 256),
                temperature=data.get("temperature", 0.7),
            )
            return aiohttp.web.json_response(result)
        elif request.path == "/status":
            return aiohttp.web.json_response(scheduler.get_status())
        elif request.path == "/health":
            return aiohttp.web.json_response({"status": "healthy"})
        else:
            return aiohttp.web.Response(status=404)
    
    app = aiohttp.web.Application()
    app.router.add_route("*", "/{path:.*}", handle_request)
    
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, args.host, args.port)
    await site.start()
    
    logger.info(f"Scheduler HTTP server running on http://{args.host}:{args.port}")
    
    # 保持运行
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
