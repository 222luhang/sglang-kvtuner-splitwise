#!/usr/bin/env python3
"""
Bootstrap Service for P/D Disaggregation

为 P/D 分离架构提供 bootstrap room id 获取服务。

Usage:
    python3 bootstrap_service.py --host 0.0.0.0 --port 8998
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from aiohttp import web
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RoomInfo:
    """Room 信息"""
    room_id: str
    prefill_node: str
    decode_node: str
    created_at: float
    expires_at: float
    status: str = "active"  # active, expired, completed
    
    def to_dict(self):
        return asdict(self)


class BootstrapManager:
    """Bootstrap 管理器"""
    
    def __init__(self, prefill_nodes: list, decode_nodes: list, room_ttl: int = 300):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes
        self.room_ttl = room_ttl  # Room 存活时间（秒）
        self.rooms: Dict[str, RoomInfo] = {}
        self.node_load: Dict[str, int] = {node: 0 for node in prefill_nodes + decode_nodes}
    
    def create_room(self) -> RoomInfo:
        """创建新的 room"""
        # 负载均衡选择节点
        prefill_node = min(self.prefill_nodes, key=lambda n: self.node_load.get(n, 0))
        decode_node = min(self.decode_nodes, key=lambda n: self.node_load.get(n, 0))
        
        room_id = str(uuid.uuid4())
        now = time.time()
        
        room = RoomInfo(
            room_id=room_id,
            prefill_node=prefill_node,
            decode_node=decode_node,
            created_at=now,
            expires_at=now + self.room_ttl,
        )
        
        self.rooms[room_id] = room
        
        # 更新节点负载
        self.node_load[prefill_node] = self.node_load.get(prefill_node, 0) + 1
        self.node_load[decode_node] = self.node_load.get(decode_node, 0) + 1
        
        logger.info(f"Created room {room_id[:8]}... on {prefill_node} → {decode_node}")
        return room
    
    def get_room(self, room_id: str) -> Optional[RoomInfo]:
        """获取 room 信息"""
        room = self.rooms.get(room_id)
        if not room:
            return None
        
        # 检查是否过期
        if time.time() > room.expires_at:
            room.status = "expired"
            return None
        
        return room
    
    def complete_room(self, room_id: str):
        """标记 room 完成"""
        room = self.rooms.get(room_id)
        if room:
            room.status = "completed"
            # 减少负载计数
            self.node_load[room.prefill_node] = max(0, self.node_load.get(room.prefill_node, 0) - 1)
            self.node_load[room.decode_node] = max(0, self.node_load.get(room.decode_node, 0) - 1)
    
    def cleanup_expired(self):
        """清理过期的 rooms"""
        now = time.time()
        expired_rooms = [
            room_id for room_id, room in self.rooms.items()
            if now > room.expires_at
        ]
        for room_id in expired_rooms:
            del self.rooms[room_id]
            logger.debug(f"Cleaned up expired room {room_id[:8]}...")
        
        if expired_rooms:
            logger.info(f"Cleaned up {len(expired_rooms)} expired rooms")


class BootstrapService:
    """Bootstrap 服务"""
    
    def __init__(
        self,
        prefill_nodes: list,
        decode_nodes: list,
        host: str = "0.0.0.0",
        port: int = 8998,
        room_ttl: int = 300,
    ):
        self.host = host
        self.port = port
        self.manager = BootstrapManager(prefill_nodes, decode_nodes, room_ttl)
        self.app = web.Application()
        self._setup_routes()
        self._setup_background_tasks()
    
    def _setup_routes(self):
        """设置 HTTP 路由"""
        self.app.router.add_post("/bootstrap", self.handle_create_room)
        self.app.router.add_get("/bootstrap/{room_id}", self.handle_get_room)
        self.app.router.add_post("/bootstrap/{room_id}/complete", self.handle_complete_room)
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/status", self.handle_status)
    
    def _setup_background_tasks(self):
        """设置后台任务"""
        self.app.on_startup.append(self._start_background_tasks)
        self.app.on_cleanup.append(self._cleanup_background_tasks)
    
    async def _start_background_tasks(self, app):
        """启动后台任务"""
        app['cleanup_task'] = asyncio.create_task(self._cleanup_loop())
        logger.info("Bootstrap service started")
    
    async def _cleanup_background_tasks(self, app):
        """清理后台任务"""
        app['cleanup_task'].cancel()
        try:
            await app['cleanup_task']
        except asyncio.CancelledError:
            pass
        logger.info("Bootstrap service stopped")
    
    async def _cleanup_loop(self):
        """定期清理过期 rooms"""
        while True:
            await asyncio.sleep(60)  # 每分钟清理一次
            self.manager.cleanup_expired()
    
    async def handle_create_room(self, request):
        """创建 Room"""
        try:
            # 可选：从请求中指定节点
            data = await request.json() if request.can_read_body else {}
            
            room = self.manager.create_room()
            
            response_data = {
                "success": True,
                "room_id": room.room_id,
                "prefill_node": room.prefill_node,
                "decode_node": room.decode_node,
                "expires_in": room.expires_at - room.created_at,
            }
            
            logger.info(f"Created room: {response_data}")
            return web.json_response(response_data)
        
        except Exception as e:
            logger.error(f"Error creating room: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )
    
    async def handle_get_room(self, request):
        """获取 Room 信息"""
        room_id = request.match_info['room_id']
        
        room = self.manager.get_room(room_id)
        if not room:
            return web.json_response(
                {"success": False, "error": "Room not found or expired"},
                status=404
            )
        
        return web.json_response({
            "success": True,
            "room": room.to_dict()
        })
    
    async def handle_complete_room(self, request):
        """完成 Room"""
        room_id = request.match_info['room_id']
        
        room = self.manager.get_room(room_id)
        if not room:
            return web.json_response(
                {"success": False, "error": "Room not found or expired"},
                status=404
            )
        
        self.manager.complete_room(room_id)
        
        return web.json_response({"success": True})
    
    async def handle_health(self, request):
        """健康检查"""
        return web.json_response({
            "status": "healthy",
            "active_rooms": len([r for r in self.manager.rooms.values() if r.status == "active"]),
        })
    
    async def handle_status(self, request):
        """服务状态"""
        return web.json_response({
            "active_rooms": len(self.manager.rooms),
            "prefill_nodes": self.manager.prefill_nodes,
            "decode_nodes": self.manager.decode_nodes,
            "node_load": self.manager.node_load,
        })
    
    def run(self):
        """运行服务"""
        web.run_app(
            self.app,
            host=self.host,
            port=self.port,
            print=lambda x: logger.info(x)
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap Service for P/D Disaggregation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8998, help="Port to bind")
    parser.add_argument("--prefill-nodes", type=str, 
                       default="10.60.6.75:30000,10.60.9.62:30000",
                       help="Comma-separated prefill nodes")
    parser.add_argument("--decode-nodes", type=str,
                       default="10.60.19.152:30001,10.60.176.217:30001",
                       help="Comma-separated decode nodes")
    parser.add_argument("--room-ttl", type=int, default=300, help="Room TTL in seconds")
    
    args = parser.parse_args()
    
    prefill_nodes = args.prefill_nodes.split(",")
    decode_nodes = args.decode_nodes.split(",")
    
    logger.info(f"Starting bootstrap service...")
    logger.info(f"  Prefill nodes: {prefill_nodes}")
    logger.info(f"  Decode nodes: {decode_nodes}")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Room TTL: {args.room_ttl}s")
    
    service = BootstrapService(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        host=args.host,
        port=args.port,
        room_ttl=args.room_ttl,
    )
    
    service.run()


if __name__ == "__main__":
    main()
