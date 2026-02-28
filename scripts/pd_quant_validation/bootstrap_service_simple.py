#!/usr/bin/env python3
"""
Simple Bootstrap Service for P/D Disaggregation

使用 Python 内置模块，无需额外依赖。

Usage:
    python3 bootstrap_service_simple.py --port 8998
"""

import json
import time
import uuid
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class BootstrapManager:
    """简单的 Bootstrap 管理器"""
    
    def __init__(self, prefill_nodes: list, decode_nodes: list, room_ttl: int = 300):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes
        self.room_ttl = room_ttl
        self.rooms = {}
        self.node_load = {node: 0 for node in prefill_nodes + decode_nodes}
        self.lock = threading.Lock()
    
    def create_room(self) -> dict:
        """创建新的 room"""
        with self.lock:
            # 负载均衡
            prefill_node = min(self.prefill_nodes, key=lambda n: self.node_load.get(n, 0))
            decode_node = min(self.decode_nodes, key=lambda n: self.node_load.get(n, 0))
            
            room_id = str(uuid.uuid4())
            now = time.time()
            
            room = {
                "room_id": room_id,
                "prefill_node": prefill_node,
                "decode_node": decode_node,
                "created_at": now,
                "expires_at": now + self.room_ttl,
                "status": "active",
            }
            
            self.rooms[room_id] = room
            self.node_load[prefill_node] += 1
            self.node_load[decode_node] += 1
            
            logger.info(f"Created room {room_id[:8]}... on {prefill_node} → {decode_node}")
            return room
    
    def get_room(self, room_id: str) -> dict:
        """获取 room 信息"""
        with self.lock:
            room = self.rooms.get(room_id)
            if not room:
                return None
            
            if time.time() > room['expires_at']:
                room['status'] = 'expired'
                return None
            
            return room
    
    def complete_room(self, room_id: str):
        """标记 room 完成"""
        with self.lock:
            room = self.rooms.get(room_id)
            if room:
                room['status'] = 'completed'
                self.node_load[room['prefill_node']] = max(0, self.node_load[room['prefill_node']] - 1)
                self.node_load[room['decode_node']] = max(0, self.node_load[room['decode_node']] - 1)
                logger.info(f"Completed room {room_id[:8]}...")
    
    def cleanup_expired(self):
        """清理过期 rooms"""
        with self.lock:
            now = time.time()
            expired = [rid for rid, r in self.rooms.items() if now > r['expires_at']]
            for rid in expired:
                del self.rooms[rid]
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired rooms")


class BootstrapHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    manager = None
    
    def do_POST(self):
        """处理 POST 请求"""
        parsed = urlparse(self.path)
        
        if parsed.path == '/bootstrap':
            self._handle_create_room()
        elif parsed.path.startswith('/bootstrap/') and parsed.path.endswith('/complete'):
            room_id = parsed.path.split('/')[2]
            self._handle_complete_room(room_id)
        else:
            self.send_error(404, "Not Found")
    
    def do_GET(self):
        """处理 GET 请求"""
        parsed = urlparse(self.path)
        
        if parsed.path == '/health':
            self._handle_health()
        elif parsed.path == '/status':
            self._handle_status()
        elif parsed.path.startswith('/bootstrap/'):
            room_id = parsed.path.split('/')[2]
            self._handle_get_room(room_id)
        else:
            self.send_error(404, "Not Found")
    
    def _handle_create_room(self):
        """创建 Room"""
        try:
            room = self.manager.create_room()
            response = {
                "success": True,
                "room_id": room['room_id'],
                "prefill_node": room['prefill_node'],
                "decode_node": room['decode_node'],
                "expires_in": room['expires_at'] - room['created_at'],
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, 500)
    
    def _handle_get_room(self, room_id: str):
        """获取 Room"""
        room = self.manager.get_room(room_id)
        if not room:
            self._send_json({"success": False, "error": "Room not found"}, 404)
        else:
            self._send_json({"success": True, "room": room})
    
    def _handle_complete_room(self, room_id: str):
        """完成 Room"""
        self.manager.complete_room(room_id)
        self._send_json({"success": True})
    
    def _handle_health(self):
        """健康检查"""
        active = len([r for r in self.manager.rooms.values() if r['status'] == 'active'])
        self._send_json({"status": "healthy", "active_rooms": active})
    
    def _handle_status(self):
        """服务状态"""
        self._send_json({
            "active_rooms": len(self.manager.rooms),
            "prefill_nodes": self.manager.prefill_nodes,
            "decode_nodes": self.manager.decode_nodes,
            "node_load": self.manager.node_load,
        })
    
    def _send_json(self, data: dict, status: int = 200):
        """发送 JSON 响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """自定义日志"""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(host: str, port: int, prefill_nodes: list, decode_nodes: list, room_ttl: int = 300):
    """运行服务器"""
    manager = BootstrapManager(prefill_nodes, decode_nodes, room_ttl)
    BootstrapHandler.manager = manager
    
    server = HTTPServer((host, port), BootstrapHandler)
    
    logger.info(f"Bootstrap server starting on {host}:{port}")
    logger.info(f"  Prefill nodes: {prefill_nodes}")
    logger.info(f"  Decode nodes: {decode_nodes}")
    logger.info(f"  Room TTL: {room_ttl}s")
    
    # 启动清理线程
    def cleanup_loop():
        while True:
            time.sleep(60)
            manager.cleanup_expired()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Bootstrap Service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--prefill-nodes", type=str, default="10.60.6.75:30000,10.60.9.62:30000")
    parser.add_argument("--decode-nodes", type=str, default="10.60.19.152:30001,10.60.176.217:30001")
    parser.add_argument("--room-ttl", type=int, default=300)
    
    args = parser.parse_args()
    
    prefill_nodes = args.prefill_nodes.split(",")
    decode_nodes = args.decode_nodes.split(",")
    
    run_server(args.host, args.port, prefill_nodes, decode_nodes, args.room_ttl)


if __name__ == "__main__":
    main()
