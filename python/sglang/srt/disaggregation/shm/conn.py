"""
Shared Memory Transfer Backend for Single-Node P/D Disaggregation

This module provides a shared memory-based KV Cache transfer mechanism
for single-node Prefill/Decode disaggregation scenarios where:
- No RDMA hardware is available
- POSIX backend cannot directly register VRAM
- GPU P2P is not supported (e.g., RTX 4090)

Transfer Flow:
1. Prefill: GPU0 VRAM -> CPU Pinned Memory -> Shared Memory
2. Decode: Shared Memory -> CPU Pinned Memory -> GPU1 VRAM
"""

from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.server_args import ServerArgs
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Guard bytes for message validation
SHM_GUARD = b"ShmMsgGuard"


@dataclasses.dataclass
class ShmTransferInfo:
    """Transfer info sent from Decode to Prefill."""
    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[int]

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if len(msg) > 7 and msg[7] != b"":
            dst_state_indices = list(np.frombuffer(msg[7], dtype=np.int32))
        else:
            dst_state_indices = []
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
        )


@dataclasses.dataclass
class ShmRegisterInfo:
    """Registration info from Decode to Prefill."""
    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list
    dst_aux_ptrs: list
    dst_state_data_ptrs: list
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if len(msg) > 7 and msg[7] != b"":
            dst_state_data_ptrs = list(struct.unpack(f"{len(msg[7]) // 8}Q", msg[7]))
        else:
            dst_state_data_ptrs = []
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=int(msg[11].decode("ascii")),
        )


@dataclasses.dataclass
class ShmTransferStatus:
    """Track transfer completion status."""
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    num_pp_ranks_expected: Optional[int] = None
    received_aux: bool = False
    received_state_per_pp: Set[int] = dataclasses.field(default_factory=set)
    expects_state: bool = False
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        if (
            self.expects_state
            and len(self.received_state_per_pp) < self.num_pp_ranks_expected
        ):
            return False
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True


class ShmBufferPool:
    """Pool of shared memory buffers for KV Cache transfer."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, buffer_size_mb: int = 256, num_buffers: int = 8):
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self.num_buffers = num_buffers
        self._buffers: Dict[str, shared_memory.SharedMemory] = {}
        self._buffer_locks: Dict[str, threading.Lock] = {}
        self._counter = 0
        
    @classmethod
    def get_instance(cls, buffer_size_mb: int = 256, num_buffers: int = 8):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(buffer_size_mb, num_buffers)
            return cls._instance
    
    def create_buffer(self, name: Optional[str] = None, size: Optional[int] = None) -> Tuple[str, shared_memory.SharedMemory]:
        """Create a new shared memory buffer."""
        size = size or self.buffer_size
        name = name or f"sglang_shm_{uuid.uuid4().hex[:8]}_{self._counter}"
        self._counter += 1
        
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        self._buffers[name] = shm
        self._buffer_locks[name] = threading.Lock()
        logger.debug(f"Created shared memory buffer: {name}, size: {size}")
        return name, shm
    
    def get_buffer(self, name: str) -> Optional[shared_memory.SharedMemory]:
        """Get an existing buffer by name."""
        if name in self._buffers:
            return self._buffers[name]
        try:
            shm = shared_memory.SharedMemory(name=name)
            self._buffers[name] = shm
            return shm
        except FileNotFoundError:
            return None
    
    def release_buffer(self, name: str):
        """Release a buffer."""
        if name in self._buffers:
            try:
                self._buffers[name].close()
                self._buffers[name].unlink()
            except Exception as e:
                logger.warning(f"Failed to release buffer {name}: {e}")
            del self._buffers[name]
            if name in self._buffer_locks:
                del self._buffer_locks[name]


class ShmKVManager(CommonKVManager):
    """Shared Memory KV Manager for single-node P/D disaggregation."""
    
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        
        self.agent_name = str(uuid.uuid4())
        self.buffer_pool = ShmBufferPool.get_instance()
        
        # Transfer tracking
        self.transfer_statuses: Dict[int, ShmTransferStatus] = defaultdict(ShmTransferStatus)
        self.transfer_infos: Dict[int, Dict[str, ShmTransferInfo]] = defaultdict(dict)
        self.decode_kv_args_table: Dict[str, ShmRegisterInfo] = {}
        
        # Pinned CPU memory for GPU transfers
        self._pinned_buffers: Dict[Tuple[int, int], torch.Tensor] = {}
        
        logger.info(f"ShmKVManager initialized with mode: {disaggregation_mode}")
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._start_heartbeat_checker_thread()
    
    def _get_pinned_buffer(self, size: int, gpu_id: int) -> torch.Tensor:
        """Get or create a pinned CPU buffer for GPU transfer."""
        key = (size, gpu_id)
        if key not in self._pinned_buffers:
            buf = torch.empty(size, dtype=torch.uint8, device="cpu")
            buf.pin_memory()
            self._pinned_buffers[key] = buf
        return self._pinned_buffers[key]
    
    def _start_heartbeat_checker_thread(self):
        """Start heartbeat checker for Decode mode."""
        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())
                for bootstrap_addr in addresses:
                    try:
                        import requests
                        session = self.session_pool.get(bootstrap_addr)
                        if session:
                            response = session.get(
                                f"http://{bootstrap_addr}/health",
                                timeout=(2, 3),
                            )
                            if response.status_code != 200:
                                logger.warning(f"Prefill health check failed: {bootstrap_addr}")
                    except Exception as e:
                        logger.warning(f"Prefill health check error: {e}")
        
        threading.Thread(target=heartbeat_checker, daemon=True).start()
    
    def _start_bootstrap_thread(self):
        """Start bootstrap thread for Prefill mode."""
        def bootstrap_thread():
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                assert waiting_req_bytes[0] == SHM_GUARD
                waiting_req_bytes = waiting_req_bytes[1:]
                
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                
                if room == "None":
                    self._add_remote_peer(ShmRegisterInfo.from_zmq(waiting_req_bytes))
                    continue
                
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = ShmTransferInfo.from_zmq(waiting_req_bytes)
                
                required_dst_info_num = self.transfer_infos[room][agent_name].required_dst_info_num
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.update_status(room, KVPoll.WaitingForInput)
        
        threading.Thread(target=bootstrap_thread, daemon=True).start()
    
    def _add_remote_peer(self, decode_kv_args: ShmRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} already registered")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        logger.info(f"Registered peer: {agent_name}")
    
    def register_to_bootstrap(self):
        """Register to bootstrap server (Decode mode)."""
        pass  # Handled by parent class
    
    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """Add a transfer request (Prefill mode)."""
        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        
        for req in reqs_to_be_processed:
            if req.is_dummy():
                continue
            
            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert req.agent_name in self.decode_kv_args_table
            
            dst_info = self.decode_kv_args_table[req.agent_name]
            
            # Perform shared memory transfer
            self._transfer_kv_via_shm(
                kv_indices,
                chunked_dst_kv_indice,
                dst_info,
                bootstrap_room,
                chunk_id,
            )
            
            if is_last:
                # Transfer aux data
                self._transfer_aux_via_shm(aux_index, dst_info, bootstrap_room)
        
        if is_last:
            del self.transfer_infos[bootstrap_room]
    
    def _transfer_kv_via_shm(
        self,
        src_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_info: ShmRegisterInfo,
        room: int,
        chunk_id: int,
    ):
        """Transfer KV Cache via shared memory."""
        # Group contiguous indices
        src_blocks, dst_blocks = group_concurrent_contiguous(src_kv_indices, dst_kv_indices)
        
        src_kv_ptrs = self.kv_args.kv_data_ptrs
        dst_kv_ptrs = dst_info.dst_kv_ptrs
        item_lens = self.kv_args.kv_item_lens
        src_gpu_id = self.kv_args.gpu_id
        dst_gpu_id = dst_info.gpu_id
        
        # Transfer each layer
        if self.is_mla_backend:
            layers_params = [(src_kv_ptrs[i], dst_kv_ptrs[i], item_lens[i]) 
                           for i in range(len(src_kv_ptrs))]
        else:
            # MHA: K and V separate
            num_layers = len(src_kv_ptrs) // 2
            layers_params = []
            for i in range(num_layers):
                layers_params.append((src_kv_ptrs[i], dst_kv_ptrs[i], item_lens[i]))
                layers_params.append((src_kv_ptrs[i + num_layers], dst_kv_ptrs[i + num_layers], item_lens[i + num_layers]))
        
        total_bytes = 0
        start_time = time.time()
        
        for src_ptr, dst_ptr, item_len in layers_params:
            for src_block, dst_block in zip(src_blocks, dst_blocks):
                src_start = src_block[0]
                dst_start = dst_block[0]
                block_len = len(src_block)
                
                # Calculate byte addresses
                src_addr = src_ptr + src_start * item_len
                dst_addr = dst_ptr + dst_start * item_len
                transfer_size = block_len * item_len
                
                # Create GPU tensor views using ctypes
                import ctypes
                src_tensor = torch.empty((transfer_size,), dtype=torch.uint8, device=f"cuda:{src_gpu_id}")
                dst_tensor = torch.empty((transfer_size,), dtype=torch.uint8, device=f"cuda:{dst_gpu_id}")
                
                # Copy GPU -> CPU (pinned)
                cpu_tensor = self._get_pinned_buffer(transfer_size, src_gpu_id)
                cpu_tensor[:transfer_size].copy_(src_tensor, non_blocking=True)
                torch.cuda.synchronize(src_gpu_id)
                
                # Create shared memory and copy
                shm_name, shm = self.buffer_pool.create_buffer(size=transfer_size)
                np_array = np.ndarray((transfer_size,), dtype=np.uint8, buffer=shm.buf)
                np_array[:] = cpu_tensor[:transfer_size].numpy()
                
                # Copy from shared memory to GPU
                dst_tensor.copy_(torch.from_numpy(np_array.copy()), non_blocking=True)
                torch.cuda.synchronize(dst_gpu_id)
                
                # Cleanup
                self.buffer_pool.release_buffer(shm_name)
                
                total_bytes += transfer_size
        
        elapsed = time.time() - start_time
        logger.debug(f"Transferred {total_bytes / 1e6:.2f} MB in {elapsed * 1000:.2f} ms")
    
    def _transfer_aux_via_shm(self, aux_index: int, dst_info: ShmRegisterInfo, room: int):
        """Transfer auxiliary data via shared memory."""
        src_aux_ptrs = self.kv_args.aux_data_ptrs
        src_aux_lens = self.kv_args.aux_item_lens
        dst_aux_ptrs = dst_info.dst_aux_ptrs
        
        for i, (src_ptr, dst_ptr) in enumerate(zip(src_aux_ptrs, dst_aux_ptrs)):
            length = src_aux_lens[i]
            src_addr = src_ptr + length * aux_index
            dst_addr = dst_ptr + length * dst_info.dst_aux_index
            
            # Copy via shared memory
            shm_name, shm = self.buffer_pool.create_buffer(size=length)
            np_array = np.ndarray((length,), dtype=np.uint8, buffer=shm.buf)
            
            # CPU to CPU transfer (aux data is in DRAM)
            import ctypes
            ctypes.memmove(np_array.ctypes.data, src_addr, length)
            ctypes.memmove(dst_addr, np_array.ctypes.data, length)
            
            self.buffer_pool.release_buffer(shm_name)
    
    def update_transfer_status(self):
        """Update transfer status (Decode mode)."""
        # For shared memory, transfers are synchronous
        pass
    
    def check_transfer_done(self, room: int):
        """Check if transfer is complete."""
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()


class ShmKVSender(CommonKVSender):
    """Shared Memory KV Sender."""
    
    def __init__(
        self,
        mgr: ShmKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.has_sent = False
        self.chunk_id = 0
    
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
            state_indices,
        )
        self.chunk_id += 1
        
        if is_last:
            self.has_sent = True
    
    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        # Shared memory transfers are synchronous
        return KVPoll.Success
    
    def failure_exception(self):
        raise RuntimeError("ShmKVSender Exception")


class ShmKVReceiver(CommonKVReceiver):
    """Shared Memory KV Receiver."""
    
    def __init__(
        self,
        mgr: ShmKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)
        self.init_time = None
    
    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(f"Could not fetch prefill info from {self.bootstrap_addr}")
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info.get("is_dummy", False)
            
            with lock:
                sock.send_multipart([
                    SHM_GUARD,
                    str(self.bootstrap_room).encode("ascii"),
                    self.kv_mgr.local_ip.encode("ascii"),
                    str(self.kv_mgr.rank_port).encode("ascii"),
                    self.kv_mgr.agent_name.encode("ascii"),
                    kv_indices.tobytes() if not is_dummy else b"",
                    str(aux_index).encode("ascii"),
                    str(self.required_dst_info_num).encode("ascii"),
                    (np.array(state_indices, dtype=np.int32).tobytes()
                     if not is_dummy and state_indices is not None else b""),
                ])
        
        self.started_transfer = True
        self.init_time = time.time()
    
    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        
        if not self.started_transfer:
            return KVPoll.WaitingForInput
        
        # Shared memory transfers are synchronous
        # Data should already be in place
        self.conclude_state = KVPoll.Success
        return KVPoll.Success
    
    def failure_exception(self):
        raise RuntimeError("ShmKVReceiver Exception")


class ShmKVBootstrapServer(CommonKVBootstrapServer):
    """Shared Memory KV Bootstrap Server."""
    pass