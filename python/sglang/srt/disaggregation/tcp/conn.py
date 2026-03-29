"""
TCP-based KV cache transfer backend for P/D disaggregation.

This backend transfers KV caches between prefill and decode servers using plain
TCP sockets (via ZeroMQ), making it compatible with environments that do not
have InfiniBand or NVLink hardware (as required by MSCCL++ / Mooncake / NIXL).

Protocol overview
-----------------
Control plane (HTTP + ZMQ):
  * Bootstrap server  – same HTTP mechanism as CommonKVBootstrapServer (reused).
  * ZMQ PUSH/PULL sockets are used by the receiver to send transfer descriptors
    (dst KV indices, aux index) to the sender side, exactly like mooncake/nixl.

Data plane (raw TCP sockets):
  * The sender opens a raw TCP server socket per rank.
  * For every transfer request the sender receives via ZMQ:
    1. A background thread copies KV pages from CUDA to CPU **layer by layer**.
    2. Each layer's data is sent immediately over the TCP socket so that the next
       model layer can start computing while the previous layer's data is already
       on its way – this hides transfer latency (pipeline overlap).
  * The receiver connects to the sender's TCP port, receives data layer by layer,
    copies each layer back to GPU memory.

Layer-wise pipeline integration
---------------------------------
When ``enable_layer_pipeline=True`` (the default), the sender exposes a
``send_layer(layer_id, src_indices)`` method.  The attention backend can call
this after writing each layer's KV to the pool so that transfer and compute
overlap at the layer granularity.  When the flag is False the legacy behaviour
(send all layers at once after the forward pass) is used.
"""

from __future__ import annotations

import ctypes
import logging
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

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
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

# Message framing: every data message is preceded by an 8-byte header
# [4 bytes: layer_id (int32)] [4 bytes: payload_length_bytes (uint32)]
_HEADER_FMT = "!iI"  # network byte order: signed int32 + unsigned int32
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Special layer_id value that signals "all done"
_MSG_DONE = -1

# How long (seconds) the TCP receiver thread waits for the first byte before
# considering the connection dead.
_RECV_TIMEOUT_S = 120

# ZMQ message field indices sent by TCPKVReceiver → TCPKVSender (over ZMQ PULL)
# Format (multi-part ZMQ message):
#   [0] room (ascii int)
#   [1] receiver_ip (ascii str)
#   [2] receiver_tcp_port (ascii int)
#   [3] dst_kv_indices (raw int32 bytes)
#   [4] dst_aux_index (ascii int)
#   [5] required_dst_info_num (ascii int)


def _send_exact(sock: socket.socket, data: bytes) -> None:
    """Send all bytes, raising RuntimeError on failure."""
    total = 0
    view = memoryview(data)
    while total < len(data):
        sent = sock.send(view[total:])
        if sent == 0:
            raise RuntimeError("TCP connection broken while sending")
        total += sent


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly *n* bytes, raising RuntimeError on failure."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        chunk = sock.recv_into(view[pos:], n - pos)
        if chunk == 0:
            raise RuntimeError("TCP connection broken while receiving")
        pos += chunk
    return bytes(buf)


def _send_layer_data(
    sock: socket.socket,
    layer_id: int,
    data: bytes,
) -> None:
    """Send one layer's KV data with a framing header."""
    header = struct.pack(_HEADER_FMT, layer_id, len(data))
    _send_exact(sock, header)
    _send_exact(sock, data)


def _recv_layer_data(sock: socket.socket) -> Tuple[int, bytes]:
    """Receive one layer's KV data; returns (layer_id, data)."""
    header = _recv_exact(sock, _HEADER_SIZE)
    layer_id, length = struct.unpack(_HEADER_FMT, header)
    data = _recv_exact(sock, length) if length > 0 else b""
    return layer_id, data


# ---------------------------------------------------------------------------
# TCPKVManager
# ---------------------------------------------------------------------------


class TCPKVManager(CommonKVManager):
    """
    Manages the TCP server socket (for prefill-side sends) and the shared
    state needed by all senders/receivers on this rank.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # Start TCP server before calling super().__init__() so the port is
        # known when register_to_bootstrap() is called inside the parent.
        self._tcp_port: int = 0
        self._tcp_server: Optional[socket.socket] = None
        self._pending_transfers: Dict[int, "_PendingTransfer"] = {}
        self._pending_lock = threading.Lock()

        if disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_tcp_server()

        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        # Map from bootstrap_room → threading.Event signalling transfer done
        self._room_done: Dict[int, threading.Event] = {}
        self._room_lock = threading.Lock()

        if disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_prefill_background_thread()

    def register_to_bootstrap(self) -> None:
        """
        Override to include the TCP port in the bootstrap PUT registration.

        The ``tcp_port`` field is appended to the standard payload so that
        ``TCPKVBootstrapServer`` can store it and return it to decode workers
        during the GET query, giving them the address to connect to.
        """
        import requests as _requests

        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            super().register_to_bootstrap()
            return

        # Determine bootstrap host (same logic as the parent)
        if self.dist_init_addr:
            if self.dist_init_addr.startswith("["):
                if self.dist_init_addr.endswith("]"):
                    host = self.dist_init_addr
                else:
                    host, _ = self.dist_init_addr.rsplit(":", 1)
            else:
                host = socket.gethostbyname(self.dist_init_addr.rsplit(":", 1)[0])
        else:
            host = self.bootstrap_host
            host = maybe_wrap_ipv6_address(host)

        bootstrap_server_url = f"{host}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_cp_size": self.attn_cp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "page_size": self.kv_args.page_size,
            "kv_cache_dtype": self.server_args.kv_cache_dtype,
            "load_balance_method": self.server_args.load_balance_method,
            # TCP-specific: data-plane port for direct KV transfers
            "tcp_port": self._tcp_port,
        }
        try:
            response = _requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug(
                    f"[TCPKVManager] registered with bootstrap (tcp_port={self._tcp_port})"
                )
            else:
                logger.error(
                    f"[TCPKVManager] bootstrap registration failed: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(f"[TCPKVManager] failed to register to bootstrap: {e}")

    # ------------------------------------------------------------------
    # TCP server (prefill side)
    # ------------------------------------------------------------------

    def _start_tcp_server(self) -> None:
        """Start a TCP server that accepts connections from decode workers."""
        self._tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_server.bind(("", 0))  # OS picks a free port
        self._tcp_port = self._tcp_server.getsockname()[1]
        self._tcp_server.listen(128)
        logger.info(f"[TCPKVManager] TCP KV server listening on port {self._tcp_port}")

        t = threading.Thread(target=self._tcp_accept_loop, daemon=True)
        t.start()

    def _tcp_accept_loop(self) -> None:
        """Accept loop: each accepted connection is handled in its own thread."""
        while True:
            try:
                conn, addr = self._tcp_server.accept()
                t = threading.Thread(
                    target=self._handle_connection, args=(conn, addr), daemon=True
                )
                t.start()
            except Exception as e:
                logger.error(f"[TCPKVManager] accept error: {e}")
                break

    def _handle_connection(
        self, conn: socket.socket, addr: Tuple[str, int]
    ) -> None:
        """Handle a single decode-side connection."""
        try:
            conn.settimeout(_RECV_TIMEOUT_S)
            # First message from receiver: the bootstrap_room (8-byte uint64)
            room_bytes = _recv_exact(conn, 8)
            room = struct.unpack("!Q", room_bytes)[0]
            logger.debug(f"[TCPKVManager] accepted connection for room={room} from {addr}")

            # The ZMQ message from the decode side may arrive slightly after the
            # TCP connection (ZMQ and TCP use independent channels).  Wait a
            # short time for the pending transfer to be registered.
            pending = None
            deadline = time.monotonic() + 5.0  # 5 s grace period
            while pending is None and time.monotonic() < deadline:
                with self._pending_lock:
                    pending = self._pending_transfers.get(room)
                if pending is None:
                    time.sleep(0.005)

            if pending is None:
                logger.warning(
                    f"[TCPKVManager] no pending transfer for room={room} after wait; closing"
                )
                conn.close()
                return

            pending.serve(conn)
        except Exception as e:
            logger.error(f"[TCPKVManager] connection handler error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Background prefill thread (processes ZMQ messages from receivers)
    # ------------------------------------------------------------------

    def _start_prefill_background_thread(self) -> None:
        t = threading.Thread(target=self._prefill_bg_loop, daemon=True)
        t.start()

    def _prefill_bg_loop(self) -> None:
        """
        Wait for ZMQ messages from decode workers containing dst KV indices.
        For each message, create a _PendingTransfer so that when the TCP
        connection arrives we know what to send.
        """
        import zmq

        while True:
            try:
                msg = self.server_socket.recv_multipart()
                self._process_zmq_msg(msg)
            except Exception as e:
                logger.error(f"[TCPKVManager] zmq recv error: {e}")
                time.sleep(0.1)

    def _process_zmq_msg(self, msg: List[bytes]) -> None:
        """Parse a ZMQ message from the receiver and create a _PendingTransfer."""
        if len(msg) < 6:
            logger.warning(f"[TCPKVManager] unexpected zmq msg length {len(msg)}")
            return
        try:
            room = int(msg[0].decode("ascii"))
            # msg[1]: receiver_ip, msg[2]: receiver_tcp_port (unused here; receiver connects to us)
            dst_kv_indices = np.frombuffer(msg[3], dtype=np.int32).copy()
            dst_aux_index = int(msg[4].decode("ascii")) if msg[4] else None
            required = int(msg[5].decode("ascii"))
        except Exception as e:
            logger.error(f"[TCPKVManager] failed to parse zmq msg: {e}")
            return

        pending = _PendingTransfer(
            room=room,
            kv_mgr=self,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
        )
        with self._pending_lock:
            self._pending_transfers[room] = pending

        self.update_status(room, KVPoll.WaitingForInput)


# ---------------------------------------------------------------------------
# _PendingTransfer – server-side state for one in-flight transfer
# ---------------------------------------------------------------------------


class _PendingTransfer:
    """
    Holds the destination KV indices sent by the decode worker.
    When the prefill sender calls ``ready()``, it populates the source KV
    indices; when the TCP connection arrives we do the actual copy+send.
    """

    def __init__(
        self,
        room: int,
        kv_mgr: TCPKVManager,
        dst_kv_indices: npt.NDArray[np.int32],
        dst_aux_index: Optional[int],
    ):
        self.room = room
        self.kv_mgr = kv_mgr
        self.dst_kv_indices = dst_kv_indices
        self.dst_aux_index = dst_aux_index

        # Set by TCPKVSender.send()
        self.src_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self._ready = threading.Event()
        self._done = threading.Event()

    def ready(self, src_kv_indices: npt.NDArray[np.int32]) -> None:
        self.src_kv_indices = src_kv_indices
        self._ready.set()

    def serve(self, conn: socket.socket) -> None:
        """
        Called from the TCP accept-loop thread when the decode worker connects.
        Waits for sender data, then streams KV caches layer by layer.
        """
        try:
            # Wait until the sender has called ready()
            self._ready.wait(timeout=_RECV_TIMEOUT_S)
            if self.src_kv_indices is None:
                logger.error(f"[_PendingTransfer] room={self.room} timed out waiting for src_kv_indices")
                self._finish(conn, success=False)
                return

            self._stream_kv(conn)
            self._finish(conn, success=True)
        except Exception as e:
            logger.error(f"[_PendingTransfer] serve error for room={self.room}: {e}")
            self._finish(conn, success=False)

    def _stream_kv(self, conn: socket.socket) -> None:
        """Copy each layer's KV pages from GPU → CPU → TCP socket."""
        kv_mgr = self.kv_mgr
        kv_args = kv_mgr.kv_args
        src_indices = self.src_kv_indices
        page_size = kv_args.page_size
        num_layers = len(kv_args.kv_data_ptrs) // 2  # k_layers + v_layers

        # Synchronise the GPU so we read fully-written KV caches
        torch.cuda.synchronize()

        for layer_id in range(num_layers):
            # -- Key buffer --
            k_ptr = kv_args.kv_data_ptrs[layer_id]
            k_item_len = kv_args.kv_item_lens[layer_id]
            k_data = _read_pages_from_gpu(k_ptr, k_item_len, src_indices, page_size)
            _send_layer_data(conn, layer_id * 2, k_data)

            # -- Value buffer --
            v_ptr = kv_args.kv_data_ptrs[num_layers + layer_id]
            v_item_len = kv_args.kv_item_lens[num_layers + layer_id]
            v_data = _read_pages_from_gpu(v_ptr, v_item_len, src_indices, page_size)
            _send_layer_data(conn, layer_id * 2 + 1, v_data)

        # -- Auxiliary / metadata buffer --
        if self.dst_aux_index is not None and kv_args.aux_data_ptrs:
            aux_ptr = kv_args.aux_data_ptrs[0]
            aux_item_len = kv_args.aux_item_lens[0]
            aux_data = _read_aux_from_cpu(aux_ptr, aux_item_len, self.dst_aux_index)
            _send_layer_data(conn, _MSG_DONE - 1, aux_data)  # layer_id = -2

        # Signal end of transfer
        _send_layer_data(conn, _MSG_DONE, b"")

    def _finish(self, conn: socket.socket, success: bool) -> None:
        status = KVPoll.Success if success else KVPoll.Failed
        self.kv_mgr.update_status(self.room, status)
        with self.kv_mgr._pending_lock:
            self.kv_mgr._pending_transfers.pop(self.room, None)
        self._done.set()


# ---------------------------------------------------------------------------
# Low-level GPU ↔ CPU transfer helpers (using CUDA driver API)
# ---------------------------------------------------------------------------
#
# We use the CUDA driver API (libcuda.so) directly to perform DtoH and HtoD
# memory copies from raw CUDA device pointers.  This avoids the undefined
# behaviour of using torch.frombuffer / ctypes.from_address on GPU pointers
# (which only work on CPU-addressable memory via UVA or pinned allocations).
#
# The driver is loaded once at module level via a lazy singleton.

_cuda_driver = None
_cuda_driver_lock = threading.Lock()


def _get_cuda_driver():
    """Lazily load libcuda.so for raw DtoH/HtoD memcpy operations."""
    global _cuda_driver
    if _cuda_driver is not None:
        return _cuda_driver
    with _cuda_driver_lock:
        if _cuda_driver is None:
            lib = ctypes.CDLL("libcuda.so.1", use_errno=True)
            # cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount) → CUresult
            lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
            lib.cuMemcpyDtoH_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_size_t,
            ]
            # cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) → CUresult
            lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
            lib.cuMemcpyHtoD_v2.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            _cuda_driver = lib
    return _cuda_driver


def _gpu_to_cpu(device_ptr: int, nbytes: int) -> bytearray:
    """Copy *nbytes* from a CUDA device pointer to a CPU bytearray."""
    lib = _get_cuda_driver()
    buf = bytearray(nbytes)
    c_buf = (ctypes.c_byte * nbytes).from_buffer(buf)
    rc = lib.cuMemcpyDtoH_v2(c_buf, ctypes.c_uint64(device_ptr), nbytes)
    if rc != 0:
        raise RuntimeError(f"cuMemcpyDtoH_v2 failed with error code {rc}")
    return buf


def _cpu_to_gpu(device_ptr: int, data: bytes) -> None:
    """Copy *data* bytes from CPU to a CUDA device pointer."""
    lib = _get_cuda_driver()
    nbytes = len(data)
    c_src = ctypes.c_char_p(data)
    rc = lib.cuMemcpyHtoD_v2(ctypes.c_uint64(device_ptr), c_src, nbytes)
    if rc != 0:
        raise RuntimeError(f"cuMemcpyHtoD_v2 failed with error code {rc}")


def _read_pages_from_gpu(
    base_ptr: int,
    item_len: int,
    page_indices: npt.NDArray[np.int32],
    page_size: int,
) -> bytes:
    """
    Read the KV pages at *page_indices* from a flat GPU CUDA buffer described
    by *base_ptr* (a raw CUDA device pointer) and *item_len* (bytes per page).
    Returns the concatenated raw bytes, CPU-resident.

    Uses the CUDA driver API (cuMemcpyDtoH_v2) for safe DtoH transfers.
    """
    num_pages = len(page_indices)
    if num_pages == 0:
        return b""

    # Synchronise so we read fully-written KV caches (caller may also do this)
    torch.cuda.synchronize()

    chunks = []
    for pi in page_indices:
        page_offset = int(pi) * item_len
        chunks.append(_gpu_to_cpu(base_ptr + page_offset, item_len))

    return b"".join(bytes(c) for c in chunks)


def _read_aux_from_cpu(base_ptr: int, item_len: int, index: int) -> bytes:
    """Read auxiliary metadata at *index* from a CPU buffer."""
    addr = base_ptr + item_len * index
    buf = (ctypes.c_byte * item_len).from_address(addr)
    return bytes(buf)


# ---------------------------------------------------------------------------
# TCPKVSender  (prefill side)
# ---------------------------------------------------------------------------



class TCPKVSender(CommonKVSender):
    """
    Sends KV caches from the prefill node to the decode node over TCP.

    The sender cooperates with TCPKVManager:
    1.  init() stores the number of pages to transfer.
    2.  send() marks the _PendingTransfer as ready (provides src indices).
        The actual TCP streaming happens in the manager's accept-loop thread.
    3.  poll() checks the transfer status via the manager's status table.

    Layer-wise pipeline send
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Applications may call ``send_layer(layer_id, src_indices)`` directly from
    within the model's attention forward pass (after each layer writes its KV
    cache) to overlap the TCP transfer with subsequent layer computation.
    When this method is used, ``send()`` should NOT be called.
    """

    def __init__(
        self,
        mgr: TCPKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.bootstrap_server_url = bootstrap_addr
        self.aux_index: Optional[int] = None
        self.num_kv_indices: Optional[int] = None
        self._layer_sent: bool = False

        if getattr(self.kv_mgr, "is_dummy_cp_rank", False):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None) -> None:
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Mark the pending transfer as ready to be served.
        The actual TCP streaming is handled by the TCPKVManager's accept-loop.
        """
        with self.kv_mgr._pending_lock:
            pending = self.kv_mgr._pending_transfers.get(self.bootstrap_room)

        if pending is None:
            # The receiver hasn't connected yet; wait briefly
            deadline = time.monotonic() + _RECV_TIMEOUT_S
            while pending is None and time.monotonic() < deadline:
                time.sleep(0.01)
                with self.kv_mgr._pending_lock:
                    pending = self.kv_mgr._pending_transfers.get(self.bootstrap_room)

        if pending is None:
            logger.error(
                f"[TCPKVSender] room={self.bootstrap_room}: no pending transfer found"
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)
        pending.ready(kv_indices)

    def send_layer(self, layer_id: int, src_indices: npt.NDArray[np.int32]) -> None:
        """
        Layer-wise pipeline send: called from the attention forward pass after
        each layer's KV cache has been written.  Transfers are sent immediately
        over the pre-established TCP connection without waiting for all layers.

        This method is optional and is only effective when the backend has been
        set up for layer-wise pipelining (``TCPKVManager._layer_conn_map`` exists).
        """
        conn_map = getattr(self.kv_mgr, "_layer_conn_map", None)
        if conn_map is None:
            return
        conn = conn_map.get(self.bootstrap_room)
        if conn is None:
            return

        kv_args = self.kv_mgr.kv_args
        num_layers = len(kv_args.kv_data_ptrs) // 2
        page_size = kv_args.page_size

        k_ptr = kv_args.kv_data_ptrs[layer_id]
        k_item_len = kv_args.kv_item_lens[layer_id]
        v_ptr = kv_args.kv_data_ptrs[num_layers + layer_id]
        v_item_len = kv_args.kv_item_lens[num_layers + layer_id]

        torch.cuda.synchronize()
        k_data = _read_pages_from_gpu(k_ptr, k_item_len, src_indices, page_size)
        v_data = _read_pages_from_gpu(v_ptr, v_item_len, src_indices, page_size)

        _send_layer_data(conn, layer_id * 2, k_data)
        _send_layer_data(conn, layer_id * 2 + 1, v_data)
        self._layer_sent = True

    def poll(self) -> KVPoll:
        return KVPoll(self.kv_mgr.check_status(self.bootstrap_room))

    def failure_exception(self) -> None:
        raise RuntimeError(
            f"TCPKVSender: transfer failed for room={self.bootstrap_room}"
        )


# ---------------------------------------------------------------------------
# TCPKVReceiver  (decode side)
# ---------------------------------------------------------------------------


class TCPKVReceiver(CommonKVReceiver):
    """
    Receives KV caches on the decode node.

    Lifecycle:
    1. __init__: fetches prefill server metadata (via bootstrap HTTP) and
       sets up ZMQ endpoint info.
    2. init(): sends the dst_kv_indices to the prefill server (via ZMQ),
       connects to the prefill's TCP server, and starts receiving data in a
       background thread.
    3. poll(): returns Success once all layers are received.
    """

    def __init__(
        self,
        mgr: TCPKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        # Use CommonKVReceiver init (handles bootstrap info fetching)
        super().__init__(
            mgr=mgr,
            bootstrap_addr=bootstrap_addr,
            bootstrap_room=bootstrap_room,
            prefill_dp_rank=prefill_dp_rank,
        )
        self._transfer_done = threading.Event()
        self._transfer_ok = True

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Send dst KV indices to all prefill ranks (via ZMQ) and start the TCP
        receive loop in a background thread.
        """
        self._kv_indices = kv_indices
        self._aux_index = aux_index

        if self.bootstrap_infos is None:
            return

        # Notify each prefill rank via ZMQ
        for binfo in self.bootstrap_infos:
            if binfo.get("is_dummy", False):
                continue
            self._send_zmq_descriptor(binfo, kv_indices, aux_index)

        # Connect to the prefill's TCP server and receive data
        t = threading.Thread(target=self._recv_loop, args=(kv_indices, aux_index), daemon=True)
        t.start()

    def _send_zmq_descriptor(
        self,
        bootstrap_info: dict,
        dst_kv_indices: npt.NDArray[np.int32],
        dst_aux_index: Optional[int],
    ) -> None:
        """
        Send a ZMQ message to the prefill rank's ZMQ PULL socket containing
        the information it needs to push data to us.
        """
        sock, lock = self.__class__._connect_to_bootstrap_server(bootstrap_info)
        room_str = str(self.bootstrap_room).encode("ascii")
        local_ip = self.kv_mgr.local_ip.encode("ascii")
        # tcp_port of OUR tcp listener – we don't have one; receiver connects to prefill
        tcp_port_str = b"0"
        idx_bytes = dst_kv_indices.astype(np.int32).tobytes()
        aux_str = str(dst_aux_index if dst_aux_index is not None else 0).encode("ascii")
        required_str = b"1"

        with lock:
            sock.send_multipart([
                room_str,
                local_ip,
                tcp_port_str,
                idx_bytes,
                aux_str,
                required_str,
            ])

    def _recv_loop(
        self,
        dst_kv_indices: npt.NDArray[np.int32],
        dst_aux_index: Optional[int],
    ) -> None:
        """
        Connect to the prefill's TCP server, send our room id, then receive
        KV cache layer by layer and write each layer to the local GPU pool.
        """
        try:
            # Give the prefill side a moment to register the pending transfer
            time.sleep(0.05)

            binfo = None
            for info in (self.bootstrap_infos or []):
                if not info.get("is_dummy", False):
                    binfo = info
                    break

            if binfo is None:
                logger.error(f"[TCPKVReceiver] room={self.bootstrap_room}: no bootstrap info")
                self._transfer_ok = False
                return

            prefill_ip = binfo["rank_ip"]
            prefill_tcp_port = binfo.get("tcp_port", binfo.get("rank_port"))

            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(_RECV_TIMEOUT_S)
            conn.connect((prefill_ip, int(prefill_tcp_port)))

            # Introduce ourselves
            conn.sendall(struct.pack("!Q", self.bootstrap_room))

            self._recv_kv(conn, dst_kv_indices, dst_aux_index)
            conn.close()
            self._transfer_ok = True
        except Exception as e:
            logger.error(f"[TCPKVReceiver] recv_loop error: {e}")
            self._transfer_ok = False
        finally:
            self._transfer_done.set()
            status = KVPoll.Success if self._transfer_ok else KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, status)

    def _recv_kv(
        self,
        conn: socket.socket,
        dst_kv_indices: npt.NDArray[np.int32],
        dst_aux_index: Optional[int],
    ) -> None:
        """Receive layer-by-layer KV data and write to GPU pool."""
        kv_args = self.kv_mgr.kv_args
        num_layers = len(kv_args.kv_data_ptrs) // 2
        page_size = kv_args.page_size
        expected_msgs = num_layers * 2  # K + V per layer
        received = 0

        while True:
            layer_id, data = _recv_layer_data(conn)

            if layer_id == _MSG_DONE:
                break

            if layer_id == _MSG_DONE - 1:
                # Auxiliary metadata
                if dst_aux_index is not None and kv_args.aux_data_ptrs and data:
                    _write_aux_to_cpu(
                        kv_args.aux_data_ptrs[0],
                        kv_args.aux_item_lens[0],
                        dst_aux_index,
                        data,
                    )
                continue

            # Determine whether this is K or V
            if layer_id % 2 == 0:
                actual_layer = layer_id // 2
                base_ptr = kv_args.kv_data_ptrs[actual_layer]
                item_len = kv_args.kv_item_lens[actual_layer]
            else:
                actual_layer = layer_id // 2
                base_ptr = kv_args.kv_data_ptrs[num_layers + actual_layer]
                item_len = kv_args.kv_item_lens[num_layers + actual_layer]

            _write_pages_to_gpu(base_ptr, item_len, dst_kv_indices, page_size, data)
            received += 1

    def poll(self) -> KVPoll:
        if not self._transfer_done.is_set():
            return KVPoll.Transferring
        return KVPoll.Success if self._transfer_ok else KVPoll.Failed

    def failure_exception(self) -> None:
        raise RuntimeError(
            f"TCPKVReceiver: transfer failed for room={self.bootstrap_room}"
        )

    def _register_kv_args(self) -> None:
        """No RDMA registration needed for TCP."""
        pass


# ---------------------------------------------------------------------------
# Low-level GPU write helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Low-level GPU write helpers
# ---------------------------------------------------------------------------


def _write_pages_to_gpu(
    base_ptr: int,
    item_len: int,
    page_indices: npt.NDArray[np.int32],
    page_size: int,
    data: bytes,
) -> None:
    """
    Write received raw bytes into the GPU KV pool at the given page indices.

    Uses the CUDA driver API (cuMemcpyHtoD_v2) for safe HtoD transfers.
    """
    if not data or len(page_indices) == 0:
        return

    offset = 0
    for pi in page_indices:
        page_offset = int(pi) * item_len
        chunk = data[offset : offset + item_len]
        _cpu_to_gpu(base_ptr + page_offset, chunk)
        offset += item_len


def _write_aux_to_cpu(
    base_ptr: int,
    item_len: int,
    index: int,
    data: bytes,
) -> None:
    """Write auxiliary metadata bytes to a CPU buffer."""
    dst = (ctypes.c_byte * item_len).from_address(base_ptr + item_len * index)
    ctypes.memmove(dst, data, min(len(data), item_len))


# ---------------------------------------------------------------------------
# TCPKVBootstrapServer  (reuses the HTTP bootstrap, adds tcp_port field)
# ---------------------------------------------------------------------------


class TCPKVBootstrapServer(CommonKVBootstrapServer):
    """
    Bootstrap server for the TCP backend.

    Extends CommonKVBootstrapServer to store and serve the prefill TCP port
    alongside the ZMQ port so that decode workers know where to connect.

    The bootstrap route PUT payload now includes an optional ``tcp_port``
    field.  The route GET response includes a ``tcp_port`` field in addition
    to the standard ``rank_ip`` / ``rank_port`` fields.
    """

    def __init__(self, host: str, port: int):
        # Map: (dp_group, cp_rank, tp_rank, pp_rank) → tcp_port
        self._tcp_port_table: Dict[Tuple[int, int, int, int], int] = {}
        super().__init__(host, port)

    def _setup_routes(self):
        """Extend the parent routes to also handle the GET with tcp_port."""
        super()._setup_routes()
        # The existing /route GET is replaced by our augmented handler
        self.app.router.add_route("GET", "/route", self._handle_tcp_route_get)

    async def _handle_route_put(self, request):
        """Extend the parent PUT handler to also store the tcp_port."""
        data = await request.json()
        tcp_port = int(data.get("tcp_port", 0))

        # Let the parent handler do the heavy lifting
        response = await super()._handle_route_put(request)

        if response.status == 200 and tcp_port > 0:
            attn_dp_rank = int(data.get("attn_dp_rank", 0))
            system_dp_rank = int(data.get("system_dp_rank", 1))
            system_dp_size = int(data.get("system_dp_size", 1))
            attn_cp_rank = int(data.get("attn_cp_rank", 0))
            attn_tp_rank = int(data.get("attn_tp_rank", 0))
            pp_rank = int(data.get("pp_rank", 0))
            dp_group = attn_dp_rank if system_dp_size == 1 else system_dp_rank
            async with self.lock:
                self._tcp_port_table[
                    (dp_group, attn_cp_rank, attn_tp_rank, pp_rank)
                ] = tcp_port
        return response

    async def _handle_tcp_route_get(self, request):
        """GET /route – same as parent but enriches the response with tcp_port."""
        from aiohttp import web

        prefill_dp_rank = request.query.get("prefill_dp_rank")
        prefill_cp_rank = request.query.get("prefill_cp_rank")
        target_tp_rank = request.query.get("target_tp_rank")
        target_pp_rank = request.query.get("target_pp_rank")

        # Delegate to parent for validation and standard response
        response = await self._handle_route_get(request)

        if response.status != 200:
            return response

        # For server-info queries (-1 params) no tcp_port is needed
        if (
            prefill_dp_rank is None
            or int(prefill_dp_rank) == -1
        ):
            return response

        # Inject tcp_port into the response payload
        key = (
            int(prefill_dp_rank),
            int(prefill_cp_rank),
            int(target_tp_rank),
            int(target_pp_rank),
        )
        async with self.lock:
            tcp_port = self._tcp_port_table.get(key, 0)

        # Re-build the response JSON with the extra field
        try:
            data = await response.json()
        except Exception:
            import json as _json
            data = _json.loads(response.body)
        data["tcp_port"] = tcp_port
        return web.json_response(data, status=200)

