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

# How long to wait for a TCP connection/receive operation (seconds).
_RECV_TIMEOUT_S = 120

# How long send_layer() waits for the TCP connection from the decode side to
# arrive.  The decode receiver sleeps ~_TCP_CONNECT_DELAY_S before connecting
# (to allow the ZMQ descriptor to be processed on the prefill side first).
# We use a conservative timeout to handle slow/loaded systems; most connections
# arrive within 100 ms in practice.
_TCP_CONN_WAIT_S = 5.0

# Initial sleep on the decode side before connecting the TCP data socket.
# This gives the prefill side time to process the ZMQ descriptor and create a
# _PendingTransfer before the TCP connection arrives.
_TCP_CONNECT_DELAY_S = 0.05

# How long TCPKVSender waits for the decode side to register its KV indices
# via ZMQ before declaring a failure.  Shorter than _RECV_TIMEOUT_S because
# the ZMQ message should arrive almost immediately after the HTTP response.
_SENDER_WAIT_S = 10.0

# ZMQ poller timeout (milliseconds).  Short enough to keep the loop responsive
# without busy-looping; long enough not to waste CPU.
_ZMQ_POLL_TIMEOUT_MS = 1000

# How long (seconds) the TCP receiver thread waits for the first byte before
# considering the connection dead.  Defined once above as _RECV_TIMEOUT_S.

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
            # TCP connection (ZMQ and TCP use independent channels).  Wait up to
            # _TCP_CONN_WAIT_S for the pending transfer to be registered.
            pending = None
            deadline = time.monotonic() + _TCP_CONN_WAIT_S
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

        Uses a ZMQ poller with a 1-second timeout so the thread doesn't block
        forever if the socket becomes unhealthy.
        """
        import zmq

        poller = zmq.Poller()
        poller.register(self.server_socket, zmq.POLLIN)

        while True:
            try:
                socks = dict(poller.poll(timeout=_ZMQ_POLL_TIMEOUT_MS))
                if self.server_socket in socks:
                    msg = self.server_socket.recv_multipart(zmq.NOBLOCK)
                    self._process_zmq_msg(msg)
            except zmq.Again:
                pass  # No message yet; loop back
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

    There are two operating modes:

    **Batch mode** (default):
      The prefill sender calls ``ready(kv_indices)`` after the full forward
      pass completes.  ``serve()`` waits for ``ready()`` then reads all KV
      pages from GPU and streams them layer-by-layer over the TCP socket.

    **Pipeline mode** (when ``layer_kv_send_fn`` is active):
      The attention backend calls ``send_layer(layer_id, src_indices)`` after
      each transformer layer writes its KV cache.  Each call sends that layer's
      data immediately over the already-established TCP socket so that the
      transfer overlaps with the next layer's computation.  After all layers
      are done, the sender calls ``done_pipeline(src_kv_indices)`` which sends
      the auxiliary metadata and the EOF marker and signals completion.

    In both modes, ``serve()`` is the TCP-accept-loop thread that owns the
    connection lifecycle and marks the transfer success/failure.
    """

    def __init__(
        self,
        room: int,
        kv_mgr: "TCPKVManager",
        dst_kv_indices: npt.NDArray[np.int32],
        dst_aux_index: Optional[int],
    ):
        self.room = room
        self.kv_mgr = kv_mgr
        self.dst_kv_indices = dst_kv_indices
        self.dst_aux_index = dst_aux_index

        # Set by TCPKVSender.send() (batch mode) or done_pipeline() (pipeline mode).
        self.src_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self._ready = threading.Event()
        self._done = threading.Event()

        # TCP connection – set by serve() so that send_layer() can write to it.
        self._conn: Optional[socket.socket] = None
        self._conn_event = threading.Event()
        # Pipeline mode flag – set by done_pipeline().
        self._pipeline_mode: bool = False

    # ------------------------------------------------------------------
    # Batch-mode API (called by TCPKVSender.send())
    # ------------------------------------------------------------------

    def ready(self, src_kv_indices: npt.NDArray[np.int32]) -> None:
        """Signal that all KV data is in the GPU pool (batch mode)."""
        self.src_kv_indices = src_kv_indices
        self._ready.set()

    # ------------------------------------------------------------------
    # Pipeline-mode API (called by TCPKVSender.send_layer() / done_pipeline())
    # ------------------------------------------------------------------

    def wait_for_conn(self, timeout: float) -> bool:
        """Wait until the TCP connection is available. Returns True if connected."""
        return self._conn_event.wait(timeout=timeout)

    def done_pipeline(self, src_kv_indices: npt.NDArray[np.int32]) -> None:
        """
        Signal that all layers have been sent via pipeline and the transfer
        is complete.  ``serve()`` will send aux + EOF and mark success.
        """
        self.src_kv_indices = src_kv_indices
        self._pipeline_mode = True
        self._ready.set()

    # ------------------------------------------------------------------
    # TCP-server-side: called from accept-loop thread
    # ------------------------------------------------------------------

    def serve(self, conn: socket.socket) -> None:
        """
        Called from the TCP accept-loop when the decode worker connects.

        1. Makes the connection available to ``send_layer()`` (pipeline mode).
        2. Waits for the sender to signal readiness (batch or pipeline).
        3. Streams data and signals completion.
        """
        # Store the connection so that send_layer() can write to it.
        self._conn = conn
        self._conn_event.set()

        try:
            # Wait for the sender to complete (batch: send(), pipeline: done_pipeline()).
            if not self._ready.wait(timeout=_RECV_TIMEOUT_S):
                logger.error(
                    f"[_PendingTransfer] room={self.room} timed out waiting for sender"
                )
                self._finish(conn, success=False)
                return

            if self._pipeline_mode:
                # Pipeline mode: all KV layers were already sent by send_layer().
                # Only the aux metadata and the EOF marker remain.
                self._send_tail(conn)
            else:
                # Batch mode: read all KV data from GPU and send now.
                if self.src_kv_indices is None:
                    logger.error(
                        f"[_PendingTransfer] room={self.room}: src_kv_indices not set"
                    )
                    self._finish(conn, success=False)
                    return
                self._stream_kv(conn)

            self._finish(conn, success=True)
        except Exception as e:
            logger.error(f"[_PendingTransfer] serve error for room={self.room}: {e}")
            self._finish(conn, success=False)

    def _stream_kv(self, conn: socket.socket) -> None:
        """Copy each layer's KV pages from GPU → CPU → TCP socket (batch mode)."""
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

        self._send_tail(conn)

    def _send_tail(self, conn: socket.socket) -> None:
        """Send auxiliary metadata (if any) and the EOF sentinel."""
        kv_args = self.kv_mgr.kv_args

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
            try:
                lib = ctypes.CDLL("libcuda.so.1", use_errno=True)
            except OSError as e:
                raise RuntimeError(
                    "Failed to load the CUDA driver library (libcuda.so.1). "
                    "The TCP KV transfer backend requires a CUDA driver installation. "
                    f"Original error: {e}"
                ) from e
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
        Complete the transfer after the forward pass.

        * **Pipeline mode** (``self._layer_sent`` is True): all KV layers were
          already sent by ``send_layer()``.  Signal ``done_pipeline()`` so that
          ``serve()`` sends aux metadata + EOF and marks the transfer done.
        * **Batch mode** (``self._layer_sent`` is False): call ``ready()`` so
          that ``serve()`` reads all KV data from GPU and streams everything now.
        """
        with self.kv_mgr._pending_lock:
            pending = self.kv_mgr._pending_transfers.get(self.bootstrap_room)

        if pending is None:
            # The receiver hasn't registered its KV indices via ZMQ yet.
            # Wait up to _SENDER_WAIT_S (much shorter than _RECV_TIMEOUT_S)
            # because the ZMQ message should arrive almost immediately.
            deadline = time.monotonic() + _SENDER_WAIT_S
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

        if self._layer_sent:
            # Pipeline mode: all KV data already sent layer-by-layer.
            # Signal serve() to send aux + EOF and finish.
            pending.done_pipeline(kv_indices)
        else:
            # Batch mode: let serve() read all GPU data and stream it now.
            pending.ready(kv_indices)

    def send_layer(self, layer_id: int, src_indices: npt.NDArray[np.int32]) -> None:
        """
        Layer-wise pipeline send: called from the attention forward pass after
        each layer's KV cache has been written to the GPU pool.

        Looks up the ``_PendingTransfer`` for this room (created when the ZMQ
        descriptor arrives from the decode side) and waits briefly for the TCP
        connection to be established.  Once the connection is available, sends
        the current layer's KV data (K + V) immediately over the socket so that
        the transfer overlaps with the computation of subsequent layers.

        If the pending transfer or the TCP connection is not yet available when
        called, the call is a no-op and ``send()`` will handle the transfer in
        batch mode instead.
        """
        with self.kv_mgr._pending_lock:
            pending = self.kv_mgr._pending_transfers.get(self.bootstrap_room)

        if pending is None:
            # ZMQ message hasn't arrived yet – can't do pipeline for this layer.
            return

        # Wait briefly for the TCP connection to be established.
        # The decode receiver delays _TCP_CONNECT_DELAY_S before connecting,
        # so the connection typically arrives within ~100 ms of the ZMQ message.
        # We use the larger _TCP_CONN_WAIT_S timeout to handle slow or loaded
        # systems where the connection setup may take significantly longer.
        if not pending.wait_for_conn(timeout=_TCP_CONN_WAIT_S):
            # TCP connection not yet available; skip pipeline for this layer.
            # send() will fall back to batch mode after the forward pass.
            return

        if pending._conn is None:
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

        try:
            _send_layer_data(pending._conn, layer_id * 2, k_data)
            _send_layer_data(pending._conn, layer_id * 2 + 1, v_data)
            self._layer_sent = True
        except Exception as e:
            logger.warning(
                f"[TCPKVSender] send_layer layer={layer_id} failed for "
                f"room={self.bootstrap_room}: {e}; will retry in batch mode"
            )
            # Reset so send() falls back to batch mode for this request.
            self._layer_sent = False

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

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
            # Sleep briefly so the prefill side has time to process the ZMQ descriptor
            # and create a _PendingTransfer before the TCP connection arrives.
            time.sleep(_TCP_CONNECT_DELAY_S)

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
            tcp_port_raw = binfo.get("tcp_port")
            if not tcp_port_raw:
                # Fallback: the bootstrap server did not return a tcp_port field
                # (e.g. an older server or a non-TCP backend bootstrap server).
                # Falling back to rank_port (ZMQ port) will almost certainly
                # fail because ZMQ and TCP use different sockets.
                tcp_port_raw = binfo.get("rank_port")
                logger.warning(
                    f"[TCPKVReceiver] room={self.bootstrap_room}: tcp_port not in "
                    f"bootstrap info; falling back to rank_port={tcp_port_raw}. "
                    "Ensure the prefill server uses the TCP backend."
                )
            prefill_tcp_port = tcp_port_raw

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

    The bootstrap route PUT payload accepts an optional ``tcp_port`` field.
    The route GET response includes a ``tcp_port`` field in addition to the
    standard ``rank_ip`` / ``rank_port`` fields so that decode workers know
    which port to open a TCP connection to.

    Design notes
    ------------
    The parent class registers a wildcard ``*`` handler for ``/route`` that
    dispatches to ``_handle_route_put`` and ``_handle_route_get`` based on the
    HTTP method.  We override those two virtual methods directly so that Python's
    MRO ensures our versions are called.  No ``_setup_routes`` override is
    needed (which would conflict with the parent's wildcard registration).
    """

    def __init__(self, host: str, port: int):
        # Map: (dp_group, cp_rank, tp_rank, pp_rank) → tcp_port
        self._tcp_port_table: Dict[Tuple[int, int, int, int], int] = {}
        super().__init__(host, port)

    async def _handle_route_put(self, request):
        """
        Extend the parent PUT handler to also store the ``tcp_port`` field.

        We call ``await request.json()`` here first to extract ``tcp_port``.
        Then we delegate to the parent, which calls ``await request.json()``
        again internally.  This is safe because aiohttp caches the raw body
        bytes in ``request._read_bytes`` on the first read; subsequent calls
        return the same cached data without re-reading the network stream.
        """
        from aiohttp import web

        data = await request.json()
        tcp_port = int(data.get("tcp_port", 0))

        # Let the parent handler do the heavy lifting (registration, validation).
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

    async def _handle_route_get(self, request):
        """
        Extend the parent GET handler to inject ``tcp_port`` into the JSON
        response so decode workers know which TCP port to connect to.
        """
        import json as _json

        from aiohttp import web

        prefill_dp_rank = request.query.get("prefill_dp_rank")
        prefill_cp_rank = request.query.get("prefill_cp_rank")
        target_tp_rank = request.query.get("target_tp_rank")
        target_pp_rank = request.query.get("target_pp_rank")

        # Delegate to the parent for all validation and standard response.
        response = await super()._handle_route_get(request)

        if response.status != 200:
            return response

        # For server-info queries (all params == -1) no tcp_port is needed.
        if prefill_dp_rank is None or int(prefill_dp_rank) == -1:
            return response

        key = (
            int(prefill_dp_rank),
            int(prefill_cp_rank),
            int(target_tp_rank),
            int(target_pp_rank),
        )
        async with self.lock:
            tcp_port = self._tcp_port_table.get(key, 0)

        # Rebuild the JSON payload with the extra field.  ``web.Response.text``
        # holds the serialised JSON string returned by the parent.
        data = _json.loads(response.text)
        data["tcp_port"] = tcp_port
        return web.json_response(data, status=200)

