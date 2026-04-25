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
import os
import queue
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
import struct as _struct

from sglang.srt.disaggregation.tcp.transfer_quant import (
    _DTYPE_CODE_BF16,
    _DTYPE_CODE_FP16,
    dequantize_from_transfer,
    dequantize_on_gpu,
    quantize_for_transfer,
    quantize_on_gpu,
)

# Wire-format header for quantized payloads (from transfer_quant.py).
# Must be captured before conn.py's own _HEADER_FMT overrides it.
_WIRE_HEADER_FMT = "!iiiI"
_WIRE_HEADER_SIZE = 16

# Lazily import Triton kernels; fall back to PyTorch if unavailable.
try:
    from sglang.srt.disaggregation.tcp.transfer_quant_triton import (
        dequantize_4bit_on_gpu as _triton_dequant_4bit,
        quantize_4bit_on_gpu as _triton_quant_4bit,
    )
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)

# Set to True to enable per-layer tensor diagnostics (sum, absmax, nonzero)
# on sender and receiver for layer 0.  Useful for debugging quant correctness
# but adds ~1ms per checked layer due to GPU sync + reduction.
_DEBUG_QUANT = False

_PIPELINE_SEND = os.environ.get("SGLANG_TCP_PIPELINE_SEND", "1") != "0"

# ---------------------------------------------------------------------------
# Triton-accelerated 4-bit helpers (drop-in for quantize_on_gpu / dequantize_on_gpu)
# ---------------------------------------------------------------------------

def _triton_quant_4bit_to_bytes(tensor: torch.Tensor, group_size: int = 64) -> bytes:
    """Quantize a GPU tensor with Triton and return wire-format bytes."""
    packed, scales = _triton_quant_4bit(tensor, group_size=group_size)
    torch.cuda.synchronize()
    is_bf16 = tensor.dtype == torch.bfloat16
    dtype_code = _DTYPE_CODE_BF16 if is_bf16 else _DTYPE_CODE_FP16
    num_elements = tensor.numel()
    header = _struct.pack(_WIRE_HEADER_FMT, 4, group_size, num_elements, dtype_code)
    return header + scales.numpy().tobytes() + packed.numpy().tobytes()


def _triton_dequant_4bit_from_bytes(packed: bytes, device: torch.device, group_size: int = 64) -> torch.Tensor:
    """Dequantize wire-format bytes with Triton and return a GPU tensor."""
    nbits, gs, num_elements, dtype_code = _struct.unpack(_WIRE_HEADER_FMT, packed[:_WIRE_HEADER_SIZE])
    is_bf16 = dtype_code == _DTYPE_CODE_BF16
    target_dtype = torch.bfloat16 if is_bf16 else torch.float16
    remainder = num_elements % gs
    pad = (gs - remainder) if remainder else 0
    num_groups = (num_elements + pad) // gs
    import numpy as _np
    scales_offset = 16
    scales_np = _np.frombuffer(packed[scales_offset : scales_offset + num_groups * 2], dtype=_np.float16).copy()
    packed_np = _np.frombuffer(packed[scales_offset + num_groups * 2 : scales_offset + num_groups * 2 + (num_elements + 1) // 2], dtype=_np.uint8).copy()
    return _triton_dequant_4bit(packed_np, scales_np, num_elements, group_size=gs, device=device, dtype=target_dtype)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

# Message framing: every data message is preceded by an 8-byte header
# [4 bytes: layer_id (int32)] [4 bytes: payload_length_bytes (uint32)]
_HEADER_FMT = "!iI"  # network byte order: signed int32 + unsigned int32
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Special layer_id value that signals "all done"
_MSG_DONE = -1

# Bit flag OR'd into layer_id to indicate the payload is quantized.
# The receiver strips this flag and dequantizes before writing to GPU.
_MSG_QUANT_FLAG = 0x40000000

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

            logger.warning(
                f"[_handle_connection] room={room} from {addr} → pending found, calling serve()"
            )
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

        # Transfer quantization: callable (layer_id) → nbits or None.
        # Set by TCPKVSender.ready() so _stream_kv can quantize per-layer.
        self._get_quant_bits: Optional[object] = None  # Callable[[int], Optional[int]]

        # Prefill-side metadata buffer index — set by ready()/done_pipeline().
        # _send_tail() must read aux data from this index (where set_buf() wrote),
        # NOT from dst_aux_index (which is the decode side's allocation).
        self.src_aux_index: Optional[int] = None

        # TCP connection – set by serve() so that send_layer() can write to it.
        self._conn: Optional[socket.socket] = None
        self._conn_event = threading.Event()
        # Pipeline mode flag – set by done_pipeline().
        self._pipeline_mode: bool = False

    # ------------------------------------------------------------------
    # Batch-mode API (called by TCPKVSender.send())
    # ------------------------------------------------------------------

    def ready(self, src_kv_indices: npt.NDArray[np.int32], get_quant_bits=None, src_aux_index=None) -> None:
        """Signal that all KV data is in the GPU pool (batch mode)."""
        self.src_kv_indices = src_kv_indices
        self._get_quant_bits = get_quant_bits
        self.src_aux_index = src_aux_index
        self._ready.set()

    # ------------------------------------------------------------------
    # Pipeline-mode API (called by TCPKVSender.send_layer() / done_pipeline())
    # ------------------------------------------------------------------

    def wait_for_conn(self, timeout: float) -> bool:
        """Wait until the TCP connection is available. Returns True if connected."""
        return self._conn_event.wait(timeout=timeout)

    def done_pipeline(self, src_kv_indices: npt.NDArray[np.int32], src_aux_index=None) -> None:
        """
        Signal that all layers have been sent via pipeline and the transfer
        is complete.  ``serve()`` will send aux + EOF and mark success.
        """
        self.src_kv_indices = src_kv_indices
        self.src_aux_index = src_aux_index
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
        # Ensure this thread has a valid CUDA context for GPU reads.
        torch.cuda.set_device(self.kv_mgr.kv_args.gpu_id)

        # Store the connection so that send_layer() can write to it.
        self._conn = conn
        self._conn_event.set()

        try:
            # Wait for the sender to complete (batch: send(), pipeline: done_pipeline()).
            logger.warning(
                f"[_PendingTransfer.serve] room={self.room} waiting for _ready..."
            )
            if not self._ready.wait(timeout=_RECV_TIMEOUT_S):
                logger.error(
                    f"[_PendingTransfer] room={self.room} timed out waiting for sender"
                )
                self._finish(conn, success=False)
                return

            logger.warning(
                f"[_PendingTransfer.serve] room={self.room} _ready set! pipeline={self._pipeline_mode}"
            )

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
        if _PIPELINE_SEND:
            self._stream_kv_pipelined(conn)
        else:
            self._stream_kv_sequential(conn)

    def _stream_kv_sequential(self, conn: socket.socket) -> None:
        """Copy each layer's KV pages from GPU → TCP socket (batch mode).

        When quantization is enabled, the flow is:
          GPU pages (scattered) → DtoD → GPU tensor → GPU quantize → DtoH (small) → TCP
        Without quantization the legacy path is used:
          GPU pages (scattered) → DtoH → TCP
        """
        kv_mgr = self.kv_mgr
        kv_args = kv_mgr.kv_args
        src_indices = self.src_kv_indices
        page_size = kv_args.page_size
        num_layers = len(kv_args.kv_data_ptrs) // 2  # k_layers + v_layers

        # Transfer quantization setup
        get_quant_bits = self._get_quant_bits
        kv_dtype = getattr(kv_mgr.server_args, "kv_cache_dtype", "auto")
        is_bf16 = kv_dtype in ("auto", "bf16", "bfloat16")
        torch_dtype = torch.bfloat16 if is_bf16 else torch.float16

        logger.warning(
            f"[_stream_kv] room={self.room} starting, layers={num_layers}, "
            f"src_indices={src_indices}, page_size={page_size}, "
            f"quant={'gpu' if get_quant_bits else 'off'}"
        )

        # Synchronise the GPU so we read fully-written KV caches
        torch.cuda.synchronize()

        logger.warning(f"[_stream_kv] room={self.room} cuda sync done, sending layers...")

        t_stream_start = time.perf_counter()
        total_quant_ms = 0.0
        total_send_ms = 0.0
        total_gather_ms = 0.0
        total_transfer_bytes = 0

        for layer_id in range(num_layers):
            if layer_id % 7 == 0:
                logger.warning(f"[_stream_kv] room={self.room} sending layer {layer_id}/{num_layers}")

            k_ptr = kv_args.kv_data_ptrs[layer_id]
            k_item_len = kv_args.kv_item_lens[layer_id]
            v_ptr = kv_args.kv_data_ptrs[num_layers + layer_id]
            v_item_len = kv_args.kv_item_lens[num_layers + layer_id]

            nbits = get_quant_bits(layer_id) if get_quant_bits else None

            t_gather = time.perf_counter()
            if nbits is not None:
                # GPU path: gather → quantize on GPU → DtoH compressed bytes
                k_tensor = _read_pages_as_gpu_tensor(
                    k_ptr, k_item_len, src_indices, page_size, torch_dtype
                )
                v_tensor = _read_pages_as_gpu_tensor(
                    v_ptr, v_item_len, src_indices, page_size, torch_dtype
                )
                # Sync: DtoD copies above use the CUDA driver API (NULL stream),
                # but quantize_on_gpu uses PyTorch ops (current stream).  Without
                # this barrier the PyTorch kernels may read partially-copied data.
                torch.cuda.synchronize()
                if _DEBUG_QUANT and layer_id == 0:
                    logger.warning(
                        f"[_stream_kv DEBUG] room={self.room} layer=0 "
                        f"k_sum={k_tensor.float().sum().item():.4f} "
                        f"v_sum={v_tensor.float().sum().item():.4f} "
                        f"k_absmax={k_tensor.float().abs().max().item():.4f} "
                        f"k_nonzero={k_tensor.count_nonzero().item()}/{k_tensor.numel()} "
                        f"indices={src_indices[:5]}"
                    )
                t_quant = time.perf_counter()
                total_gather_ms += (t_quant - t_gather) * 1000
                if _HAS_TRITON and nbits == 4:
                    k_data = _triton_quant_4bit_to_bytes(k_tensor)
                    v_data = _triton_quant_4bit_to_bytes(v_tensor)
                else:
                    k_data = quantize_on_gpu(k_tensor, nbits=nbits)
                    v_data = quantize_on_gpu(v_tensor, nbits=nbits)
                # No sync needed: quantize_on_gpu returns CPU bytes
                # (implicit sync via .cpu() inside quantize_on_gpu)
                # Explicitly free large GPU temporaries to prevent fragmentation
                del k_tensor, v_tensor
                quant_ms = (time.perf_counter() - t_quant) * 1000
                total_quant_ms += quant_ms
                k_lid = (layer_id * 2) | _MSG_QUANT_FLAG
                v_lid = (layer_id * 2 + 1) | _MSG_QUANT_FLAG
            else:
                # Legacy CPU path (no quantization)
                k_data = _read_pages_from_gpu(k_ptr, k_item_len, src_indices, page_size)
                v_data = _read_pages_from_gpu(v_ptr, v_item_len, src_indices, page_size)
                quant_ms = 0.0
                total_gather_ms += (time.perf_counter() - t_gather) * 1000
                k_lid = layer_id * 2
                v_lid = layer_id * 2 + 1

            layer_bytes = len(k_data) + len(v_data)
            total_transfer_bytes += layer_bytes

            t_send = time.perf_counter()
            _send_layer_data(conn, k_lid, k_data)
            _send_layer_data(conn, v_lid, v_data)
            send_ms = (time.perf_counter() - t_send) * 1000
            total_send_ms += send_ms

            logger.info(
                f"[TIMING] room={self.room} side=sender layer={layer_id} "
                f"quant_ms={quant_ms:.2f} send_ms={send_ms:.2f} "
                f"bytes={layer_bytes} nbits={nbits or 16}"
            )

        total_ms = (time.perf_counter() - t_stream_start) * 1000
        logger.warning(
            f"[TIMING_SUMMARY] room={self.room} side=sender total_ms={total_ms:.2f} "
            f"layers={num_layers} total_gather_ms={total_gather_ms:.2f} "
            f"total_quant_ms={total_quant_ms:.2f} total_send_ms={total_send_ms:.2f} "
            f"transfer_bytes={total_transfer_bytes}"
        )

        logger.warning(f"[_stream_kv] room={self.room} all {num_layers} layers sent, calling _send_tail")
        self._send_tail(conn)

    def _stream_kv_pipelined(self, conn: socket.socket) -> None:
        """Pipelined version: GPU quantize overlaps with TCP send via a bounded queue."""
        kv_mgr = self.kv_mgr
        kv_args = kv_mgr.kv_args
        src_indices = self.src_kv_indices
        page_size = kv_args.page_size
        num_layers = len(kv_args.kv_data_ptrs) // 2

        get_quant_bits = self._get_quant_bits
        kv_dtype = getattr(kv_mgr.server_args, "kv_cache_dtype", "auto")
        is_bf16 = kv_dtype in ("auto", "bf16", "bfloat16")
        torch_dtype = torch.bfloat16 if is_bf16 else torch.float16

        logger.warning(
            f"[_stream_kv] room={self.room} starting (pipelined), layers={num_layers}, "
            f"src_indices={src_indices}, page_size={page_size}, "
            f"quant={'gpu' if get_quant_bits else 'off'}"
        )

        torch.cuda.synchronize()
        logger.warning(f"[_stream_kv] room={self.room} cuda sync done, sending layers...")

        t_stream_start = time.perf_counter()
        total_quant_ms = 0.0
        total_gather_ms = 0.0
        total_transfer_bytes = 0

        send_queue: queue.Queue = queue.Queue(maxsize=2)
        send_error: list = [None]
        send_total_ms: list = [0.0]

        def _send_worker():
            try:
                while True:
                    item = send_queue.get()
                    if item is None:
                        break
                    k_lid, k_data, v_lid, v_data, lid, qms, lbytes, nb = item
                    t_s = time.perf_counter()
                    _send_layer_data(conn, k_lid, k_data)
                    _send_layer_data(conn, v_lid, v_data)
                    sms = (time.perf_counter() - t_s) * 1000
                    send_total_ms[0] += sms
                    logger.info(
                        f"[TIMING] room={self.room} side=sender layer={lid} "
                        f"quant_ms={qms:.2f} send_ms={sms:.2f} "
                        f"bytes={lbytes} nbits={nb or 16}"
                    )
            except Exception as e:
                send_error[0] = e

        sender = threading.Thread(target=_send_worker, daemon=True)
        sender.start()

        try:
            for layer_id in range(num_layers):
                if send_error[0] is not None:
                    raise RuntimeError(f"Send thread failed: {send_error[0]}") from send_error[0]

                if layer_id % 7 == 0:
                    logger.warning(f"[_stream_kv] room={self.room} sending layer {layer_id}/{num_layers}")

                k_ptr = kv_args.kv_data_ptrs[layer_id]
                k_item_len = kv_args.kv_item_lens[layer_id]
                v_ptr = kv_args.kv_data_ptrs[num_layers + layer_id]
                v_item_len = kv_args.kv_item_lens[num_layers + layer_id]

                nbits = get_quant_bits(layer_id) if get_quant_bits else None

                t_gather = time.perf_counter()
                if nbits is not None:
                    k_tensor = _read_pages_as_gpu_tensor(
                        k_ptr, k_item_len, src_indices, page_size, torch_dtype
                    )
                    v_tensor = _read_pages_as_gpu_tensor(
                        v_ptr, v_item_len, src_indices, page_size, torch_dtype
                    )
                    torch.cuda.synchronize()
                    if _DEBUG_QUANT and layer_id == 0:
                        logger.warning(
                            f"[_stream_kv DEBUG] room={self.room} layer=0 "
                            f"k_sum={k_tensor.float().sum().item():.4f} "
                            f"v_sum={v_tensor.float().sum().item():.4f} "
                            f"k_absmax={k_tensor.float().abs().max().item():.4f} "
                            f"k_nonzero={k_tensor.count_nonzero().item()}/{k_tensor.numel()} "
                            f"indices={src_indices[:5]}"
                        )
                    t_quant = time.perf_counter()
                    total_gather_ms += (t_quant - t_gather) * 1000
                    if _HAS_TRITON and nbits == 4:
                        k_data = _triton_quant_4bit_to_bytes(k_tensor)
                        v_data = _triton_quant_4bit_to_bytes(v_tensor)
                    else:
                        k_data = quantize_on_gpu(k_tensor, nbits=nbits)
                        v_data = quantize_on_gpu(v_tensor, nbits=nbits)
                    del k_tensor, v_tensor
                    quant_ms = (time.perf_counter() - t_quant) * 1000
                    total_quant_ms += quant_ms
                    k_lid = (layer_id * 2) | _MSG_QUANT_FLAG
                    v_lid = (layer_id * 2 + 1) | _MSG_QUANT_FLAG
                else:
                    k_data = _read_pages_from_gpu(k_ptr, k_item_len, src_indices, page_size)
                    v_data = _read_pages_from_gpu(v_ptr, v_item_len, src_indices, page_size)
                    quant_ms = 0.0
                    total_gather_ms += (time.perf_counter() - t_gather) * 1000
                    k_lid = layer_id * 2
                    v_lid = layer_id * 2 + 1

                layer_bytes = len(k_data) + len(v_data)
                total_transfer_bytes += layer_bytes

                send_queue.put(
                    (k_lid, k_data, v_lid, v_data, layer_id, quant_ms, layer_bytes, nbits)
                )
        finally:
            send_queue.put(None)
            sender.join(timeout=120)

        if send_error[0] is not None:
            raise RuntimeError(f"Send thread failed: {send_error[0]}") from send_error[0]

        total_send_ms = send_total_ms[0]
        total_ms = (time.perf_counter() - t_stream_start) * 1000
        logger.warning(
            f"[TIMING_SUMMARY] room={self.room} side=sender total_ms={total_ms:.2f} "
            f"layers={num_layers} total_gather_ms={total_gather_ms:.2f} "
            f"total_quant_ms={total_quant_ms:.2f} total_send_ms={total_send_ms:.2f} "
            f"transfer_bytes={total_transfer_bytes}"
        )

        logger.warning(f"[_stream_kv] room={self.room} all {num_layers} layers sent, calling _send_tail")
        self._send_tail(conn)

    def _send_tail(self, conn: socket.socket) -> None:
        """Send auxiliary metadata (if any) and the EOF sentinel."""
        kv_args = self.kv_mgr.kv_args

        # Read aux data from the PREFILL's metadata buffer index (src_aux_index),
        # where set_buf() wrote the bootstrap_room and other metadata.
        # dst_aux_index is the DECODE side's index — only used to decide whether
        # the decode side expects aux data at all.
        read_index = self.src_aux_index if self.src_aux_index is not None else self.dst_aux_index
        if read_index is not None and kv_args.aux_data_ptrs:
            # Send ALL aux buffers (output_ids, cached_tokens, ..., bootstrap_room)
            # so that the decode side receives the complete metadata.
            chunks = []
            for i, (aux_ptr, aux_item_len) in enumerate(
                zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens)
            ):
                chunk = _read_aux_from_cpu(aux_ptr, aux_item_len, read_index)
                chunks.append(chunk)
            aux_data = b"".join(chunks)
            logger.warning(
                f"[_send_tail] room={self.room} sending aux metadata, "
                f"buffers={len(kv_args.aux_data_ptrs)} total_len={len(aux_data)}"
            )
            _send_layer_data(conn, _MSG_DONE - 1, aux_data)  # layer_id = -2

        # Signal end of transfer
        logger.warning(f"[_send_tail] room={self.room} sending EOF marker")
        _send_layer_data(conn, _MSG_DONE, b"")
        logger.warning(f"[_send_tail] room={self.room} done!")

    def _finish(self, conn: socket.socket, success: bool) -> None:
        status = KVPoll.Success if success else KVPoll.Failed
        logger.warning(f"[_PendingTransfer._finish] room={self.room} success={success}")
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
            # cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount) → CUresult
            lib.cuMemcpyDtoD_v2.restype = ctypes.c_int
            lib.cuMemcpyDtoD_v2.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
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

    # Note: caller (_stream_kv) is responsible for synchronizing before
    # this function is called.  Removed redundant torch.cuda.synchronize().

    chunks = []
    for pi in page_indices:
        page_offset = int(pi) * item_len
        chunks.append(_gpu_to_cpu(base_ptr + page_offset, item_len))

    return b"".join(bytes(c) for c in chunks)


def _gpu_to_gpu(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Copy *nbytes* between two CUDA device pointers (DtoD)."""
    lib = _get_cuda_driver()
    rc = lib.cuMemcpyDtoD_v2(
        ctypes.c_uint64(dst_ptr), ctypes.c_uint64(src_ptr), nbytes
    )
    if rc != 0:
        raise RuntimeError(f"cuMemcpyDtoD_v2 failed with error code {rc}")


def _read_pages_as_gpu_tensor(
    base_ptr: int,
    item_len: int,
    page_indices: npt.NDArray[np.int32],
    page_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Gather scattered KV pages from the GPU pool into a contiguous GPU tensor.

    Uses DtoD copies (GPU memory bandwidth, ~900 GB/s on A100) instead of
    DtoH, so the full-size data never crosses the PCIe bus.
    """
    num_pages = len(page_indices)
    if num_pages == 0:
        return torch.empty(0, dtype=dtype, device="cuda")

    total_bytes = num_pages * item_len
    elem_size = 2  # bf16 / fp16
    tensor = torch.empty(total_bytes // elem_size, dtype=dtype, device="cuda")
    dst_ptr = tensor.data_ptr()

    for i, pi in enumerate(page_indices):
        src = base_ptr + int(pi) * item_len
        _gpu_to_gpu(dst_ptr + i * item_len, src, item_len)

    return tensor


def _write_gpu_tensor_to_pages(
    base_ptr: int,
    item_len: int,
    page_indices: npt.NDArray[np.int32],
    page_size: int,
    tensor: torch.Tensor,
) -> None:
    """
    Scatter a contiguous GPU tensor back into the KV pool's page slots.

    Inverse of :func:`_read_pages_as_gpu_tensor`.
    """
    src_ptr = tensor.data_ptr()
    for i, pi in enumerate(page_indices):
        dst = base_ptr + int(pi) * item_len
        _gpu_to_gpu(dst, src_ptr + i * item_len, item_len)


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
        self._pipeline_aborted: bool = False

        # Async send thread for send_layer: TCP sends happen in a background
        # thread so the forward pass doesn't block on network I/O.
        self._send_queue: Optional[queue.Queue] = None
        self._send_thread: Optional[threading.Thread] = None
        self._send_error: list = [None]
        self._send_thread_total_ms: list = [0.0]

        # Transfer quantization config
        server_args: Optional[ServerArgs] = getattr(mgr, "server_args", None)
        if server_args and getattr(server_args, "enable_transfer_quant", False):
            self._transfer_quant_bits: Optional[int] = server_args.transfer_quant_bits
            self._transfer_quant_layer_map: Optional[Dict[int, int]] = (
                self._parse_layer_bits(server_args)
            )
        else:
            self._transfer_quant_bits = None
            self._transfer_quant_layer_map = None

        if getattr(self.kv_mgr, "is_dummy_cp_rank", False):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
            return

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None) -> None:
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def _ensure_send_thread(self, conn: socket.socket) -> None:
        """Lazily create the async send thread for this request."""
        if self._send_thread is not None:
            return
        self._send_queue = queue.Queue(maxsize=2)
        self._send_error = [None]
        self._send_thread_total_ms = [0.0]
        _conn_ref = [conn]

        def _worker():
            try:
                while True:
                    item = self._send_queue.get()
                    if item is None:
                        break
                    k_lid, k_data, v_lid, v_data, lid, qms, gms, lbytes, nb = item
                    t_s = time.perf_counter()
                    _send_layer_data(_conn_ref[0], k_lid, k_data)
                    _send_layer_data(_conn_ref[0], v_lid, v_data)
                    sms = (time.perf_counter() - t_s) * 1000
                    self._send_thread_total_ms[0] += sms
                    logger.info(
                        f"[TIMING] room={self.bootstrap_room} side=sender_pipeline layer={lid} "
                        f"gather_ms={gms:.2f} quant_ms={qms:.2f} send_ms={sms:.2f} "
                        f"bytes={lbytes} nbits={nb or 16}"
                    )
            except Exception as e:
                self._send_error[0] = e

        self._send_thread = threading.Thread(target=_worker, daemon=True)
        self._send_thread.start()

    def _drain_send_thread(self) -> None:
        """Wait for the async send thread to finish all queued work."""
        if self._send_queue is None:
            return
        self._send_queue.put(None)
        self._send_thread.join(timeout=120)

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ) -> None:
        logger.warning(
            f"[TCPKVSender.send] room={self.bootstrap_room} "
            f"kv_indices={len(kv_indices)} _layer_sent={self._layer_sent}"
        )
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

        logger.warning(
            f"[TCPKVSender.send] room={self.bootstrap_room} pending={'found' if pending else 'None'}"
        )

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

        logger.warning(
            f"[TCPKVSender.send] room={self.bootstrap_room} calling ready() kv_indices={len(kv_indices)}"
        )

        if self._layer_sent:
            # Wait for async send thread to flush all queued data.
            self._drain_send_thread()
            if self._send_error[0] is not None:
                logger.error(
                    f"[TCPKVSender] room={self.bootstrap_room} "
                    f"send thread failed: {self._send_error[0]}"
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
            pending.done_pipeline(kv_indices, src_aux_index=self.aux_index)
        else:
            # Batch mode: let serve() read all GPU data and stream it now.
            pending.ready(kv_indices, get_quant_bits=self._get_layer_quant_bits, src_aux_index=self.aux_index)

    def send_layer(self, layer_id: int, src_indices) -> None:
        """
        Layer-wise pipeline send: called from the attention forward pass after
        each layer's KV cache has been written to the GPU pool.

        When ``SGLANG_TCP_PIPELINE_SEND=1`` (default), GPU work (gather + quantize)
        is decoupled from TCP sends via a background thread.  The forward pass
        only blocks on GPU operations; TCP sends happen asynchronously.

        If the pending transfer or the TCP connection is not yet available when
        called, the call is a no-op and ``send()`` will handle the transfer in
        batch mode instead.
        """
        if self._pipeline_aborted:
            return

        # Accept both torch.Tensor and np.ndarray
        if isinstance(src_indices, torch.Tensor):
            src_indices = src_indices.cpu().numpy().astype(np.int32)

        with self.kv_mgr._pending_lock:
            pending = self.kv_mgr._pending_transfers.get(self.bootstrap_room)

        if pending is None:
            return

        if not pending.wait_for_conn(timeout=_TCP_CONN_WAIT_S):
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

        nbits = self._get_layer_quant_bits(layer_id)
        t_gather = time.perf_counter()
        if nbits is not None:
            kv_dtype = getattr(self.kv_mgr.server_args, "kv_cache_dtype", "auto")
            is_bf16 = kv_dtype in ("auto", "bf16", "bfloat16")
            torch_dtype = torch.bfloat16 if is_bf16 else torch.float16
            k_tensor = _read_pages_as_gpu_tensor(
                k_ptr, k_item_len, src_indices, page_size, torch_dtype
            )
            v_tensor = _read_pages_as_gpu_tensor(
                v_ptr, v_item_len, src_indices, page_size, torch_dtype
            )
            torch.cuda.synchronize()
            if _DEBUG_QUANT and layer_id == 0:
                logger.warning(
                    f"[send_layer DEBUG] room={self.bootstrap_room} layer=0 "
                    f"k_sum={k_tensor.float().sum().item():.4f} "
                    f"v_sum={v_tensor.float().sum().item():.4f} "
                    f"k_absmax={k_tensor.float().abs().max().item():.4f} "
                    f"k_nonzero={k_tensor.count_nonzero().item()}/{k_tensor.numel()} "
                    f"indices={src_indices[:5]}"
                )
            t_quant = time.perf_counter()
            gather_ms = (t_quant - t_gather) * 1000
            if _HAS_TRITON and nbits == 4:
                k_data = _triton_quant_4bit_to_bytes(k_tensor)
                v_data = _triton_quant_4bit_to_bytes(v_tensor)
            else:
                k_data = quantize_on_gpu(k_tensor, nbits=nbits)
                v_data = quantize_on_gpu(v_tensor, nbits=nbits)
            del k_tensor, v_tensor
            quant_ms = (time.perf_counter() - t_quant) * 1000
            k_lid = (layer_id * 2) | _MSG_QUANT_FLAG
            v_lid = (layer_id * 2 + 1) | _MSG_QUANT_FLAG
        else:
            k_data = _read_pages_from_gpu(k_ptr, k_item_len, src_indices, page_size)
            v_data = _read_pages_from_gpu(v_ptr, v_item_len, src_indices, page_size)
            gather_ms = (time.perf_counter() - t_gather) * 1000
            quant_ms = 0.0
            k_lid = layer_id * 2
            v_lid = layer_id * 2 + 1

        layer_bytes = len(k_data) + len(v_data)
        self._layer_sent = True

        if _PIPELINE_SEND:
            self._ensure_send_thread(pending._conn)
            if self._send_error[0] is not None:
                self._pipeline_aborted = True
                return
            self._send_queue.put(
                (k_lid, k_data, v_lid, v_data, layer_id, quant_ms, gather_ms, layer_bytes, nbits)
            )
        else:
            try:
                t_send = time.perf_counter()
                _send_layer_data(pending._conn, k_lid, k_data)
                _send_layer_data(pending._conn, v_lid, v_data)
                send_ms = (time.perf_counter() - t_send) * 1000
                logger.info(
                    f"[TIMING] room={self.bootstrap_room} side=sender_pipeline layer={layer_id} "
                    f"gather_ms={gather_ms:.2f} quant_ms={quant_ms:.2f} send_ms={send_ms:.2f} "
                    f"bytes={layer_bytes} nbits={nbits or 16}"
                )
            except Exception as e:
                logger.warning(
                    f"[TCPKVSender] send_layer layer={layer_id} failed for "
                    f"room={self.bootstrap_room}: {e}; will retry in batch mode"
                )
                self._pipeline_aborted = True
                if not self._layer_sent:
                    self._layer_sent = False

    # ------------------------------------------------------------------
    # Transfer quantization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_layer_bits(server_args: "ServerArgs") -> Optional[Dict[int, int]]:
        """Parse kvtuner_layer_bits into a {layer_id: nbits} map."""
        raw = getattr(server_args, "kvtuner_layer_bits", None)
        if not raw:
            return None
        try:
            import json as _json
            bits_list = _json.loads(raw) if raw.strip().startswith("[") else [int(x) for x in raw.split(",")]
            return {i: b for i, b in enumerate(bits_list)}
        except Exception as e:
            logger.warning(f"[TCPKVSender] failed to parse kvtuner_layer_bits: {e}")
            return None

    def _get_layer_quant_bits(self, layer_id: int) -> Optional[int]:
        """Return quantization bits for a given layer, or the global default."""
        if self._transfer_quant_bits is None:
            return None
        if self._transfer_quant_layer_map and layer_id in self._transfer_quant_layer_map:
            return self._transfer_quant_layer_map[layer_id]
        return self._transfer_quant_bits

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        """Release request_status entry to prevent unbounded growth."""
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)

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
        # Transition from Bootstrapping to WaitingForInput now that bootstrap
        # info has been fetched (same pattern as MooncakeKVReceiver).
        if self.bootstrap_infos is not None:
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
        self._transfer_started = False
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
        self._transfer_started = True

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
            # Ensure this background thread has a valid CUDA context so that
            # cuMemcpyHtoD_v2 calls in _write_pages_to_gpu succeed.
            torch.cuda.set_device(self.kv_mgr.kv_args.gpu_id)

            logger.warning(
                f"[_recv_loop] room={self.bootstrap_room} started, "
                f"bootstrap_infos={'yes' if self.bootstrap_infos else 'None'}"
            )

            # Sleep briefly so the prefill side has time to process the ZMQ descriptor
            # and create a _PendingTransfer before the TCP connection arrives.
            logger.warning(f"[_recv_loop] room={self.bootstrap_room} sleeping {_TCP_CONNECT_DELAY_S}s before connect")
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
                tcp_port_raw = binfo.get("rank_port")
                logger.warning(
                    f"[TCPKVReceiver] room={self.bootstrap_room}: tcp_port not in "
                    f"bootstrap info; falling back to rank_port={tcp_port_raw}. "
                    "Ensure the prefill server uses the TCP backend."
                )
            prefill_tcp_port = tcp_port_raw

            logger.warning(
                f"[_recv_loop] room={self.bootstrap_room} connecting to "
                f"{prefill_ip}:{prefill_tcp_port} (timeout={_RECV_TIMEOUT_S}s)"
            )
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.settimeout(_RECV_TIMEOUT_S)
            conn.connect((prefill_ip, int(prefill_tcp_port)))
            logger.warning(f"[_recv_loop] room={self.bootstrap_room} TCP connected!")

            # Introduce ourselves
            conn.sendall(struct.pack("!Q", self.bootstrap_room))
            logger.warning(f"[_recv_loop] room={self.bootstrap_room} sent room id, starting _recv_kv")

            self._recv_kv(conn, dst_kv_indices, dst_aux_index)
            # Final sync: ensure all GPU writes (DtoD scatter from
            # _write_gpu_tensor_to_pages) are visible before the decode
            # model reads the KV cache.
            torch.cuda.synchronize()
            logger.warning(f"[_recv_loop] room={self.bootstrap_room} _recv_kv completed!")
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
        """Receive layer-by-layer KV data and write to GPU pool.

        When prefix caching is active on the prefill side, the sender may
        transmit only the *incremental* (non-cached) pages while the decode
        side has allocated pages for the full sequence.  We detect this by
        comparing the received tensor/data size against the expected size
        derived from ``dst_kv_indices`` and, when they differ, write only
        to the **tail** of ``dst_kv_indices`` (the incremental portion).
        """
        kv_args = self.kv_mgr.kv_args
        num_layers = len(kv_args.kv_data_ptrs) // 2
        page_size = kv_args.page_size
        expected_msgs = num_layers * 2  # K + V per layer
        received = 0
        # Will be set on the first received layer if prefix caching is detected
        _effective_indices: Optional[npt.NDArray[np.int32]] = None

        logger.warning(
            f"[_recv_kv] room={self.bootstrap_room} expecting {expected_msgs} msgs "
            f"(layers={num_layers}), dst_kv_indices={len(dst_kv_indices)}"
        )

        t_recv_start = time.perf_counter()
        total_recv_ms = 0.0
        total_dequant_ms = 0.0
        total_write_ms = 0.0

        while True:
            t_recv = time.perf_counter()
            layer_id, data = _recv_layer_data(conn)
            recv_ms = (time.perf_counter() - t_recv) * 1000

            if layer_id == _MSG_DONE:
                total_ms = (time.perf_counter() - t_recv_start) * 1000
                logger.warning(
                    f"[_recv_kv] room={self.bootstrap_room} got EOF, "
                    f"received={received}/{expected_msgs}"
                )
                logger.warning(
                    f"[TIMING_SUMMARY] room={self.bootstrap_room} side=receiver "
                    f"total_ms={total_ms:.2f} layers={num_layers} "
                    f"total_recv_ms={total_recv_ms:.2f} "
                    f"total_dequant_ms={total_dequant_ms:.2f} "
                    f"total_write_ms={total_write_ms:.2f}"
                )
                break

            if layer_id == _MSG_DONE - 1:
                # Auxiliary metadata — contains ALL aux buffers concatenated
                logger.warning(
                    f"[_recv_kv] room={self.bootstrap_room} got aux metadata, "
                    f"len={len(data)}"
                )
                if dst_aux_index is not None and kv_args.aux_data_ptrs and data:
                    # Split the concatenated blob back into individual aux buffers
                    offset = 0
                    for i, (aux_ptr, aux_item_len) in enumerate(
                        zip(kv_args.aux_data_ptrs, kv_args.aux_item_lens)
                    ):
                        chunk = data[offset : offset + aux_item_len]
                        if chunk:
                            _write_aux_to_cpu(aux_ptr, aux_item_len, dst_aux_index, chunk)
                        offset += aux_item_len
                continue

            total_recv_ms += recv_ms

            # Check and strip quantization flag
            is_quantized = bool(layer_id & _MSG_QUANT_FLAG)
            if is_quantized:
                layer_id &= ~_MSG_QUANT_FLAG

            # Determine whether this is K or V
            if layer_id % 2 == 0:
                actual_layer = layer_id // 2
                base_ptr = kv_args.kv_data_ptrs[actual_layer]
                item_len = kv_args.kv_item_lens[actual_layer]
            else:
                actual_layer = layer_id // 2
                base_ptr = kv_args.kv_data_ptrs[num_layers + actual_layer]
                item_len = kv_args.kv_item_lens[num_layers + actual_layer]

            # Dequantize on GPU and scatter to KV pool, or write raw bytes
            dequant_ms = 0.0
            if is_quantized:
                gpu_device = torch.device("cuda", self.kv_mgr.kv_args.gpu_id)
                t_dequant = time.perf_counter()
                nbits = _struct.unpack(_WIRE_HEADER_FMT, data[:_WIRE_HEADER_SIZE])[0]
                if _HAS_TRITON and nbits == 4:
                    tensor = _triton_dequant_4bit_from_bytes(data, gpu_device)
                else:
                    tensor = dequantize_on_gpu(data, gpu_device)
                # Sync: dequantize_on_gpu uses PyTorch ops (current stream),
                # but _write_gpu_tensor_to_pages uses cuMemcpyDtoD (NULL stream).
                # Ensure dequantized data is fully materialized before scatter.
                torch.cuda.synchronize()
                dequant_ms = (time.perf_counter() - t_dequant) * 1000
                total_dequant_ms += dequant_ms

                # --- Prefix-caching adaptation ---
                # Compute the number of received pages from the tensor size.
                recv_pages = tensor.numel() * tensor.element_size() // item_len
                if _effective_indices is None and recv_pages != len(dst_kv_indices):
                    # Sender transmitted only incremental pages (prefix cached).
                    # Use the *tail* of dst_kv_indices for the write target.
                    _effective_indices = dst_kv_indices[-recv_pages:]
                    logger.warning(
                        f"[_recv_kv] room={self.bootstrap_room} prefix-cache detected: "
                        f"recv_pages={recv_pages} dst_pages={len(dst_kv_indices)} "
                        f"using tail indices"
                    )
                elif _effective_indices is None:
                    _effective_indices = dst_kv_indices
                write_indices = _effective_indices

                if _DEBUG_QUANT and actual_layer == 0 and layer_id % 2 == 0:
                    logger.warning(
                        f"[_recv_kv DEBUG] room={self.bootstrap_room} layer=0 K "
                        f"dequant_sum={tensor.float().sum().item():.4f} "
                        f"dequant_absmax={tensor.float().abs().max().item():.4f} "
                        f"dequant_nonzero={tensor.count_nonzero().item()}/{tensor.numel()} "
                        f"packed_bytes={len(data)}"
                    )
                if received % 7 == 0:
                    logger.warning(
                        f"[_recv_kv] room={self.bootstrap_room} writing layer_id={layer_id} "
                        f"(actual_layer={actual_layer}, {'K' if layer_id % 2 == 0 else 'V'}) "
                        f"tensor={tensor.shape} quant=True received={received}/{expected_msgs}"
                    )
                t_write = time.perf_counter()
                _write_gpu_tensor_to_pages(
                    base_ptr, item_len, write_indices, page_size, tensor
                )
                write_ms = (time.perf_counter() - t_write) * 1000
                total_write_ms += write_ms
                if _DEBUG_QUANT and actual_layer == 0 and layer_id % 2 == 0:
                    # Read-back verification: re-gather what we just wrote
                    kv_dtype = getattr(self.kv_mgr.server_args, "kv_cache_dtype", "auto")
                    is_bf16 = kv_dtype in ("auto", "bf16", "bfloat16")
                    rb_dtype = torch.bfloat16 if is_bf16 else torch.float16
                    readback = _read_pages_as_gpu_tensor(
                        base_ptr, item_len, write_indices, page_size, rb_dtype
                    )
                    torch.cuda.synchronize()
                    diff = (tensor.float() - readback.float()).abs()
                    logger.warning(
                        f"[_recv_kv VERIFY] room={self.bootstrap_room} layer=0 K "
                        f"readback_sum={readback.float().sum().item():.4f} "
                        f"max_diff={diff.max().item():.6f} "
                        f"mean_diff={diff.mean().item():.6f} "
                        f"match={diff.max().item() < 1e-5}"
                    )
                    del readback, diff
                # Free dequantized tensor immediately to reduce GPU memory pressure
                del tensor
            else:
                # --- Prefix-caching adaptation (non-quantized path) ---
                recv_pages = len(data) // item_len
                if _effective_indices is None and recv_pages != len(dst_kv_indices):
                    _effective_indices = dst_kv_indices[-recv_pages:]
                    logger.warning(
                        f"[_recv_kv] room={self.bootstrap_room} prefix-cache detected: "
                        f"recv_pages={recv_pages} dst_pages={len(dst_kv_indices)} "
                        f"using tail indices"
                    )
                elif _effective_indices is None:
                    _effective_indices = dst_kv_indices
                write_indices = _effective_indices

                if received % 7 == 0:
                    logger.warning(
                        f"[_recv_kv] room={self.bootstrap_room} writing layer_id={layer_id} "
                        f"(actual_layer={actual_layer}, {'K' if layer_id % 2 == 0 else 'V'}) "
                        f"data_len={len(data)} quant=False received={received}/{expected_msgs}"
                    )
                t_write = time.perf_counter()
                _write_pages_to_gpu(base_ptr, item_len, write_indices, page_size, data)
                write_ms = (time.perf_counter() - t_write) * 1000
                total_write_ms += write_ms

            logger.info(
                f"[TIMING] room={self.bootstrap_room} side=receiver layer={actual_layer} "
                f"kv={'K' if layer_id % 2 == 0 else 'V'} recv_ms={recv_ms:.2f} "
                f"dequant_ms={dequant_ms:.2f} write_ms={write_ms:.2f} "
                f"bytes={len(data)} quant={is_quantized}"
            )
            received += 1

    def poll(self) -> KVPoll:
        if not self._transfer_started:
            return KVPoll.WaitingForInput
        if not self._transfer_done.is_set():
            return KVPoll.Transferring
        return KVPoll.Success if self._transfer_ok else KVPoll.Failed

    def failure_exception(self) -> None:
        raise RuntimeError(
            f"TCPKVReceiver: transfer failed for room={self.bootstrap_room}"
        )

    def clear(self) -> None:
        """Release request_status entry to prevent unbounded growth."""
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)

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

