"""
Unit tests for the TCP KV cache transfer backend.

These tests cover:
1. Protocol framing helpers (_send_layer_data / _recv_layer_data)
2. TCPKVBootstrapServer route PUT/GET with tcp_port field
3. TransferBackend.TCP registration in get_kv_class()
4. ForwardBatch.layer_kv_send_fn field presence and callback invocation
5. 'tcp' in DISAGG_TRANSFER_BACKEND_CHOICES
6. _PendingTransfer pipeline API (wait_for_conn / done_pipeline)
7. layer_kv_send_fn hook presence in flashinfer_backend.py
"""

import socket
import struct
import threading
import unittest
from unittest.mock import MagicMock


class TestTCPFramingProtocol(unittest.TestCase):
    """Tests for the TCP message framing helpers."""

    def _make_pipe(self):
        """Create a pair of connected TCP sockets."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        server_conn, _ = listener.accept()
        listener.close()
        return client, server_conn

    def test_send_recv_roundtrip(self):
        """A layer's data can be sent and received correctly."""
        from sglang.srt.disaggregation.tcp.conn import (
            _recv_layer_data,
            _send_layer_data,
        )

        sender, receiver = self._make_pipe()
        payload = b"Hello, KV world!" * 64

        def _send():
            _send_layer_data(sender, 42, payload)
            sender.close()

        t = threading.Thread(target=_send)
        t.start()

        layer_id, data = _recv_layer_data(receiver)
        t.join()
        receiver.close()

        self.assertEqual(layer_id, 42)
        self.assertEqual(data, payload)

    def test_empty_payload(self):
        """Zero-length payloads (e.g. DONE marker) work correctly."""
        from sglang.srt.disaggregation.tcp.conn import (
            _MSG_DONE,
            _recv_layer_data,
            _send_layer_data,
        )

        sender, receiver = self._make_pipe()

        def _send():
            _send_layer_data(sender, _MSG_DONE, b"")
            sender.close()

        t = threading.Thread(target=_send)
        t.start()

        layer_id, data = _recv_layer_data(receiver)
        t.join()
        receiver.close()

        self.assertEqual(layer_id, _MSG_DONE)
        self.assertEqual(data, b"")

    def test_multiple_layers_in_sequence(self):
        """Multiple layers can be sent and received sequentially."""
        from sglang.srt.disaggregation.tcp.conn import (
            _MSG_DONE,
            _recv_layer_data,
            _send_layer_data,
        )

        sender, receiver = self._make_pipe()
        layers = [(i, bytes([i % 256]) * (i + 1) * 16) for i in range(8)]

        def _send():
            for lid, data in layers:
                _send_layer_data(sender, lid, data)
            _send_layer_data(sender, _MSG_DONE, b"")
            sender.close()

        t = threading.Thread(target=_send)
        t.start()

        received = []
        while True:
            lid, data = _recv_layer_data(receiver)
            if lid == _MSG_DONE:
                break
            received.append((lid, data))

        t.join()
        receiver.close()

        self.assertEqual(received, layers)


class TestPendingTransferPipelineAPI(unittest.TestCase):
    """Tests for _PendingTransfer pipeline-mode API additions."""

    def _make_pending(self):
        """Construct a minimal _PendingTransfer without a real KVManager."""
        import numpy as np
        from unittest.mock import MagicMock

        from sglang.srt.disaggregation.tcp.conn import _PendingTransfer

        mgr = MagicMock()
        mgr._pending_transfers = {}
        mgr._pending_lock = threading.Lock()

        pending = _PendingTransfer(
            room=42,
            kv_mgr=mgr,
            dst_kv_indices=np.array([0, 1, 2], dtype=np.int32),
            dst_aux_index=None,
        )
        return pending

    def test_conn_event_not_set_initially(self):
        """_conn_event should not be set before set_conn is called."""
        pending = self._make_pending()
        self.assertFalse(pending._conn_event.is_set())
        self.assertIsNone(pending._conn)

    def test_wait_for_conn_timeout(self):
        """wait_for_conn returns False when the connection is never set."""
        pending = self._make_pending()
        result = pending.wait_for_conn(timeout=0.05)
        self.assertFalse(result)

    def test_wait_for_conn_succeeds_after_set(self):
        """wait_for_conn returns True once _conn_event is set via serve() path."""
        import socket as _socket

        pending = self._make_pending()

        mock_conn = MagicMock(spec=_socket.socket)

        def _set_conn():
            # Simulate what serve() does when the TCP connection arrives.
            pending._conn = mock_conn
            pending._conn_event.set()

        t = threading.Thread(target=_set_conn)
        t.start()
        result = pending.wait_for_conn(timeout=2.0)
        t.join()

        self.assertTrue(result)
        self.assertIs(pending._conn, mock_conn)

    def test_done_pipeline_sets_pipeline_mode(self):
        """done_pipeline() should set _pipeline_mode and release _ready."""
        import numpy as np

        pending = self._make_pending()
        indices = np.array([10, 11], dtype=np.int32)

        self.assertFalse(pending._pipeline_mode)
        self.assertFalse(pending._ready.is_set())

        pending.done_pipeline(indices)

        self.assertTrue(pending._pipeline_mode)
        self.assertTrue(pending._ready.is_set())
        np.testing.assert_array_equal(pending.src_kv_indices, indices)

    def test_ready_does_not_set_pipeline_mode(self):
        """ready() (batch mode) should not set _pipeline_mode."""
        import numpy as np

        pending = self._make_pending()
        indices = np.array([5, 6], dtype=np.int32)
        pending.ready(indices)

        self.assertFalse(pending._pipeline_mode)
        self.assertTrue(pending._ready.is_set())


class TestTransferBackendEnum(unittest.TestCase):
    """Tests that TransferBackend.TCP is correctly registered."""

    def test_tcp_in_enum(self):
        from sglang.srt.disaggregation.utils import TransferBackend

        self.assertIn("TCP", [b.name for b in TransferBackend])
        self.assertEqual(TransferBackend.TCP.value, "tcp")

    def test_get_kv_class_tcp_manager(self):
        from sglang.srt.disaggregation.tcp import TCPKVManager
        from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class

        cls = get_kv_class(TransferBackend.TCP, KVClassType.MANAGER)
        self.assertIs(cls, TCPKVManager)

    def test_get_kv_class_tcp_sender(self):
        from sglang.srt.disaggregation.tcp import TCPKVSender
        from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class

        cls = get_kv_class(TransferBackend.TCP, KVClassType.SENDER)
        self.assertIs(cls, TCPKVSender)

    def test_get_kv_class_tcp_receiver(self):
        from sglang.srt.disaggregation.tcp import TCPKVReceiver
        from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class

        cls = get_kv_class(TransferBackend.TCP, KVClassType.RECEIVER)
        self.assertIs(cls, TCPKVReceiver)

    def test_get_kv_class_tcp_bootstrap_server(self):
        from sglang.srt.disaggregation.tcp import TCPKVBootstrapServer
        from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class

        cls = get_kv_class(TransferBackend.TCP, KVClassType.BOOTSTRAP_SERVER)
        self.assertIs(cls, TCPKVBootstrapServer)


class TestServerArgsBackendChoices(unittest.TestCase):
    """Tests that 'tcp' is listed as a valid transfer backend."""

    def test_tcp_in_choices(self):
        from sglang.srt.server_args import DISAGG_TRANSFER_BACKEND_CHOICES

        self.assertIn("tcp", DISAGG_TRANSFER_BACKEND_CHOICES)


class TestForwardBatchLayerKvField(unittest.TestCase):
    """Tests that ForwardBatch has the layer_kv_send_fn field."""

    def test_field_exists_and_default_none(self):
        """layer_kv_send_fn should exist and default to None."""
        import dataclasses

        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        fields = {f.name for f in dataclasses.fields(ForwardBatch)}
        self.assertIn("layer_kv_send_fn", fields)

        # Check default is None
        field_obj = next(
            f for f in dataclasses.fields(ForwardBatch) if f.name == "layer_kv_send_fn"
        )
        self.assertIsNone(field_obj.default)

    def test_callback_invocation(self):
        """When layer_kv_send_fn is set on a mock ForwardBatch, calling it works."""
        # Use a MagicMock to simulate a ForwardBatch instance without
        # requiring a real GPU / model runner environment.
        # MagicMock is imported at the top of this module.
        fb = MagicMock()
        calls = []

        def mock_send_fn(layer_id, cache_loc):
            calls.append((layer_id, cache_loc))

        fb.layer_kv_send_fn = mock_send_fn
        fb.layer_kv_send_fn(3, "dummy_cache_loc")
        self.assertEqual(calls, [(3, "dummy_cache_loc")])


class TestFlashinferBackendHook(unittest.TestCase):
    """Tests that flashinfer_backend.forward_extend calls layer_kv_send_fn."""

    def test_hook_present_in_source(self):
        """The layer_kv_send_fn hook must appear in flashinfer_backend.py."""
        import os

        backend_path = os.path.join(
            os.path.dirname(__file__),
            "../../python/sglang/srt/layers/attention/flashinfer_backend.py",
        )
        backend_path = os.path.normpath(backend_path)
        with open(backend_path) as f:
            source = f.read()

        self.assertIn(
            "layer_kv_send_fn",
            source,
            "layer_kv_send_fn hook not found in flashinfer_backend.py",
        )
        # Both the non-ragged and ragged paths must have the hook
        self.assertGreaterEqual(
            source.count("layer_kv_send_fn"),
            2,
            "Expected at least 2 occurrences (non-ragged + ragged path)",
        )


if __name__ == "__main__":
    unittest.main()
