#!/usr/bin/env python3
"""
Parse TIMING logs from prefill/decode servers to decompose TTFT.

Usage:
    python parse_micro_benchmark.py --prefill-log /tmp/prefill.log \
        --decode-log /tmp/decode.log [--output /tmp/ttft_decomposition.json]

Reads TIMING/TIMING_SUMMARY lines from both server logs and computes:
  - T_prefill_compute: time from first send_layer to send() entry
  - T_kv_transfer: sender total_ms (gather+quant+send)
  - T_recv_total: receiver total_ms (recv+dequant+write)
  - Per-layer breakdown of quant/send/recv/dequant/write
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_timing_summary(log_text: str, side: str) -> list:
    """Extract TIMING_SUMMARY lines from log."""
    pattern = rf"\[TIMING_SUMMARY\] room=(\S+) side={side} (.+)"
    results = []
    for m in re.finditer(pattern, log_text):
        room = m.group(1)
        kv_str = m.group(2)
        kv = {}
        for pair in re.finditer(r'(\w+)=(\S+)', kv_str):
            key, val = pair.group(1), pair.group(2)
            try:
                kv[key] = float(val)
            except ValueError:
                kv[key] = val
        kv["room"] = room
        results.append(kv)
    return results


def parse_timing_lines(log_text: str, side: str) -> list:
    """Extract per-layer TIMING lines."""
    pattern = rf"\[TIMING\] room=(\S+) side={side}(.+)"
    results = []
    for m in re.finditer(pattern, log_text):
        room = m.group(1)
        rest = m.group(2)
        kv = {}
        for pair in re.finditer(r'(\w+)=(\S+)', rest):
            key, val = pair.group(1), pair.group(2)
            try:
                kv[key] = float(val)
            except ValueError:
                if val.startswith("ts="):
                    kv[key] = val  # keep timestamp as string
                else:
                    kv[key] = val
        kv["room"] = room
        results.append(kv)
    return results


def parse_prefill_markers(log_text: str) -> dict:
    """Extract prefill timing markers: send_entry, first_send_layer, transfer_done."""
    markers = {}
    for m in re.finditer(r'\[TIMING\] room=(\S+) side=prefill (\w+) (.+)', log_text):
        room = m.group(1)
        event = m.group(2)
        rest = m.group(3)
        kv = {"room": room}
        for pair in re.finditer(r'(\w+)=(\S+)', rest):
            kv[pair.group(1)] = pair.group(2)
        markers.setdefault(room, {})[event] = kv
    return markers


def decompose_ttft(prefill_log: str, decode_log: str) -> dict:
    """Decompose TTFT from server logs."""

    # Parse sender summaries
    sender_summaries = parse_timing_summary(prefill_log, "sender")
    # Parse receiver summaries
    recv_summaries = parse_timing_summary(decode_log, "receiver")

    # Parse per-layer timing
    sender_layers = parse_timing_lines(prefill_log, "sender")
    sender_pipeline_layers = parse_timing_lines(prefill_log, "sender_pipeline")
    recv_layers = parse_timing_lines(decode_log, "receiver")

    # Parse prefill markers
    prefill_markers = parse_prefill_markers(prefill_log)

    decomposition = {
        "sender_summaries": sender_summaries,
        "receiver_summaries": recv_summaries,
        "sender_layers": sender_layers,
        "sender_pipeline_layers": sender_pipeline_layers,
        "receiver_layers": recv_layers,
        "prefill_markers": prefill_markers,
    }

    # Compute aggregate metrics per room
    for room_markers in prefill_markers.values():
        if "send_entry" in room_markers and "first_send_layer" in room_markers:
            try:
                t_entry = float(room_markers["send_entry"].get("ts", 0))
                t_first = float(room_markers["first_send_layer"].get("ts", 0))
                prefill_compute_ms = (t_first - t_entry) * 1000
                room_markers["prefill_compute_ms"] = round(prefill_compute_ms, 2)
            except (ValueError, TypeError):
                pass

        if "send_entry" in room_markers and "transfer_done" in room_markers:
            try:
                t_entry = float(room_markers["send_entry"].get("ts", 0))
                t_done = float(room_markers["transfer_done"].get("ts", 0))
                total_send_ms = (t_done - t_entry) * 1000
                room_markers["total_send_ms_from_markers"] = round(total_send_ms, 2)
            except (ValueError, TypeError):
                pass

    return decomposition


def print_report(decomposition: dict):
    """Print a human-readable TTFT decomposition report."""

    sender_summaries = decomposition["sender_summaries"]
    recv_summaries = decomposition["receiver_summaries"]
    sender_layers = decomposition["sender_layers"]
    sender_pipeline_layers = decomposition["sender_pipeline_layers"]
    recv_layers = decomposition["receiver_layers"]
    prefill_markers = decomposition["prefill_markers"]

    all_layers = sender_layers + sender_pipeline_layers
    all_recv = recv_layers

    if not all_layers and not all_recv:
        print("No TIMING data found in logs.")
        print("Make sure LOG_LEVEL=info is set when running the benchmark.")
        return

    # Find the room used (usually the latest)
    rooms_in_markers = list(prefill_markers.keys())
    room = rooms_in_markers[-1] if rooms_in_markers else (sender_summaries[0]["room"] if sender_summaries else "unknown")

    print(f"\n{'='*80}")
    print(f"TTFT Decomposition Report (room: {room})")
    print(f"{'='*80}")

    # Sender summary
    if sender_summaries:
        s = sender_summaries[-1]  # latest
        print(f"\n--- Sender (prefill) ---")
        print(f"  Total transfer time: {s.get('total_ms', 0):.2f} ms")
        print(f"    Gather (DtoD copy):  {s.get('total_gather_ms', 0):.2f} ms")
        print(f"    Quantize (GPU):      {s.get('total_quant_ms', 0):.2f} ms")
        print(f"    TCP Send:            {s.get('total_send_ms', 0):.2f} ms")
        print(f"  Transfer bytes: {s.get('transfer_bytes', 0) / 1024:.1f} KB")
        print(f"  Layers: {s.get('layers', 0)}")

        # Overlap analysis
        compute_ms = s.get('total_gather_ms', 0) + s.get('total_quant_ms', 0)
        send_ms = s.get('total_send_ms', 0)
        total_ms = s.get('total_ms', 0)
        if total_ms > 0:
            overlap_ms = compute_ms + send_ms - total_ms
            overlap_pct = (overlap_ms / (compute_ms + send_ms) * 100) if (compute_ms + send_ms) > 0 else 0
            print(f"\n  Overlap analysis:")
            print(f"    gather+quant = {compute_ms:.2f} ms")
            print(f"    send         = {send_ms:.2f} ms")
            print(f"    sum          = {compute_ms + send_ms:.2f} ms")
            print(f"    wall-clock   = {total_ms:.2f} ms")
            print(f"    overlap saved= {max(0, overlap_ms):.2f} ms ({overlap_pct:.1f}%)")

    # Receiver summary
    if recv_summaries:
        r = recv_summaries[-1]
        print(f"\n--- Receiver (decode) ---")
        print(f"  Total receive time: {r.get('total_ms', 0):.2f} ms")
        print(f"    TCP Recv:          {r.get('total_recv_ms', 0):.2f} ms")
        print(f"    Dequantize (GPU):  {r.get('total_dequant_ms', 0):.2f} ms")
        print(f"    Write (DtoD):      {r.get('total_write_ms', 0):.2f} ms")
        print(f"  Layers: {r.get('layers', 0)}")

    # Prefill markers
    if prefill_markers and room in prefill_markers:
        m = prefill_markers[room]
        print(f"\n--- Prefill Compute ---")
        if "prefill_compute_ms" in m:
            print(f"  Prefill forward (send_entry → first_send_layer): {m['prefill_compute_ms']:.2f} ms")
        if "total_send_ms_from_markers" in m:
            print(f"  Total send() duration (send_entry → transfer_done): {m['total_send_ms_from_markers']:.2f} ms")

    # Per-layer breakdown (last 5 layers)
    if all_layers:
        print(f"\n--- Per-layer Sender Timing (last 5 layers) ---")
        print(f"  {'Layer':>5} {'Gather(ms)':>11} {'Quant(ms)':>11} {'Send(ms)':>11} {'Bytes':>10}")
        recent = all_layers[-10:]  # last 10 entries (5 layers × 2 K/V)
        for l in recent:
            lid = int(l.get("layer", 0))
            g = l.get("gather_ms", "-")
            q = l.get("quant_ms", "-")
            s = l.get("send_ms", "-")
            b = int(l.get("bytes", 0))
            if isinstance(g, float):
                g = f"{g:.2f}"
            if isinstance(q, float):
                q = f"{q:.2f}"
            if isinstance(s, float):
                s = f"{s:.2f}"
            kv = "K" if lid % 2 == 0 else "V"
            layer_num = lid // 2 if lid > 20 else lid
            print(f"  L{layer_num:3d} {kv}    {g:>11} {q:>11} {s:>11} {b/1024:>9.1f}K")

    if all_recv:
        print(f"\n--- Per-layer Receiver Timing (last 5 layers) ---")
        print(f"  {'Layer':>5} {'Recv(ms)':>11} {'Dequant(ms)':>11} {'Write(ms)':>11} {'Bytes':>10}")
        recent = all_recv[-10:]
        for l in recent:
            lid = int(l.get("layer", 0))
            r = l.get("recv_ms", "-")
            d = l.get("dequant_ms", "-")
            w = l.get("write_ms", "-")
            b = int(l.get("bytes", 0))
            if isinstance(r, float):
                r = f"{r:.2f}"
            if isinstance(d, float):
                d = f"{d:.2f}"
            if isinstance(w, float):
                w = f"{w:.2f}"
            kv = "K" if lid % 2 == 0 else "V"
            layer_num = lid // 2 if lid > 20 else lid
            print(f"  L{layer_num:3d} {kv}    {r:>11} {d:>11} {w:>11} {b/1024:>9.1f}K")

    # TTFT estimate
    if sender_summaries and recv_summaries:
        s = sender_summaries[-1]
        r = recv_summaries[-1]
        sender_total = s.get("total_ms", 0)
        recv_total = r.get("total_ms", 0)

        # TTFT components
        prefill_compute = 0
        if prefill_markers and room in prefill_markers:
            m = prefill_markers[room]
            prefill_compute = m.get("prefill_compute_ms", 0)

        print(f"\n{'='*80}")
        print(f"TTFT Decomposition Estimate")
        print(f"{'='*80}")
        print(f"  T_prefill_compute:    {prefill_compute:>8.2f} ms  (prefill forward pass)")
        print(f"  T_kv_transfer:        {max(sender_total, recv_total):>8.2f} ms  (max of sender/receiver)")
        print(f"    sender total:       {sender_total:>8.2f} ms  (gather+quant+send)")
        print(f"    receiver total:     {recv_total:>8.2f} ms  (recv+dequant+write)")
        print(f"  T_http_overhead:      {'??':>8} ms  (HTTP request/response)")
        print(f"  T_decode_first_token: {'??':>8} ms  (decode forward pass)")
        print(f"  ----------------------------------------")
        estimated_min = prefill_compute + max(sender_total, recv_total)
        print(f"  Estimated minimum:    {estimated_min:>8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Parse TIMING logs for TTFT decomposition")
    parser.add_argument("--prefill-log", required=True, help="Path to prefill server log")
    parser.add_argument("--decode-log", required=True, help="Path to decode server log")
    parser.add_argument("--output", default="/tmp/ttft_decomposition.json", help="Output JSON path")
    args = parser.parse_args()

    prefill_text = Path(args.prefill_log).read_text()
    decode_text = Path(args.decode_log).read_text()

    decomposition = decompose_ttft(prefill_text, decode_text)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(decomposition, f, indent=2, default=str)
    print(f"JSON saved to {args.output}")

    # Print report
    print_report(decomposition)


if __name__ == "__main__":
    main()
