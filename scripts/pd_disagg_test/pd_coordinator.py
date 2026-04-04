#!/usr/bin/env python3
"""
Minimal PD disaggregation coordinator that bypasses the Rust Router.

This script simulates what the Router does:
1. Queries the Prefill bootstrap server to get a bootstrap_room
2. Sends the request to Prefill with bootstrap_room
3. Sends the request to Decode with bootstrap_room
4. Collects responses from both

Usage:
    python3 pd_coordinator.py \
        --prefill-host 10.60.23.70 --prefill-port 30000 \
        --decode-host 10.60.30.66 --decode-port 30001 \
        --bootstrap-port 8998 \
        --text "Hello, how are you?" \
        --max-new-tokens 32
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def query_bootstrap_server(host: str, bootstrap_port: int) -> dict:
    """Query the Prefill bootstrap server for server info."""
    url = f"http://{host}:{bootstrap_port}/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1"
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Failed to query bootstrap server: {e}")
        sys.exit(1)


def query_bootstrap_room(host: str, bootstrap_port: int) -> dict:
    """Query the bootstrap server for a specific room assignment."""
    url = f"http://{host}:{bootstrap_port}/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0"
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"[ERROR] Bootstrap room query failed: {e.code} {e.read().decode()}")
        sys.exit(1)


def send_generate_request(
    host: str, port: int, text: str, max_new_tokens: int,
    bootstrap_host: str, bootstrap_port: int, bootstrap_room: int,
    rid: str, timeout: int = 120,
) -> dict:
    """Send a /generate request to a sglang server."""
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        "rid": rid,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[ERROR] Generate request to {host}:{port} failed: {e.code} {body}")
        return {"error": {"code": e.code, "message": body}}
    except Exception as e:
        print(f"[ERROR] Generate request to {host}:{port} failed: {e}")
        return {"error": {"message": str(e)}}


def wait_for_health(host: str, port: int, timeout: int = 120) -> bool:
    """Wait for a server's /health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description="PD Disaggregation Coordinator")
    parser.add_argument("--prefill-host", default="10.60.23.70")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-host", default="10.60.30.66")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--text", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-wait", action="store_true", help="Skip health check wait")
    args = parser.parse_args()

    prefill_addr = f"{args.prefill_host}:{args.prefill_port}"
    decode_addr = f"{args.decode_host}:{args.decode_port}"

    print(f"=== PD Coordinator ===")
    print(f"Prefill: http://{prefill_addr}")
    print(f"Decode:  http://{decode_addr}")
    print(f"Bootstrap: {args.prefill_host}:{args.bootstrap_port}")
    print(f"Text: {args.text!r}")
    print()

    # Step 1: Wait for health
    if not args.no_wait:
        print("[1/4] Waiting for Prefill health...", end=" ", flush=True)
        if not wait_for_health(args.prefill_host, args.prefill_port):
            print("FAILED")
            sys.exit(1)
        print("OK")

        print("[2/4] Waiting for Decode health...", end=" ", flush=True)
        if not wait_for_health(args.decode_host, args.decode_port):
            print("FAILED")
            sys.exit(1)
        print("OK")
    else:
        print("[1/4] Skipping health checks")

    # Step 2: Query bootstrap server for prefill info
    print("[2/4] Querying bootstrap server...", end=" ", flush=True)
    server_info = query_bootstrap_server(args.prefill_host, args.bootstrap_port)
    print(f"OK (tp={server_info.get('attn_tp_size')}, pp={server_info.get('pp_size')}, page_size={server_info.get('page_size')})")

    # Step 3: Generate bootstrap room (Router assigns this, not the bootstrap server)
    print("[3/4] Generating bootstrap room...", end=" ", flush=True)
    # Use timestamp-based unique integer as bootstrap_room (same as Router does)
    bootstrap_room = int(time.time() * 1000000) % (2**63)
    print(f"OK (room={bootstrap_room})")

    # Generate a unique request ID
    import uuid
    rid = str(uuid.uuid4())[:32]

    # Step 4: Send requests concurrently
    print(f"[4/4] Sending requests (rid={rid})...")

    import threading
    results = {}
    errors = []

    def send_prefill():
        print(f"  -> Prefill: sending...", flush=True)
        t0 = time.time()
        result = send_generate_request(
            args.prefill_host, args.prefill_port,
            args.text, args.max_new_tokens,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )
        elapsed = time.time() - t0
        results["prefill"] = result
        if "error" in result:
            errors.append(f"Prefill: {result['error']}")
            print(f"  <- Prefill: ERROR ({elapsed:.1f}s) {result['error']}", flush=True)
        else:
            text = result[0].get("text", "")[:50] if result else "(empty)"
            print(f"  <- Prefill: OK ({elapsed:.1f}s) text={text!r}", flush=True)

    def send_decode():
        print(f"  -> Decode: sending...", flush=True)
        t0 = time.time()
        result = send_generate_request(
            args.decode_host, args.decode_port,
            args.text, args.max_new_tokens,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )
        elapsed = time.time() - t0
        results["decode"] = result
        if "error" in result:
            errors.append(f"Decode: {result['error']}")
            print(f"  <- Decode: ERROR ({elapsed:.1f}s) {result['error']}", flush=True)
        else:
            text = result[0].get("text", "")[:100] if result else "(empty)"
            print(f"  <- Decode: OK ({elapsed:.1f}s) text={text!r}", flush=True)

    t1 = threading.Thread(target=send_prefill)
    t2 = threading.Thread(target=send_decode)
    t1.start()
    t2.start()
    t1.join(timeout=args.timeout + 10)
    t2.join(timeout=args.timeout + 10)

    print()
    if errors:
        print("=== FAILED ===")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("=== SUCCESS ===")
        if "decode" in results:
            decoded = results["decode"]
            for item in decoded:
                print(f"  Text: {item.get('text', '(empty)')}")
                meta = item.get("meta_info", {})
                print(f"  Tokens: prompt={meta.get('prompt_tokens')}, completion={meta.get('completion_tokens')}")
                print(f"  Finish: {meta.get('finish_reason')}")
        elif "prefill" in results:
            for item in results["prefill"]:
                print(f"  Text (prefill only): {item.get('text', '(empty)')}")


if __name__ == "__main__":
    main()
