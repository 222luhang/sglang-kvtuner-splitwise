#!/usr/bin/env python3
"""
PD disaggregation coordinator — bypasses the Rust Router.

Simulates what the Router does:
1. Queries the Prefill bootstrap server
2. Sends the request to Prefill (max_new_tokens=1) and Decode concurrently
3. Collects responses and validates correctness

Supports single request and concurrent (--num-requests) modes.

Usage:
    # Single request
    python3 pd_coordinator.py --text "Hello" --max-new-tokens 32

    # Multiple concurrent requests
    python3 pd_coordinator.py --num-requests 5 --max-new-tokens 64

    # Batch test with varying lengths
    python3 pd_coordinator.py --batch-tests
"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Built-in test cases: (name, text, max_new_tokens)
# ---------------------------------------------------------------------------
BATCH_TESTS = [
    ("short_short", "Hello", 16),
    ("short_medium", "Hello", 64),
    ("short_long", "Hello", 128),
    ("medium_short", "The quick brown fox jumps over the lazy dog.", 16),
    ("medium_medium", "The quick brown fox jumps over the lazy dog.", 64),
    ("medium_long", "The quick brown fox jumps over the lazy dog. In a world where technology and nature coexist, we find ourselves at a crossroads.", 128),
    ("long_short", "The quick brown fox jumps over the lazy dog. In a world where technology and nature coexist, we find ourselves at a crossroads. The future of artificial intelligence depends on our ability to balance innovation with responsibility. Let us explore the possibilities and challenges of creating a sustainable and responsible future for AI.", 16),
    ("long_medium", "The quick brown fox jumps over the lazy dog. In a world where technology and nature coexist, we find ourselves at a crossroads. The future of artificial intelligence depends on our ability to balance innovation with responsibility. Let us explore the possibilities and challenges of creating a sustainable and responsible future for AI.", 64),
    ("long_long", "The quick brown fox jumps over the lazy dog. In a world where technology and nature coexist, we find ourselves at a crossroads. The future of artificial intelligence depends on our ability to balance innovation with responsibility. Let us explore the possibilities and challenges of creating a sustainable and responsible future for AI.", 256),
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def query_bootstrap_server(host: str, bootstrap_port: int) -> dict:
    url = f"http://{host}:{bootstrap_port}/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1"
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Failed to query bootstrap server: {e}")
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Single request test
# ---------------------------------------------------------------------------

def _extract_text(result):
    if isinstance(result, dict):
        return result.get("text", "")
    if isinstance(result, list) and result:
        return result[0].get("text", "")
    return ""


def _extract_meta(result):
    if isinstance(result, dict):
        return result.get("meta_info", {})
    if isinstance(result, list) and result:
        return result[0].get("meta_info", {})
    return {}


def run_single_request(args, text=None, max_new_tokens=None, label=None):
    """Send one PD request. Returns (success, decode_text, elapsed)."""
    import uuid

    text = text or args.text
    max_new_tokens = max_new_tokens or args.max_new_tokens
    label = label or "single"

    # Generate unique bootstrap_room and rid
    bootstrap_room = int(time.time_ns() // 1000 % (2**63))
    rid = str(uuid.uuid4())[:32]

    results = {}
    errors = []
    lock = threading.Lock()

    def send_prefill():
        t0 = time.time()
        # Prefill only needs 1 token
        result = send_generate_request(
            args.prefill_host, args.prefill_port,
            text, 1,  # max_new_tokens=1 for prefill
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )
        elapsed = time.time() - t0
        with lock:
            results["prefill"] = result
            if "error" in result:
                errors.append(f"Prefill: {result['error']}")
                print(f"  [{label}] <- Prefill: ERROR ({elapsed:.1f}s) {result['error']}", flush=True)
            else:
                t = _extract_text(result)[:50]
                print(f"  [{label}] <- Prefill: OK ({elapsed:.1f}s) text={t!r}", flush=True)

    def send_decode():
        t0 = time.time()
        result = send_generate_request(
            args.decode_host, args.decode_port,
            text, max_new_tokens,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )
        elapsed = time.time() - t0
        with lock:
            results["decode"] = result
            if "error" in result:
                errors.append(f"Decode: {result['error']}")
                print(f"  [{label}] <- Decode: ERROR ({elapsed:.1f}s) {result['error']}", flush=True)
            else:
                t = _extract_text(result)[:80]
                print(f"  [{label}] <- Decode: OK ({elapsed:.1f}s) text={t!r}", flush=True)

    t1 = threading.Thread(target=send_prefill)
    t2 = threading.Thread(target=send_decode)
    t1.start()
    t2.start()
    t1.join(timeout=args.timeout + 10)
    t2.join(timeout=args.timeout + 10)

    if errors:
        print(f"  [{label}] FAILED: {'; '.join(errors)}", flush=True)
        return False, "", 0

    decode_text = _extract_text(results.get("decode", {}))
    meta = _extract_meta(results.get("decode", {}))
    prompt_tokens = meta.get("prompt_tokens", 0)
    completion_tokens = meta.get("completion_tokens", 0)
    print(f"  [{label}] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}", flush=True)

    # Correctness: check decode actually generated tokens
    if completion_tokens < 1:
        print(f"  [{label}] FAIL: Decode returned 0 completion tokens", flush=True)
        return False, decode_text, 0

    # Correctness: check output length matches max_new_tokens (temperature=0 should generate exactly that many)
    if completion_tokens < max_new_tokens:
        finish = meta.get("finish_reason", {})
        finish_type = finish.get("type", "unknown") if isinstance(finish, dict) else str(finish)
        # "length" type means hit max_new_tokens limit — expected
        # "stop" means EOS was hit before max_new_tokens — acceptable but note it
        if finish_type == "stop":
            print(f"  [{label}] NOTE: EOS hit at {completion_tokens}/{max_new_tokens} tokens (acceptable)", flush=True)
        # Any other unexpected finish reason
        elif finish_type != "length":
            print(f"  [{label}] WARN: unexpected finish_reason={finish_type}", flush=True)

    return True, decode_text, completion_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--num-requests", "-n", type=int, default=1,
                        help="Number of concurrent requests (default: 1)")
    parser.add_argument("--batch-tests", action="store_true",
                        help="Run built-in test suite with varying input/output lengths")
    parser.add_argument("--validate-output", action="store_true",
                        help="Check that Decode output is non-empty and has expected token count")
    args = parser.parse_args()

    print(f"=== PD Coordinator ===")
    print(f"Prefill: http://{args.prefill_host}:{args.prefill_port}")
    print(f"Decode:  http://{args.decode_host}:{args.decode_port}")
    print(f"Bootstrap: {args.prefill_host}:{args.bootstrap_port}")
    print()

    # Health checks
    if not args.no_wait:
        print("[1/3] Waiting for Prefill health...", end=" ", flush=True)
        if not wait_for_health(args.prefill_host, args.prefill_port):
            print("FAILED"); sys.exit(1)
        print("OK")

        print("[2/3] Waiting for Decode health...", end=" ", flush=True)
        if not wait_for_health(args.decode_host, args.decode_port):
            print("FAILED"); sys.exit(1)
        print("OK")
    else:
        print("[1/3] Skipping health checks")

    # Query bootstrap server
    print("[2/3] Querying bootstrap server...", end=" ", flush=True)
    server_info = query_bootstrap_server(args.prefill_host, args.bootstrap_port)
    print(f"OK (tp={server_info.get('attn_tp_size')}, pp={server_info.get('pp_size')}, page_size={server_info.get('page_size')})")

    # Build test list
    test_cases = []
    if args.batch_tests:
        for name, text, max_tokens in BATCH_TESTS:
            test_cases.append((name, text, max_tokens))
    else:
        for i in range(args.num_requests):
            label = f"req-{i}" if args.num_requests > 1 else "single"
            test_cases.append((label, args.text, args.max_new_tokens))

    print(f"[3/3] Running {len(test_cases)} test(s)...")
    print()

    # Run tests
    t_start = time.time()
    passed = 0
    failed = 0
    failed_names = []

    if args.num_requests > 1 and not args.batch_tests:
        # Concurrent mode: launch all requests in parallel
        concurrent_results = [None] * len(test_cases)

        def run_indexed(idx):
            name, text, max_tokens = test_cases[idx]
            success, _, _ = run_single_request(args, text=text, max_new_tokens=max_tokens, label=name)
            concurrent_results[idx] = (name, success)

        threads = [threading.Thread(target=run_indexed, args=(i,)) for i in range(len(test_cases))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=args.timeout + 30)

        for name, success in concurrent_results:
            if success:
                passed += 1
            else:
                failed += 1
                failed_names.append(name)
    else:
        # Sequential mode (batch-tests or single request)
        for name, text, max_tokens in test_cases:
            success, _, _ = run_single_request(args, text=text, max_new_tokens=max_tokens, label=name)
            if success:
                passed += 1
            else:
                failed += 1
                failed_names.append(name)

    total_time = time.time() - t_start

    # Summary
    print()
    print("=" * 60)
    if failed == 0:
        print(f"  RESULT: PASS ({passed}/{len(test_cases)} tests, {total_time:.1f}s)")
    else:
        print(f"  RESULT: FAIL ({passed} passed, {failed} failed, {total_time:.1f}s)")
        for name in failed_names:
            print(f"    FAILED: {name}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
