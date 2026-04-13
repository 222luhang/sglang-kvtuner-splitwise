#!/usr/bin/env python3
"""
TTFT (Time To First Token) benchmark for PD disaggregated inference.

Measures TTFT by sending streaming requests to the decode endpoint and timing
the first SSE chunk that contains token content.  Also records end-to-end
latency and token counts.

For per-layer timing breakdown (quant/send/recv/dequant), see the [TIMING]
and [TIMING_SUMMARY] logs emitted by conn.py on the server side.

Usage:
    # Single config, all input lengths, 5 runs each
    python3 eval/bench_ttft.py \
        --prefill-host 10.60.23.70 --decode-host 10.60.30.66 \
        --config-name baseline --num-runs 5 \
        --output results/exp1_ttft/baseline.csv

    # Specific input lengths only
    python3 eval/bench_ttft.py \
        --config-name quant-8bit --input-types short,long \
        --output results/exp1_ttft/quant8.csv

    # Custom prompt with explicit token target
    python3 eval/bench_ttft.py \
        --config-name baseline --prompt-tokens 256 --max-new-tokens 32 \
        --num-runs 3 --output results/custom.csv
"""

import argparse
import csv
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Input matrix (from thesis_test_plan.md section 3.1)
# ---------------------------------------------------------------------------
INPUT_MATRIX = {
    "tiny":   {"target_tokens": 1,    "max_new_tokens": 32,  "label": "1 token"},
    "short":  {"target_tokens": 10,   "max_new_tokens": 64,  "label": "10 tokens"},
    "medium": {"target_tokens": 60,   "max_new_tokens": 64,  "label": "60 tokens"},
    "long":   {"target_tokens": 256,  "max_new_tokens": 32,  "label": "256 tokens"},
    "xlong":  {"target_tokens": 512,  "max_new_tokens": 32,  "label": "512 tokens"},
    "xxlong": {"target_tokens": 1024, "max_new_tokens": 16,  "label": "1024 tokens"},
}

# Seed text repeated to reach target token counts (~1.3 tokens per word)
_SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a world where technology and nature coexist, "
    "we find ourselves at a crossroads between innovation and tradition. "
    "The future of artificial intelligence depends on our ability to balance "
    "progress with responsibility. Let us explore the possibilities and "
    "challenges that lie ahead as we navigate this complex landscape. "
)


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt that approximates the target token count.

    Uses a simple heuristic: ~1.3 tokens per whitespace-separated word.
    The actual token count is reported by the server in meta_info.
    """
    if target_tokens <= 1:
        return "Hi"
    target_words = int(target_tokens / 1.3)
    words = _SEED_TEXT.split()
    result_words = []
    while len(result_words) < target_words:
        result_words.extend(words)
    return " ".join(result_words[:target_words])


# ---------------------------------------------------------------------------
# HTTP helpers


def _wait_server_ready(args, timeout: int = 30) -> None:
    """Block until both prefill and decode servers respond to /health."""
    deadline = time.time() + timeout
    ok = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://{args.prefill_host}:{args.prefill_port}/health", timeout=2
            )
            urllib.request.urlopen(
                f"http://{args.decode_host}:{args.decode_port}/health", timeout=2
            )
            ok = True
            break
        except Exception:
            time.sleep(1)
    if not ok:
        print("  [WARN] servers did not become healthy within timeout", flush=True)
# ---------------------------------------------------------------------------


def wait_for_health(host: str, port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def send_prefill_request(
    host: str, port: int, text: str,
    bootstrap_host: str, bootstrap_port: int, bootstrap_room: int,
    rid: str, timeout: int = 120,
) -> dict:
    """Send prefill request (max_new_tokens=1, non-streaming)."""
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "rid": rid,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_decode_streaming(
    host: str, port: int, text: str, max_new_tokens: int,
    bootstrap_host: str, bootstrap_port: int, bootstrap_room: int,
    rid: str, timeout: int = 120,
) -> dict:
    """Send decode request with streaming and measure TTFT.

    Returns dict with keys: ttft_ms, e2e_ms, prompt_tokens,
    completion_tokens, output_text, success, error.
    """
    result = {
        "ttft_ms": 0.0, "e2e_ms": 0.0,
        "prompt_tokens": 0, "completion_tokens": 0,
        "output_text": "", "success": False, "error": None,
    }

    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        "rid": rid,
        "stream": True,
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
    }
    data = json.dumps(payload).encode()

    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/generate", body=data,
                     headers={"Content-Type": "application/json"})
        t_request = time.perf_counter()
        resp = conn.getresponse()

        if resp.status != 200:
            result["error"] = f"HTTP {resp.status}: {resp.read().decode()}"
            return result

        ttft_recorded = False
        chunks_text = []
        meta_info = {}

        # Read SSE stream
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            # Process complete SSE lines
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"data: "):
                        json_str = line[6:].decode()
                        if json_str.strip() == "[DONE]":
                            continue
                        try:
                            obj = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
                        text_chunk = obj.get("text", "")
                        if text_chunk and not ttft_recorded:
                            result["ttft_ms"] = (time.perf_counter() - t_request) * 1000
                            ttft_recorded = True
                        chunks_text.append(text_chunk)
                        if "meta_info" in obj:
                            meta_info = obj["meta_info"]

        result["e2e_ms"] = (time.perf_counter() - t_request) * 1000
        result["output_text"] = "".join(chunks_text)
        result["prompt_tokens"] = meta_info.get("prompt_tokens", 0)
        result["completion_tokens"] = meta_info.get("completion_tokens", 0)
        result["success"] = ttft_recorded and result["completion_tokens"] > 0
        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Single PD request with TTFT measurement
# ---------------------------------------------------------------------------


def run_one_request(args, text: str, max_new_tokens: int, label: str) -> dict:
    """Run one PD request and return timing results."""
    bootstrap_room = int(time.time_ns() // 1000 % (2**63))
    rid = uuid.uuid4().hex[:24]

    prefill_result = [None]
    decode_result = [None]

    def do_prefill():
        prefill_result[0] = send_prefill_request(
            args.prefill_host, args.prefill_port, text,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )

    def do_decode():
        decode_result[0] = send_decode_streaming(
            args.decode_host, args.decode_port, text, max_new_tokens,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )

    t1 = threading.Thread(target=do_prefill)
    t2 = threading.Thread(target=do_decode)
    t1.start()
    t2.start()
    t1.join(timeout=args.timeout + 10)
    t2.join(timeout=args.timeout + 10)

    dr = decode_result[0] or {"success": False, "error": "timeout"}
    pr = prefill_result[0] or {}

    if dr.get("error"):
        print(f"  [{label}] FAILED: {dr['error']}", flush=True)
    elif dr["success"]:
        print(
            f"  [{label}] TTFT={dr['ttft_ms']:.1f}ms  E2E={dr['e2e_ms']:.1f}ms  "
            f"prompt={dr['prompt_tokens']}  completion={dr['completion_tokens']}",
            flush=True,
        )
    else:
        print(f"  [{label}] FAILED: no tokens received", flush=True)

    return dr


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(args):
    """Run the full benchmark matrix and write CSV output."""
    # Determine input types to test
    if args.prompt_tokens is not None:
        # Custom single input
        input_types = {
            "custom": {
                "target_tokens": args.prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "label": f"{args.prompt_tokens} tokens",
            }
        }
    elif args.input_types:
        names = [s.strip() for s in args.input_types.split(",")]
        input_types = {k: v for k, v in INPUT_MATRIX.items() if k in names}
        if not input_types:
            print(f"[ERROR] No valid input types in: {args.input_types}")
            print(f"  Available: {', '.join(INPUT_MATRIX.keys())}")
            sys.exit(1)
    else:
        input_types = INPUT_MATRIX

    # Health checks
    if not args.no_wait:
        print("Waiting for servers...", end=" ", flush=True)
        if not wait_for_health(args.prefill_host, args.prefill_port):
            print("Prefill FAILED")
            sys.exit(1)
        if not wait_for_health(args.decode_host, args.decode_port):
            print("Decode FAILED")
            sys.exit(1)
        print("OK")

    total_tests = len(input_types) * args.num_runs
    print(f"\nConfig: {args.config_name}")
    print(f"Input types: {', '.join(input_types.keys())}")
    print(f"Runs per input: {args.num_runs}")
    print(f"Total requests: {total_tests}")
    print()

    rows = []
    for input_name, input_cfg in input_types.items():
        target_tokens = input_cfg["target_tokens"]
        max_new = input_cfg["max_new_tokens"]
        prompt = generate_prompt(target_tokens)

        print(f"--- {input_name} ({input_cfg['label']}, max_new={max_new}) ---")

        for run_id in range(1, args.num_runs + 1):
            label = f"{input_name}/run{run_id}"
            dr = run_one_request(args, prompt, max_new, label)

            rows.append({
                "config": args.config_name,
                "input_type": input_name,
                "prompt_tokens": dr.get("prompt_tokens", 0),
                "max_new_tokens": max_new,
                "run_id": run_id,
                "ttft_ms": f"{dr['ttft_ms']:.2f}" if dr["success"] else "",
                "e2e_ms": f"{dr['e2e_ms']:.2f}" if dr["success"] else "",
                "completion_tokens": dr.get("completion_tokens", 0),
                "success": str(dr["success"]).lower(),
                "error": dr.get("error", "") or "",
            })

            # Wait for server readiness before next request.
            # The prefill+decode threads have already joined above, but the
            # server may still be cleaning up internal state (KV pools, TCP
            # bootstrap sockets, etc.).  A health check ensures the server is
            # truly idle and ready for the next request.
            if run_id < args.num_runs:
                if args.health_check:
                    _wait_server_ready(args)

        print()

    # Write CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fieldnames = [
            "config", "input_type", "prompt_tokens", "max_new_tokens",
            "run_id", "ttft_ms", "e2e_ms", "completion_tokens", "success", "error",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to {args.output}")

    # Print summary
    successful = [r for r in rows if r["success"] == "true"]
    if successful:
        ttfts = [float(r["ttft_ms"]) for r in successful]
        e2es = [float(r["e2e_ms"]) for r in successful]
        print(f"\nSummary ({len(successful)}/{len(rows)} successful):")
        print(f"  TTFT  mean={sum(ttfts)/len(ttfts):.1f}ms  "
              f"min={min(ttfts):.1f}ms  max={max(ttfts):.1f}ms")
        print(f"  E2E   mean={sum(e2es)/len(e2es):.1f}ms  "
              f"min={min(e2es):.1f}ms  max={max(e2es):.1f}ms")
    else:
        print("\nNo successful requests.")

    return 0 if len(successful) == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for PD disaggregated inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prefill-host", default="10.60.23.70")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-host", default="10.60.30.66")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--config-name", default="baseline",
                        help="Label for this config (used in CSV output)")
    parser.add_argument("--input-types",
                        help="Comma-separated input types: tiny,short,medium,long,xlong,xxlong")
    parser.add_argument("--prompt-tokens", type=int,
                        help="Override: use a single custom prompt token count")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Max new tokens (used with --prompt-tokens)")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs per input type")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between runs (default: 1.0)")
    parser.add_argument("--health-check", action="store_true", default=True,
                        help="Health-check both servers between requests (default: True)")
    parser.add_argument("--no-health-check", dest="health_check", action="store_false",
                        help="Skip health checks between requests")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip health check wait")
    parser.add_argument("--output", "-o",
                        help="Output CSV file path")
    args = parser.parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
