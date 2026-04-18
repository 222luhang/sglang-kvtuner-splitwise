#!/usr/bin/env python3
"""
Throughput benchmark for P/D disaggregation with transfer quantization.

Tests concurrent request throughput under different bandwidth constraints.
Uses the correct P/D disaggregation flow (prefill + decode in parallel).
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp


@dataclass
class RequestResult:
    success: bool
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str = ""


async def send_prefill_request(
    session: aiohttp.ClientSession,
    prefill_host: str,
    prefill_port: int,
    prompt: str,
    bootstrap_host: str,
    bootstrap_port: int,
    bootstrap_room: int,
    timeout: float = 30.0,
) -> dict:
    """Send prefill request (max_new_tokens=1, non-streaming)."""
    url = f"http://{prefill_host}:{prefill_port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return {"error": f"HTTP {resp.status}"}
    except Exception as e:
        return {"error": str(e)}


async def send_decode_streaming(
    session: aiohttp.ClientSession,
    decode_host: str,
    decode_port: int,
    prompt: str,
    max_new_tokens: int,
    bootstrap_host: str,
    bootstrap_port: int,
    bootstrap_room: int,
    timeout: float = 120.0,
) -> dict:
    """Send decode request with streaming and measure TTFT."""
    url = f"http://{decode_host}:{decode_port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
        "stream": True,
    }

    start_time = time.perf_counter()
    ttft_time = None
    completion_tokens = 0
    prompt_tokens = 0

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}"}

            async for line in resp.content:
                if line:
                    try:
                        line_str = line.decode().strip()
                        if line_str.startswith("data:"):
                            json_str = line_str[6:]
                            data = json.loads(json_str)
                            if ttft_time is None and "text" in data:
                                ttft_time = time.perf_counter()
                            if "meta_info" in data:
                                completion_tokens = data["meta_info"].get("completion_tokens", 0)
                                prompt_tokens = data["meta_info"].get("prompt_tokens", 0)
                    except json.JSONDecodeError:
                        pass

        end_time = time.perf_counter()

        return {
            "success": True,
            "ttft_ms": (ttft_time - start_time) * 1000 if ttft_time else 0,
            "e2e_ms": (end_time - start_time) * 1000,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    except asyncio.TimeoutError:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_single_request(
    session: aiohttp.ClientSession,
    prefill_host: str,
    prefill_port: int,
    decode_host: str,
    decode_port: int,
    prompt: str,
    max_new_tokens: int,
    bootstrap_port: int,
    request_id: int,
    timeout: float,
) -> RequestResult:
    """Run a single P/D disaggregated request."""

    # Generate unique bootstrap_room for this request
    bootstrap_room = int(time.time_ns() // 1000 % (2**63)) + request_id

    # Run prefill and decode in parallel
    prefill_task = send_prefill_request(
        session, prefill_host, prefill_port, prompt,
        prefill_host, bootstrap_port, bootstrap_room, timeout
    )
    decode_task = send_decode_streaming(
        session, decode_host, decode_port, prompt, max_new_tokens,
        prefill_host, bootstrap_port, bootstrap_room, timeout
    )

    prefill_result, decode_result = await asyncio.gather(prefill_task, decode_task)

    # Check for errors
    if prefill_result.get("error"):
        return RequestResult(
            success=False, ttft_ms=0, e2e_ms=0,
            prompt_tokens=0, completion_tokens=0,
            error=f"Prefill: {prefill_result['error']}"
        )

    if not decode_result.get("success"):
        return RequestResult(
            success=False, ttft_ms=0, e2e_ms=0,
            prompt_tokens=0, completion_tokens=0,
            error=decode_result.get("error", "Unknown decode error")
        )

    return RequestResult(
        success=True,
        ttft_ms=decode_result.get("ttft_ms", 0),
        e2e_ms=decode_result.get("e2e_ms", 0),
        prompt_tokens=decode_result.get("prompt_tokens", 0),
        completion_tokens=decode_result.get("completion_tokens", 0),
    )


async def run_concurrent_requests(
    prefill_host: str,
    prefill_port: int,
    decode_host: str,
    decode_port: int,
    bootstrap_port: int,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    timeout: float,
) -> list[RequestResult]:
    """Run concurrent requests."""

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(prompt: str, idx: int):
            async with semaphore:
                return await run_single_request(
                    session, prefill_host, prefill_port, decode_host, decode_port,
                    prompt, max_new_tokens, bootstrap_port, idx, timeout
                )

        tasks = [bounded_request(p, i) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)


def generate_prompts(num_requests: int, prompt_type: str = "medium") -> list[str]:
    """Generate test prompts."""

    templates = {
        "short": "Write a brief summary of AI.",
        "medium": "Explain machine learning concepts including supervised, unsupervised, and reinforcement learning with examples.",
        "long": "Write a comprehensive essay about AI history from 1950s to present, covering milestones, researchers, technologies, and industry impacts.",
    }

    template = templates.get(prompt_type, templates["medium"])
    return [f"[Request {i}] {template}" for i in range(num_requests)]


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for P/D disaggregation")
    parser.add_argument("--prefill-host", default="10.60.23.70")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-host", default="10.60.30.66")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--prompt-type", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--config-name", default="throughput_test")
    parser.add_argument("--output", default="/tmp/throughput_results.json")
    args = parser.parse_args()

    print(f"Throughput Benchmark - {args.config_name}")
    print(f"Concurrency: {args.concurrency}, Requests: {args.num_requests}")
    print(f"Prompt type: {args.prompt_type}, Max new tokens: {args.max_new_tokens}")
    print()

    prompts = generate_prompts(args.num_requests, args.prompt_type)

    print(f"Starting {args.num_requests} requests with concurrency {args.concurrency}...")
    start_time = time.perf_counter()

    results = asyncio.run(
        run_concurrent_requests(
            args.prefill_host, args.prefill_port,
            args.decode_host, args.decode_port,
            args.bootstrap_port,
            prompts, args.max_new_tokens, args.concurrency, args.timeout
        )
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate metrics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in successful)
    total_completion_tokens = sum(r.completion_tokens for r in successful)

    requests_per_sec = len(successful) / total_time
    tokens_per_sec = total_tokens / total_time if total_tokens > 0 else 0
    output_tokens_per_sec = total_completion_tokens / total_time if total_completion_tokens > 0 else 0

    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    e2es = [r.e2e_ms for r in successful]

    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    mean_e2e = sum(e2es) / len(e2es) if e2es else 0
    p50_ttft = sorted(ttfts)[len(ttfts) // 2] if ttfts else 0
    p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 1 else (ttfts[0] if ttfts else 0)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {len(successful)}/{args.num_requests}, Failed: {len(failed)}")
    print(f"\nThroughput:")
    print(f"  Requests/sec: {requests_per_sec:.2f}")
    print(f"  Total tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Output tokens/sec: {output_tokens_per_sec:.2f}")
    print(f"\nLatency:")
    print(f"  Mean TTFT: {mean_ttft:.1f}ms, P50: {p50_ttft:.1f}ms, P99: {p99_ttft:.1f}ms")
    print(f"  Mean E2E: {mean_e2e:.1f}ms")

    if failed:
        print(f"\nErrors:")
        error_counts = {}
        for r in failed:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1
        for error, count in error_counts.items():
            print(f"  {error[:80]}: {count}")

    # Save results
    output_data = {
        "config": args.config_name,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "prompt_type": args.prompt_type,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": {
            "total_time_s": round(total_time, 2),
            "successful": len(successful),
            "failed": len(failed),
            "requests_per_sec": round(requests_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "output_tokens_per_sec": round(output_tokens_per_sec, 2),
            "mean_ttft_ms": round(mean_ttft, 1),
            "p50_ttft_ms": round(p50_ttft, 1),
            "p99_ttft_ms": round(p99_ttft, 1),
            "mean_e2e_ms": round(mean_e2e, 1),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
