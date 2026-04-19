#!/usr/bin/env python3
"""
Run standard LLM evaluation benchmarks through PD disaggregated inference.

Supported benchmarks:
  - gsm8k:     Math reasoning (0-shot CoT, accuracy)
  - mmlu:      Language understanding (0-shot direct, accuracy)
  - hellaswag: Commonsense reasoning (0-shot direct, accuracy)

Usage:
    # GSM8K with 200 samples, 0-shot
    python3 eval/run_benchmark.py \
        --benchmark gsm8k \
        --prefill-host 10.60.23.70 --decode-host 10.60.30.66 \
        --num-samples 200 --output results/gsm8k_baseline.json

    # MMLU with default samples
    python3 eval/run_benchmark.py \
        --benchmark mmlu \
        --prefill-host 10.60.23.70 --decode-host 10.60.30.66 \
        --output results/mmlu_quant8.json

    # Dry run (load data + format prompts, no requests)
    python3 eval/run_benchmark.py --benchmark gsm8k --dry-run
"""

import argparse
import ast
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# PD request helpers (adapted from bench_ttft.py)
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


def _send_generate(host, port, text, max_new_tokens, bootstrap_host,
                   bootstrap_port, bootstrap_room, rid, timeout=120):
    """Send a /generate request (non-streaming) and return parsed JSON."""
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
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": {"code": e.code, "message": e.read().decode()}}
    except Exception as e:
        return {"error": {"message": str(e)}}


def run_pd_request(args, text: str, max_new_tokens: int) -> dict:
    """Send one PD request (prefill + decode in parallel). Returns output text or error."""
    bootstrap_room = int(time.time_ns() // 1000 % (2**63))
    rid = uuid.uuid4().hex[:24]

    prefill_result = [None]
    decode_result = [None]

    def do_prefill():
        prefill_result[0] = _send_generate(
            args.prefill_host, args.prefill_port, text, 1,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )

    def do_decode():
        decode_result[0] = _send_generate(
            args.decode_host, args.decode_port, text, max_new_tokens,
            args.prefill_host, args.bootstrap_port, bootstrap_room,
            rid, timeout=args.timeout,
        )

    t1 = threading.Thread(target=do_prefill)
    t2 = threading.Thread(target=do_decode)
    t1.start(); t2.start()
    t1.join(timeout=args.timeout + 10)
    t2.join(timeout=args.timeout + 10)

    dr = decode_result[0] or {}
    if "error" in dr:
        return {"text": "", "error": str(dr["error"]), "success": False}

    # Extract text from response
    if isinstance(dr, list) and dr:
        dr = dr[0]
    output_text = dr.get("text", "")
    meta = dr.get("meta_info", {})
    return {
        "text": output_text,
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
        "success": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# GSM8K benchmark
# ---------------------------------------------------------------------------

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
_GSM8K_INVALID = -9999999


def _gsm8k_get_answer_value(answer_str):
    """Extract numeric answer from GSM8K response.

    Priority:
    1. Look for '#### <number>' (GSM8K canonical format)
    2. Look for 'the answer is <number>' pattern
    3. Fall back to the last number in the text
    """
    answer_str = answer_str.replace(",", "")

    # Priority 1: GSM8K canonical "#### number"
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_str)
    if m:
        try:
            return ast.literal_eval(m.group(1))
        except (SyntaxError, ValueError):
            pass

    # Priority 2: "the answer is <number>"
    m = re.search(r"(?i)the\s+answer\s+is\s*:?\s*\$?(-?\d+[\d,]*\.?\d*)", answer_str)
    if m:
        try:
            return ast.literal_eval(m.group(1).replace(",", ""))
        except (SyntaxError, ValueError):
            pass

    # Priority 3: last number in text
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if not numbers:
        return _GSM8K_INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return _GSM8K_INVALID


def _gsm8k_format_prompt(question, few_shot_examples, num_shots):
    """Format a GSM8K prompt."""
    if num_shots > 0:
        return few_shot_examples + f"Question: {question}\nAnswer:"
    # 0-shot with CoT guidance
    return (
        f"Question: {question}\n"
        f"Let's solve this step by step, then give the final numeric answer "
        f"after '####'.\nAnswer:"
    )


def load_gsm8k(num_samples, num_shots=0, data_dir=None):
    """Load GSM8K dataset. Returns samples list."""
    # Try local file first, then download
    local_path = os.path.join(data_dir, "gsm8k_test.jsonl") if data_dir else None
    if local_path and os.path.isfile(local_path):
        filename = local_path
    else:
        try:
            from sglang.utils import download_and_cache_file, read_jsonl
            filename = download_and_cache_file(GSM8K_URL)
        except ImportError:
            filename = _download_file(GSM8K_URL, "gsm8k_test.jsonl")

    lines = list(_read_jsonl(filename))

    # Build few-shot prompt from first num_shots examples
    few_shot = ""
    for i in range(min(num_shots, len(lines))):
        few_shot += f"Question: {lines[i]['question']}\nAnswer: {lines[i]['answer']}\n\n"

    # Evaluation data excludes few-shot examples
    eval_lines = lines[num_shots:]
    if num_samples and num_samples < len(eval_lines):
        eval_lines = eval_lines[:num_samples]

    samples = []
    for i, line in enumerate(eval_lines):
        prompt = _gsm8k_format_prompt(line["question"], few_shot, num_shots)
        label = _gsm8k_get_answer_value(line["answer"])
        samples.append({
            "id": i,
            "prompt": prompt,
            "label": label,
            "label_str": line["answer"].split("####")[-1].strip() if "####" in line["answer"] else str(label),
            "max_new_tokens": 1024,
        })

    return samples


def eval_gsm8k(prediction, label):
    """Evaluate a single GSM8K prediction."""
    pred_value = _gsm8k_get_answer_value(prediction)
    # Use tolerance for float comparison
    if isinstance(pred_value, float) or isinstance(label, float):
        correct = abs(pred_value - label) < 1e-3 if pred_value != _GSM8K_INVALID else False
    else:
        correct = pred_value == label
    return {
        "correct": correct,
        "predicted": pred_value,
        "expected": label,
    }


# ---------------------------------------------------------------------------
# MMLU benchmark
# ---------------------------------------------------------------------------

_MMLU_QUERY_TEMPLATE_DIRECT = """{question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer:""".strip()

_MMLU_QUERY_TEMPLATE_COT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}""".strip()

_ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"


def _extract_multichoice(prediction):
    """Extract A/B/C/D from model output with multiple fallback strategies."""
    # Strategy 1: "Answer: X" pattern
    m = re.search(_ANSWER_PATTERN_MULTICHOICE, prediction)
    if m:
        return m.group(1).upper()

    # Strategy 2: standalone letter at start or after newline
    m = re.search(r"(?:^|\n)\s*([A-D])\s*[)\.]", prediction)
    if m:
        return m.group(1).upper()

    # Strategy 3: first single A/B/C/D letter in output
    cleaned = prediction.strip()
    if cleaned and cleaned[0] in "ABCD":
        return cleaned[0]

    # Strategy 4: look for "(X)" pattern
    m = re.search(r"\(([A-D])\)", prediction)
    if m:
        return m.group(1).upper()

    return None


def load_mmlu(num_samples, num_shots=0):
    """Load MMLU dataset from HuggingFace. Returns samples list."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    samples = []
    count = 0
    for item in ds:
        if num_samples and count >= num_samples:
            break
        choices = item["choices"]
        prompt = _MMLU_QUERY_TEMPLATE_DIRECT.format(
            question=item["question"],
            A=choices[0], B=choices[1], C=choices[2], D=choices[3],
        )
        label_idx = item["answer"]  # 0-3
        label_letter = "ABCD"[label_idx]
        samples.append({
            "id": count,
            "prompt": prompt,
            "label": label_letter,
            "label_str": label_letter,
            "subject": item.get("subject", "unknown"),
            "max_new_tokens": 32,
        })
        count += 1

    return samples


def eval_mmlu(prediction, label):
    """Evaluate a single MMLU prediction."""
    extracted = _extract_multichoice(prediction)
    return {
        "correct": extracted == label,
        "predicted": extracted,
        "expected": label,
    }


# ---------------------------------------------------------------------------
# HellaSwag benchmark
# ---------------------------------------------------------------------------

_HELLASWAG_TEMPLATE = """Pick the most plausible continuation.

{context}

A) {A}
B) {B}
C) {C}
D) {D}

Answer:""".strip()


def load_hellaswag(num_samples):
    """Load HellaSwag dataset from HuggingFace. Returns samples list."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")

    samples = []
    count = 0
    for item in ds:
        if num_samples and count >= num_samples:
            break
        endings = item["endings"]
        if len(endings) < 4:
            continue
        context = item.get("ctx", "") or item.get("activity_label", "")
        prompt = _HELLASWAG_TEMPLATE.format(
            context=context,
            A=endings[0], B=endings[1], C=endings[2], D=endings[3],
        )
        label_idx = int(item["label"])
        label_letter = "ABCD"[label_idx]
        samples.append({
            "id": count,
            "prompt": prompt,
            "label": label_letter,
            "label_str": label_letter,
            "max_new_tokens": 32,
        })
        count += 1

    return samples


def eval_hellaswag(prediction, label):
    """Evaluate a single HellaSwag prediction."""
    extracted = _extract_multichoice(prediction)
    return {
        "correct": extracted == label,
        "predicted": extracted,
        "expected": label,
    }


# ---------------------------------------------------------------------------
# Utility: JSONL reader (fallback if sglang.utils not available)
# ---------------------------------------------------------------------------

def _read_jsonl(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _download_file(url, filename):
    """Download a file to a cache directory."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sglang_eval")
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    if not os.path.isfile(filepath):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "gsm8k": {"loader": load_gsm8k, "evaluator": eval_gsm8k},
    "mmlu": {"loader": load_mmlu, "evaluator": eval_mmlu},
    "hellaswag": {"loader": load_hellaswag, "evaluator": eval_hellaswag},
}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args):
    bench = BENCHMARKS[args.benchmark]

    # Load data
    print(f"Loading {args.benchmark} data (num_shots={args.num_shots})...", flush=True)
    if args.benchmark == "gsm8k":
        samples = bench["loader"](args.num_samples, num_shots=args.num_shots, data_dir=args.data_dir)
    elif args.benchmark == "mmlu":
        samples = bench["loader"](args.num_samples, num_shots=args.num_shots)
    else:
        samples = bench["loader"](args.num_samples)

    print(f"Loaded {len(samples)} samples", flush=True)

    if args.dry_run:
        print("\n--- Dry run: showing first 2 prompts ---")
        for s in samples[:2]:
            print(f"\n[Sample {s['id']}] label={s['label_str']}")
            print(f"max_new_tokens={s['max_new_tokens']}")
            print(f"Prompt ({len(s['prompt'])} chars):")
            print(s["prompt"][:500])
            if len(s["prompt"]) > 500:
                print("...")
        print(f"\nTotal samples: {len(samples)}")
        return 0

    # Health checks
    if not args.no_wait:
        print("Waiting for servers...", end=" ", flush=True)
        if not wait_for_health(args.prefill_host, args.prefill_port):
            print("Prefill FAILED"); return 1
        if not wait_for_health(args.decode_host, args.decode_port):
            print("Decode FAILED"); return 1
        print("OK")

    # Run evaluation
    evaluator = bench["evaluator"]
    results = []
    correct = 0
    total = 0
    t_start = time.time()

    def process_sample(sample):
        resp = run_pd_request(args, sample["prompt"], sample["max_new_tokens"])
        if not resp["success"]:
            return {**sample, "output": "", "error": resp["error"],
                    "correct": False, "predicted": None}
        eval_result = evaluator(resp["text"], sample["label"])
        return {
            "id": sample["id"],
            "label": sample["label_str"],
            "output": resp["text"][:500],
            "correct": eval_result["correct"],
            "predicted": eval_result["predicted"],
            "expected": eval_result["expected"],
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "completion_tokens": resp.get("completion_tokens", 0),
            "error": None,
        }

    print(f"\nRunning {len(samples)} requests (parallel={args.parallel}, interval={args.interval}s)...\n")

    if args.parallel <= 1:
        # Sequential mode with interval between requests
        for sample in samples:
            result = process_sample(sample)
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1
            status = "OK" if result["correct"] else "WRONG"
            if result.get("error"):
                status = "ERR"
            print(f"  [{total}/{len(samples)}] id={result['id']} "
                  f"{status} pred={result.get('predicted')} "
                  f"expected={result.get('expected')}", flush=True)
            # Interval between requests to avoid server state pollution
            if total < len(samples):
                time.sleep(args.interval)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_sample, s): s for s in samples}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                total += 1
                if result["correct"]:
                    correct += 1
                status = "OK" if result["correct"] else "WRONG"
                if result.get("error"):
                    status = "ERR"
                print(f"  [{total}/{len(samples)}] id={result['id']} "
                      f"{status} pred={result.get('predicted')} "
                      f"expected={result.get('expected')}", flush=True)

    elapsed = time.time() - t_start
    accuracy = correct / total if total > 0 else 0.0

    # Sort results by id
    results.sort(key=lambda r: r["id"])

    # Build output
    output = {
        "benchmark": args.benchmark,
        "config_name": args.config_name,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "args": {
            "prefill_host": args.prefill_host,
            "decode_host": args.decode_host,
            "num_samples": args.num_samples,
            "parallel": args.parallel,
        },
        "samples": results,
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Config:    {args.config_name}")
    print(f"  Accuracy:  {accuracy:.2%} ({correct}/{total})")
    print(f"  Time:      {elapsed:.1f}s")
    errors = sum(1 for r in results if r.get("error"))
    if errors:
        print(f"  Errors:    {errors}")
    print(f"{'='*50}")

    # Save output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks through PD disaggregated inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARKS.keys()),
                        help="Benchmark to run")
    parser.add_argument("--prefill-host", default="10.60.23.70")
    parser.add_argument("--prefill-port", type=int, default=30000)
    parser.add_argument("--decode-host", default="10.60.30.66")
    parser.add_argument("--decode-port", type=int, default=30001)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--config-name", default="baseline",
                        help="Label for this config (used in output)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples to evaluate")
    parser.add_argument("--num-shots", type=int, default=0,
                        help="Number of few-shot examples (default: 0)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent requests")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-request timeout in seconds")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between sequential requests (default: 2.0)")
    parser.add_argument("--data-dir", default=None,
                        help="Local directory for cached datasets")
    parser.add_argument("--output", "-o",
                        help="Output JSON file path")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip health check wait")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and show prompts without sending requests")
    args = parser.parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
