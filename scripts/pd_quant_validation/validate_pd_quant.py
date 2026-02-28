#!/usr/bin/env python3
"""
validate_pd_quant.py - P/D 分离与量化结合验证脚本

此脚本提供详细的 P/D 分离和 KVTuner 量化功能验证。
"""

import argparse
import json
import time
import statistics
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("错误：需要安装 requests 库")
    print("运行：pip install requests")
    sys.exit(1)


@dataclass
class TestResult:
    """测试结果数据类"""
    name: str
    passed: bool
    message: str
    metrics: Optional[Dict] = None


@dataclass
class ValidationReport:
    """验证报告数据类"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    results: List[TestResult]
    performance_metrics: Optional[Dict] = None


class PDQuantValidator:
    """P/D 分离与量化验证器"""

    def __init__(
        self,
        prefill_url: str = "http://localhost:30000",
        decode_url: str = "http://localhost:30001",
        router_url: str = "http://localhost:8000",
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        timeout: int = 60,
    ):
        self.prefill_url = prefill_url
        self.decode_url = decode_url
        self.router_url = router_url
        self.model_path = model_path
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout

    def _get(self, url: str, path: str) -> Optional[Dict]:
        """发送 GET 请求"""
        try:
            response = self.session.get(f"{url}{path}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None

    def _post(self, url: str, path: str, data: Dict) -> Optional[Dict]:
        """发送 POST 请求"""
        try:
            response = self.session.post(
                f"{url}{path}",
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None

    def test_health(self) -> TestResult:
        """测试 1: 健康检查"""
        results = []

        for name, url in [
            ("Prefill", self.prefill_url),
            ("Decode", self.decode_url),
            ("Router", self.router_url),
        ]:
            try:
                resp = self.session.get(f"{url}/health", timeout=5)
                if resp.status_code == 200:
                    results.append(f"✓ {name} 健康")
                else:
                    results.append(f"✗ {name} 不健康 (HTTP {resp.status_code})")
            except Exception as e:
                results.append(f"✗ {name} 无法连接：{e}")

        passed = all("✓" in r for r in results)
        return TestResult(
            name="健康检查",
            passed=passed,
            message="\n".join(results),
        )

    def test_server_info(self) -> TestResult:
        """测试 2: 服务器信息验证"""
        metrics = {}
        results = []

        for name, url, mode in [
            ("Prefill", self.prefill_url, "prefill"),
            ("Decode", self.decode_url, "decode"),
        ]:
            info = self._get(url, "/get_server_info")
            if info:
                actual_mode = info.get("disaggregation_mode", "unknown")
                kv_dtype = info.get("kv_cache_dtype", "unknown")
                enable_kvtuner = info.get("enable_kvtuner_quant", False)

                metrics[name.lower()] = {
                    "mode": actual_mode,
                    "kv_cache_dtype": kv_dtype,
                    "enable_kvtuner": enable_kvtuner,
                }

                if actual_mode == mode:
                    results.append(f"✓ {name} 模式正确：{actual_mode}")
                else:
                    results.append(f"✗ {name} 模式错误：期望 {mode}, 实际 {actual_mode}")

                results.append(f"  KV Cache 量化：{kv_dtype}")
                results.append(f"  KVTuner: {enable_kvtuner}")
            else:
                results.append(f"✗ {name} 无法获取服务器信息")

        return TestResult(
            name="服务器信息验证",
            passed=all("✓" in r for r in results),
            message="\n".join(results),
            metrics=metrics,
        )

    def test_quantization_config(self) -> TestResult:
        """测试 3: 量化配置验证"""
        results = []

        prefill_info = self._get(self.prefill_url, "/get_server_info")
        decode_info = self._get(self.decode_url, "/get_server_info")

        if prefill_info and decode_info:
            # 检查 KV Cache 量化
            prefill_kv = prefill_info.get("kv_cache_dtype")
            decode_kv = decode_info.get("kv_cache_dtype")

            if prefill_kv and decode_kv:
                results.append(f"✓ KV Cache 量化已启用")
                results.append(f"  Prefill: {prefill_kv}")
                results.append(f"  Decode: {decode_kv}")
            else:
                results.append("⚠ KV Cache 量化未启用或未报告")

            # 检查 KVTuner
            prefill_kvtuner = prefill_info.get("enable_kvtuner_quant")
            decode_kvtuner = decode_info.get("enable_kvtuner_quant")

            if prefill_kvtuner or decode_kvtuner:
                results.append(f"✓ KVTuner 量化已启用")
            else:
                results.append("⚠ KVTuner 量化未启用或未报告")
        else:
            results.append("✗ 无法获取服务器信息")

        return TestResult(
            name="量化配置验证",
            passed=any("✓" in r for r in results),
            message="\n".join(results),
        )

    def test_basic_inference(self) -> TestResult:
        """测试 4: 基本推理测试"""
        test_prompts = [
            "Hello, how are you?",
            "请介绍一下你自己。",
            "What is the capital of France?",
        ]

        results = []
        success_count = 0

        for prompt in test_prompts:
            response = self._post(
                self.router_url,
                "/v1/completions",
                {
                    "model": self.model_path,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0.7,
                },
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0].get("text", "")
                results.append(f"✓ 提示：{prompt[:30]}...")
                results.append(f"  响应：{text[:50]}...")
                success_count += 1
            else:
                results.append(f"✗ 提示：{prompt[:30]}... - 失败")

        return TestResult(
            name="基本推理测试",
            passed=success_count > 0,
            message=f"成功：{success_count}/{len(test_prompts)}\n" + "\n".join(results),
            metrics={"success_rate": success_count / len(test_prompts)},
        )

    def test_pd_communication(self) -> TestResult:
        """测试 5: P/D 通信验证"""
        results = []

        # 通过 Router 发送请求，验证 P/D 通信
        prompt = "Test P/D communication with a short response."
        start_time = time.time()

        response = self._post(
            self.router_url,
            "/v1/completions",
            {
                "model": self.model_path,
                "prompt": prompt,
                "max_tokens": 16,
                "temperature": 0.0,
            },
        )

        latency = time.time() - start_time

        if response and "choices" in response:
            results.append(f"✓ P/D 通信成功")
            results.append(f"  延迟：{latency:.3f}s")
            results.append(f"  响应：{response['choices'][0].get('text', '')[:50]}")
        else:
            results.append(f"✗ P/D 通信失败")
            if response:
                results.append(f"  错误：{response}")

        return TestResult(
            name="P/D 通信验证",
            passed=response is not None and "choices" in response,
            message="\n".join(results),
            metrics={"latency": latency},
        )

    def test_performance(
        self,
        num_requests: int = 10,
        max_concurrency: int = 4,
    ) -> TestResult:
        """测试 6: 性能基准测试"""
        results = []
        latencies = []
        throughputs = []
        success_count = 0

        prompt = "The quick brown fox jumps over the lazy dog. " * 10

        def send_request():
            start = time.time()
            response = self._post(
                self.router_url,
                "/v1/completions",
                {
                    "model": self.model_path,
                    "prompt": prompt,
                    "max_tokens": 64,
                    "temperature": 0.7,
                },
            )
            latency = time.time() - start

            if response and "choices" in response:
                text = response["choices"][0].get("text", "")
                tokens = len(text.split())
                return {
                    "success": True,
                    "latency": latency,
                    "tokens": tokens,
                    "throughput": tokens / latency if latency > 0 else 0,
                }
            return {"success": False, "latency": latency}

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    success_count += 1
                    latencies.append(result["latency"])
                    throughputs.append(result["throughput"])

        if latencies:
            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]
            avg_throughput = statistics.mean(throughputs)

            results.append(f"成功：{success_count}/{num_requests}")
            results.append(f"平均延迟：{avg_latency:.3f}s")
            results.append(f"P50 延迟：{p50_latency:.3f}s")
            results.append(f"P99 延迟：{p99_latency:.3f}s")
            results.append(f"平均吞吐：{avg_throughput:.2f} tokens/s")

            metrics = {
                "success_rate": success_count / num_requests,
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p99_latency": p99_latency,
                "avg_throughput": avg_throughput,
            }
        else:
            results.append("所有请求失败")
            metrics = {"success_rate": 0}

        return TestResult(
            name="性能基准测试",
            passed=success_count > 0,
            message="\n".join(results),
            metrics=metrics,
        )

    def run_all_tests(self, include_performance: bool = True) -> ValidationReport:
        """运行所有测试"""
        results = []

        print("=" * 60)
        print("P/D 分离与量化验证测试")
        print("=" * 60)
        print()

        # 测试 1: 健康检查
        print("[1/6] 健康检查...")
        result = self.test_health()
        results.append(result)
        print(f"  {'✓ 通过' if result.passed else '✗ 失败'}: {result.message.split(chr(10))[0]}")
        print()

        # 测试 2: 服务器信息
        print("[2/6] 服务器信息验证...")
        result = self.test_server_info()
        results.append(result)
        print(f"  {'✓ 通过' if result.passed else '✗ 失败'}")
        print()

        # 测试 3: 量化配置
        print("[3/6] 量化配置验证...")
        result = self.test_quantization_config()
        results.append(result)
        print(f"  {'✓ 通过' if result.passed else '✗ 失败'}")
        print()

        # 测试 4: 基本推理
        print("[4/6] 基本推理测试...")
        result = self.test_basic_inference()
        results.append(result)
        print(f"  {'✓ 通过' if result.passed else '✗ 失败'}")
        print()

        # 测试 5: P/D 通信
        print("[5/6] P/D 通信验证...")
        result = self.test_pd_communication()
        results.append(result)
        print(f"  {'✓ 通过' if result.passed else '✗ 失败'}")
        print()

        # 测试 6: 性能测试
        performance_metrics = None
        if include_performance:
            print("[6/6] 性能基准测试...")
            result = self.test_performance()
            results.append(result)
            print(f"  {'✓ 通过' if result.passed else '✗ 失败'}")
            performance_metrics = result.metrics
            print()

        # 生成报告
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        report = ValidationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(results),
            passed=passed,
            failed=failed,
            results=results,
            performance_metrics=performance_metrics,
        )

        # 打印报告
        print("=" * 60)
        print("验证报告")
        print("=" * 60)
        print(f"时间：{report.timestamp}")
        print(f"通过：{passed}/{len(results)}")
        print(f"失败：{failed}/{len(results)}")
        print()

        if failed == 0:
            print("✓ 所有测试通过！")
        else:
            print("✗ 部分测试失败，请检查日志")

        return report


def main():
    parser = argparse.ArgumentParser(description="P/D 分离与量化验证脚本")
    parser.add_argument(
        "--prefill-url",
        default="http://localhost:30000",
        help="Prefill 服务 URL",
    )
    parser.add_argument(
        "--decode-url",
        default="http://localhost:30001",
        help="Decode 服务 URL",
    )
    parser.add_argument(
        "--router-url",
        default="http://localhost:8000",
        help="Router URL",
    )
    parser.add_argument(
        "--model-path",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="模型路径",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="请求超时时间（秒）",
    )
    parser.add_argument(
        "--no-performance",
        action="store_true",
        help="跳过性能测试",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出报告文件路径",
    )

    args = parser.parse_args()

    validator = PDQuantValidator(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        router_url=args.router_url,
        model_path=args.model_path,
        timeout=args.timeout,
    )

    report = validator.run_all_tests(include_performance=not args.no_performance)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "total_tests": report.total_tests,
                    "passed": report.passed,
                    "failed": report.failed,
                    "results": [asdict(r) for r in report.results],
                    "performance_metrics": report.performance_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n报告已保存到：{args.output}")

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
