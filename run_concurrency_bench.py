"""RAG 系统并发性能评测

测试不同并发级别下的 QPS、延迟分布、错误率等指标。
使用 asyncio + httpx 模拟并发请求。
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import httpx

# ============ 配置 ============

BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

# 测试问题集（模拟真实用户查询）
TEST_QUESTIONS = [
    "Redis的ZSet底层使用什么数据结构？",
    "什么是缓存穿透？怎么解决？",
    "TCP三次握手的过程是什么？",
    "Transformer中Self-Attention的计算公式是什么？",
    "Docker多阶段构建有什么好处？",
    "数据库索引为什么用B+树？",
    "HTTPS的TLS握手过程是怎样的？",
    "Python中GIL是什么？",
    "微服务架构中服务注册与发现有哪些方案？",
    "LoRA微调的原理是什么？",
    "系统设计中如何保证高可用？",
    "缓存和数据库的一致性如何保证？",
]

# 并发级别
CONCURRENCY_LEVELS = [1, 3, 5, 10, 20]

# 每个并发级别的总请求数
REQUESTS_PER_LEVEL = 12


# ============ 数据结构 ============

@dataclass
class RequestResult:
    """单次请求结果"""
    success: bool
    latency_ms: float
    status_code: int = 0
    error: str = ""
    retrieval_time_ms: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """某个并发级别的评测结果"""
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    error_rate: float
    total_duration_s: float
    qps: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    retrieval_avg_ms: float
    generation_avg_ms: float


# ============ 核心逻辑 ============

async def send_request(
    client: httpx.AsyncClient,
    question: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """发送单次请求并记录结果"""
    async with semaphore:
        payload = {
            "question": question,
            "stream": False,
            "use_query_transform": True,
            "use_rerank": True,
        }
        start = time.perf_counter()
        try:
            resp = await client.post(
                CHAT_ENDPOINT,
                json=payload,
                timeout=120.0,
            )
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                return RequestResult(
                    success=True,
                    latency_ms=latency,
                    status_code=200,
                    retrieval_time_ms=data.get("retrieval_detail", {}).get("retrieval_time_ms", 0),
                    total_time_ms=data.get("total_time_ms", 0),
                )
            else:
                return RequestResult(
                    success=False,
                    latency_ms=latency,
                    status_code=resp.status_code,
                    error=f"HTTP {resp.status_code}",
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=False,
                latency_ms=latency,
                error=str(e)[:100],
            )


async def run_benchmark(concurrency: int, num_requests: int) -> BenchmarkResult:
    """在指定并发级别下运行评测"""
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        # 预热：发一个请求确保连接建立
        warmup_q = TEST_QUESTIONS[0]
        await send_request(client, warmup_q, asyncio.Semaphore(1))

        # 构建任务列表（循环使用问题）
        tasks = []
        for i in range(num_requests):
            q = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
            tasks.append(send_request(client, q, semaphore))

        # 并发执行
        start_time = time.perf_counter()
        results: List[RequestResult] = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    # 统计
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = sorted([r.latency_ms for r in successful])

    if not latencies:
        return BenchmarkResult(
            concurrency=concurrency,
            total_requests=num_requests,
            successful=0,
            failed=num_requests,
            error_rate=1.0,
            total_duration_s=total_duration,
            qps=0,
            latency_avg_ms=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            retrieval_avg_ms=0,
            generation_avg_ms=0,
        )

    def percentile(data: List[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    retrieval_times = [r.retrieval_time_ms for r in successful if r.retrieval_time_ms > 0]
    total_times = [r.total_time_ms for r in successful if r.total_time_ms > 0]
    retrieval_avg = statistics.mean(retrieval_times) if retrieval_times else 0
    total_avg = statistics.mean(total_times) if total_times else 0
    generation_avg = total_avg - retrieval_avg if total_avg > retrieval_avg else 0

    return BenchmarkResult(
        concurrency=concurrency,
        total_requests=num_requests,
        successful=len(successful),
        failed=len(failed),
        error_rate=len(failed) / num_requests,
        total_duration_s=round(total_duration, 2),
        qps=round(len(successful) / total_duration, 2),
        latency_avg_ms=round(statistics.mean(latencies), 1),
        latency_p50_ms=round(percentile(latencies, 50), 1),
        latency_p95_ms=round(percentile(latencies, 95), 1),
        latency_p99_ms=round(percentile(latencies, 99), 1),
        latency_min_ms=round(latencies[0], 1),
        latency_max_ms=round(latencies[-1], 1),
        retrieval_avg_ms=round(retrieval_avg, 1),
        generation_avg_ms=round(generation_avg, 1),
    )


async def main():
    print(f"\n{'='*70}")
    print(f"  RAG 系统并发性能评测")
    print(f"  目标: {BASE_URL}")
    print(f"  并发级别: {CONCURRENCY_LEVELS}")
    print(f"  每级请求数: {REQUESTS_PER_LEVEL}")
    print(f"{'='*70}\n")

    # 健康检查
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/api/health", timeout=10)
            if resp.status_code == 200:
                health = resp.json()
                print(f"  ✓ 服务健康: {health.get('status')}, 已索引 {health.get('indexed_documents')} 篇文档\n")
            else:
                print(f"  ✗ 服务异常: HTTP {resp.status_code}")
                return
        except Exception as e:
            print(f"  ✗ 无法连接服务: {e}")
            return

    all_results: List[BenchmarkResult] = []

    for level in CONCURRENCY_LEVELS:
        print(f"  ▶ 并发={level} 测试中...", end="", flush=True)
        result = await run_benchmark(level, REQUESTS_PER_LEVEL)
        all_results.append(result)
        status = "✓" if result.error_rate == 0 else "⚠"
        print(f" {status} QPS={result.qps}, P50={result.latency_p50_ms}ms, P95={result.latency_p95_ms}ms, 错误率={result.error_rate:.1%}")

    # 打印详细报告
    print(f"\n{'='*70}")
    print(f"  详细评测报告")
    print(f"{'='*70}")
    print(f"\n  {'并发':>4} | {'QPS':>6} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'P99(ms)':>8} | {'Avg(ms)':>8} | {'错误率':>6} | {'检索(ms)':>8} | {'生成(ms)':>8}")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for r in all_results:
        print(f"  {r.concurrency:>4} | {r.qps:>6.2f} | {r.latency_p50_ms:>8.1f} | {r.latency_p95_ms:>8.1f} | {r.latency_p99_ms:>8.1f} | {r.latency_avg_ms:>8.1f} | {r.error_rate:>5.1%} | {r.retrieval_avg_ms:>8.1f} | {r.generation_avg_ms:>8.1f}")

    # 性能评级
    print(f"\n{'='*70}")
    print(f"  性能评级")
    print(f"{'='*70}")

    # 以并发=5为基准评级
    base = next((r for r in all_results if r.concurrency == 5), all_results[0])
    print(f"\n  基准（并发=5）:")
    print(f"    QPS:        {base.qps:.2f}")
    print(f"    P50 延迟:   {base.latency_p50_ms:.0f} ms")
    print(f"    P95 延迟:   {base.latency_p95_ms:.0f} ms")
    print(f"    错误率:     {base.error_rate:.1%}")

    # 评级标准
    if base.error_rate > 0.05:
        grade = "D - 不合格（错误率过高）"
    elif base.latency_p95_ms > 30000:
        grade = "C - 合格（P95延迟偏高）"
    elif base.latency_p95_ms > 15000:
        grade = "B - 良好"
    elif base.latency_p95_ms > 8000:
        grade = "A - 优秀"
    else:
        grade = "S - 卓越"

    print(f"\n  综合评级: {grade}")

    # 瓶颈分析
    print(f"\n  瓶颈分析:")
    if base.retrieval_avg_ms > 0 and base.generation_avg_ms > 0:
        retrieval_pct = base.retrieval_avg_ms / (base.retrieval_avg_ms + base.generation_avg_ms) * 100
        print(f"    检索占比: {retrieval_pct:.0f}% ({base.retrieval_avg_ms:.0f}ms)")
        print(f"    生成占比: {100-retrieval_pct:.0f}% ({base.generation_avg_ms:.0f}ms)")
        if retrieval_pct > 60:
            print(f"    → 瓶颈在检索阶段（Embedding/Rerank 计算）")
        else:
            print(f"    → 瓶颈在生成阶段（LLM API 调用）")

    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_url": BASE_URL,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "requests_per_level": REQUESTS_PER_LEVEL,
        },
        "results": [
            {
                "concurrency": r.concurrency,
                "qps": r.qps,
                "latency_avg_ms": r.latency_avg_ms,
                "latency_p50_ms": r.latency_p50_ms,
                "latency_p95_ms": r.latency_p95_ms,
                "latency_p99_ms": r.latency_p99_ms,
                "latency_min_ms": r.latency_min_ms,
                "latency_max_ms": r.latency_max_ms,
                "error_rate": r.error_rate,
                "successful": r.successful,
                "failed": r.failed,
                "retrieval_avg_ms": r.retrieval_avg_ms,
                "generation_avg_ms": r.generation_avg_ms,
            }
            for r in all_results
        ],
        "grade": grade,
    }

    with open("data/concurrency_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: data/concurrency_report.json")


if __name__ == "__main__":
    asyncio.run(main())
