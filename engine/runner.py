# -*- coding: utf-8 -*-
"""
Benchmark Runner — Lab 14: AI Evaluation Factory
=================================================
Chạy toàn bộ pipeline đánh giá bất đồng bộ (async) với:
  - asyncio.gather() để chạy song song theo batch (tránh rate limit)
  - Đo latency chi tiết cho từng test case
  - Tổng hợp p95 latency, total runtime, token usage
"""

import asyncio
import statistics
import time
from typing import List, Dict, Any


class BenchmarkRunner:
    """
    Chạy benchmark song song với asyncio.gather().
    Mục tiêu Expert: toàn bộ 55 cases < 2 phút.
    """

    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        """Chạy 1 test case: Agent → Retrieval Eval → Multi-Judge."""
        start_time = time.perf_counter()

        # 1. Gọi Agent (async)
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Chạy Retrieval + RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)

        # 3. Chạy Multi-Judge (2 model song song bên trong LLMJudge)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", ""),
        )

        return {
            "test_id": test_case.get("id", "unknown"),
            "test_case": test_case["question"],
            "difficulty": test_case.get("difficulty", "unknown"),
            "type": test_case.get("type", "unknown"),
            "agent_response": response["answer"],
            "latency": round(latency, 4),
            "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(
        self, dataset: List[Dict], batch_size: int = 10
    ) -> List[Dict]:
        """
        Chạy song song toàn bộ dataset theo batch.
        batch_size=10: đủ nhanh mà không bị rate limit khi có API key thật.
        """
        pipeline_start = time.perf_counter()
        results = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)

        total_runtime = time.perf_counter() - pipeline_start
        # Lưu total_runtime vào list để main.py có thể đọc
        self._last_total_runtime = total_runtime
        self._last_results = results
        return results

    # ------------------------------------------------------------------ #
    # Performance report
    # ------------------------------------------------------------------ #
    def get_performance_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Tổng hợp báo cáo hiệu năng:
          - avg / p50 / p95 / max latency
          - total runtime
          - total & avg token usage
        """
        latencies = [r["latency"] for r in results]
        tokens = [r.get("tokens_used", 0) for r in results]
        n = len(latencies)

        sorted_lat = sorted(latencies)
        p95_idx = int(0.95 * n)
        p50_idx = int(0.50 * n)

        return {
            "total_cases": n,
            "total_runtime_seconds": round(getattr(self, "_last_total_runtime", 0), 3),
            "avg_latency_seconds": round(statistics.mean(latencies), 4),
            "p50_latency_seconds": round(sorted_lat[p50_idx], 4),
            "p95_latency_seconds": round(sorted_lat[min(p95_idx, n - 1)], 4),
            "max_latency_seconds": round(max(latencies), 4),
            "total_tokens": sum(tokens),
            "avg_tokens_per_case": round(statistics.mean(tokens), 1) if tokens else 0,
            "under_2min_target": getattr(self, "_last_total_runtime", 999) < 120,
        }
