# -*- coding: utf-8 -*-
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.main_agent import MainAgent
from agent.v1_agent import LegacyAgentV1
from engine.llm_judge import LLMJudge
from engine.regression_gate import evaluate_release_gate
from engine.retrieval_eval import (
    MockVectorDB,
    RetrievalEvaluator,
    evaluate_retrieval_pipeline,
)
from engine.runner import BenchmarkRunner


AGENT_TOKEN_PRICE_PER_1M = 0.15


class ExpertEvaluator:
    """Generation evaluator enriched with retrieval metrics from Phase 1."""

    def __init__(self, retrieval_evaluator: Optional[RetrievalEvaluator] = None):
        self.retrieval_evaluator = retrieval_evaluator

    async def score(self, case: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        if self.retrieval_evaluator:
            retrieval_result = await self.retrieval_evaluator.evaluate_single(
                case, top_k=5
            )
            hit_rate = retrieval_result["metrics"]["hit_rate@3"]
            mrr = retrieval_result["metrics"]["mrr"]
        else:
            hit_rate = 1.0
            mrr = 0.5

        return {
            "faithfulness": 0.9,
            "relevancy": 0.85,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
            },
        }


async def run_benchmark_with_results(
    agent: Any,
    agent_version: str,
    retrieval_evaluator: Optional[RetrievalEvaluator] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    print(f"[START] Khoi dong Benchmark cho {agent_version}...")

    dataset = _load_dataset()
    if not dataset:
        return None, None

    evaluator = ExpertEvaluator(retrieval_evaluator=retrieval_evaluator)
    judge = LLMJudge()
    print(f"   [Judge] Multi-Judge mode: {judge.mode} ({len(judge.judges)} judges)")

    runner = BenchmarkRunner(agent, evaluator, judge)
    started = time.perf_counter()
    results = await runner.run_all(dataset)
    total_runtime = time.perf_counter() - started

    summary = _build_summary(
        agent=agent,
        agent_version=agent_version,
        results=results,
        cost_report=judge.get_cost_report(),
        total_runtime=total_runtime,
    )
    return results, summary


async def run_benchmark(
    agent: Any,
    version: str,
    retrieval_evaluator: Optional[RetrievalEvaluator] = None,
) -> Optional[Dict[str, Any]]:
    _, summary = await run_benchmark_with_results(agent, version, retrieval_evaluator)
    return summary


async def main() -> None:
    print("=" * 65)
    print("[BENCHMARK] AI Evaluation Factory - Lab 14")
    print("=" * 65)

    print("\n[PHASE 1] Running Retrieval Evaluation...")
    retrieval_results = await evaluate_retrieval_pipeline()
    _print_retrieval_summary(retrieval_results)

    print("\n[PHASE 2] Running Generation Benchmark (V1 Legacy vs V2 Optimized)...")
    retrieval_evaluator = _build_retrieval_evaluator()

    v1_results, v1_summary = await run_benchmark_with_results(
        LegacyAgentV1(),
        "Agent_V1_Legacy",
        retrieval_evaluator,
    )
    v2_results, v2_summary = await run_benchmark_with_results(
        MainAgent(),
        "Agent_V2_Optimized",
        retrieval_evaluator,
    )

    if not v1_results or not v1_summary or not v2_results or not v2_summary:
        print("[ERROR] Khong the chay Benchmark. Kiem tra lai data/golden_set.jsonl.")
        return

    print("\n[PHASE 3] Regression Release Gate (V1 vs V2)...")
    gate = evaluate_release_gate(v1_summary, v2_summary)
    _print_delta_table(gate["deltas"])
    _print_gate_result(gate)

    if retrieval_results:
        v2_summary["retrieval_eval"] = retrieval_results["retrieval_metrics"]
        v2_summary["retrieval_by_difficulty"] = retrieval_results["by_difficulty"]

    v2_summary["regression"] = gate

    os.makedirs("reports", exist_ok=True)
    _write_json("reports/baseline_summary.json", v1_summary)
    _write_json("reports/summary.json", v2_summary)
    _write_json("reports/benchmark_results.json", v2_results)
    _write_json(
        "reports/regression_comparison.json",
        {
            "baseline_summary": v1_summary,
            "candidate_summary": v2_summary,
            "gate": gate,
            "delta_table": gate["deltas"],
        },
    )

    print("\n[OK] Baseline saved -> reports/baseline_summary.json")
    print("[OK] Summary saved  -> reports/summary.json")
    print("[OK] Results saved  -> reports/benchmark_results.json")
    print("[OK] Gate saved     -> reports/regression_comparison.json")
    print("\n[DONE] Run 'python check_lab.py' to validate submission format.")


def _load_dataset() -> Optional[List[Dict[str, Any]]]:
    dataset_path = "data/golden_set.jsonl"
    if not os.path.exists(dataset_path):
        print("[ERROR] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERROR] File data/golden_set.jsonl rong. Hay tao it nhat 1 test case.")
        return None
    return dataset


def _build_retrieval_evaluator() -> Optional[RetrievalEvaluator]:
    corpus_path = "data/document_corpus.json"
    if not os.path.exists(corpus_path):
        return None

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    vector_db = MockVectorDB(corpus)
    return RetrievalEvaluator(vector_db=vector_db)


def _build_summary(
    agent: Any,
    agent_version: str,
    results: List[Dict[str, Any]],
    cost_report: Dict[str, Any],
    total_runtime: float,
) -> Dict[str, Any]:
    total = len(results)
    latencies = [float(r.get("latency", 0.0)) for r in results]
    agent_tokens = [
        int(r.get("agent_metadata", {}).get("tokens_used", 0) or 0) for r in results
    ]
    total_agent_tokens = sum(agent_tokens)
    total_agent_cost = total_agent_tokens / 1_000_000 * AGENT_TOKEN_PRICE_PER_1M
    avg_agent_cost = total_agent_cost / total if total else 0.0
    avg_judge_cost = float(cost_report.get("avg_cost_per_eval_usd", 0.0) or 0.0)

    return {
        "metadata": {
            "version": agent_version,
            "agent_name": getattr(agent, "name", agent.__class__.__name__),
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": _avg(results, lambda r: r["judge"]["final_score"]),
            "hit_rate": _avg(results, lambda r: r["ragas"]["retrieval"]["hit_rate"]),
            "mrr": _avg(results, lambda r: r["ragas"]["retrieval"]["mrr"]),
            "agreement_rate": _avg(results, lambda r: r["judge"]["agreement_rate"]),
            "conflict_rate": _avg(
                results, lambda r: 1.0 if r["judge"].get("needs_review") else 0.0
            ),
            "faithfulness": _avg(results, lambda r: r["ragas"]["faithfulness"]),
            "relevancy": _avg(results, lambda r: r["ragas"]["relevancy"]),
            "avg_agent_tokens": _avg(agent_tokens, lambda x: x),
            "avg_agent_cost_usd": round(avg_agent_cost, 8),
            "avg_judge_cost_usd": avg_judge_cost,
            "avg_eval_cost_usd": round(avg_agent_cost + avg_judge_cost, 8),
            "avg_latency_seconds": _avg(latencies, lambda x: x),
            "p95_latency_seconds": _percentile(latencies, 0.95),
            "total_runtime_seconds": round(total_runtime, 3),
        },
        "cost": {
            **cost_report,
            "total_agent_tokens": total_agent_tokens,
            "avg_agent_tokens": _avg(agent_tokens, lambda x: x),
            "total_agent_cost_usd": round(total_agent_cost, 8),
            "avg_agent_cost_usd": round(avg_agent_cost, 8),
            "avg_eval_cost_usd": round(avg_agent_cost + avg_judge_cost, 8),
        },
    }


def _avg(items: List[Any], getter) -> float:
    if not items:
        return 0.0
    return sum(float(getter(item)) for item in items) / len(items)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 4)


def _print_retrieval_summary(retrieval_results: Optional[Dict[str, Any]]) -> None:
    if not retrieval_results:
        return

    rm = retrieval_results["retrieval_metrics"]
    print("\n[RETRIEVAL SUMMARY]")
    print(f"  Hit Rate @ 3 : {rm['hit_rate@3']:.4f} ({rm['hit_rate@3'] * 100:.1f}%)")
    print(f"  MRR          : {rm['mrr']:.4f} ({rm['mrr'] * 100:.1f}%)")
    if rm["hit_rate@3"] >= 0.80:
        print("  [OK] Retrieval du dieu kien de chay Generation Eval!")
    else:
        print("  [WARN] Retrieval chua du tot - ket qua Generation co the bi anh huong!")


def _print_delta_table(deltas: Dict[str, Dict[str, Optional[float]]]) -> None:
    print(f"\n  {'Metric':<26} {'V1':>10} {'V2':>10} {'Delta':>10}")
    print(f"  {'-' * 60}")
    for metric in [
        "avg_score",
        "hit_rate",
        "mrr",
        "agreement_rate",
        "avg_eval_cost_usd",
        "avg_latency_seconds",
        "p95_latency_seconds",
        "total_runtime_seconds",
    ]:
        row = deltas[metric]
        delta = row["delta"]
        sign = "+" if delta is not None and delta >= 0 else ""
        print(
            f"  {metric:<26} "
            f"{row['baseline']:>10.4f} "
            f"{row['candidate']:>10.4f} "
            f"{sign}{delta:>9.4f}"
        )


def _print_gate_result(gate: Dict[str, Any]) -> None:
    print("\n[RELEASE GATE]")
    for check in gate["checks"]:
        status = "PASS" if check["status"] == "pass" else "FAIL"
        print(f"  {status:<4} {check['name']}")

    print(f"\n[DECISION] {gate['decision']}")
    if gate["blocking_failures"]:
        print("[BLOCKING FAILURES]")
        for failure in gate["blocking_failures"]:
            print(f"  - {failure['name']}: {failure['message']}")
    if gate["warnings"]:
        print("[WARNINGS]")
        for warning in gate["warnings"]:
            print(f"  - {warning['name']}: {warning['message']}")


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
