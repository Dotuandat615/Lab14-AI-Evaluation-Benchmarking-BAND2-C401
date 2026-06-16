# -*- coding: utf-8 -*-
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import MockVectorDB, RetrievalEvaluator, evaluate_retrieval_pipeline
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent


# ---------------------------------------------------------------------------
# Expert components — Kết hợp Retrieval Eval (Phase 1) + LLMJudge (Phase 2)
# ---------------------------------------------------------------------------
class ExpertEvaluator:
    """
    Evaluator tích hợp Retrieval Eval thực sự từ Phase 1.
    Giai đoạn 2: thay thế faithfulness/relevancy bằng RAGAS metrics thực sự.
    """
    def __init__(self, retrieval_evaluator: RetrievalEvaluator = None):
        self.retrieval_evaluator = retrieval_evaluator

    async def score(self, case, resp):
        # Retrieval metrics thực sự từ RetrievalEvaluator (Phase 1)
        if self.retrieval_evaluator:
            retrieval_result = await self.retrieval_evaluator.evaluate_single(case, top_k=5)
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
            }
        }


# ---------------------------------------------------------------------------
# Benchmark pipeline — Kết hợp Retrieval Eval + LLMJudge
# ---------------------------------------------------------------------------
async def run_benchmark_with_results(agent_version: str, retrieval_evaluator=None):
    print(f"[START] Khoi dong Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("[ERROR] Thieu data/golden_set.jsonl. Hay chay 'python data/synthetic_gen.py' truoc.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERROR] File data/golden_set.jsonl rong. Hay tao it nhat 1 test case.")
        return None, None

    evaluator = ExpertEvaluator(retrieval_evaluator=retrieval_evaluator)

    # Dùng LLMJudge thực sự từ Phase 2 (có fallback mock nếu không có API key)
    judge = LLMJudge()
    print(f"   [Judge] Multi-Judge mode: {judge.mode} ({len(judge.judges)} judges)")
    runner = BenchmarkRunner(MainAgent(), evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    cost_report = judge.get_cost_report()

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "conflict_rate": sum(1 for r in results if r["judge"].get("needs_review")) / total,
            "faithfulness": sum(r["ragas"]["faithfulness"] for r in results) / total,
            "relevancy": sum(r["ragas"]["relevancy"] for r in results) / total,
            "avg_eval_cost_usd": cost_report["avg_cost_per_eval_usd"],
        },
        "cost": cost_report,
    }
    return results, summary


async def run_benchmark(version, retrieval_evaluator=None):
    _, summary = await run_benchmark_with_results(version, retrieval_evaluator)
    return summary


# ---------------------------------------------------------------------------
# Main pipeline — Phase 1 Retrieval → Phase 2 Generation → Phase 3 Regression
# ---------------------------------------------------------------------------
async def main():
    print("=" * 65)
    print("[BENCHMARK] AI Evaluation Factory - Lab 14")
    print("=" * 65)

    # ---- Phase 1: Retrieval Evaluation ----
    print("\n[PHASE 1] Running Retrieval Evaluation...")
    retrieval_results = await evaluate_retrieval_pipeline()

    if retrieval_results:
        rm = retrieval_results["retrieval_metrics"]
        print(f"\n[RETRIEVAL SUMMARY]")
        print(f"  Hit Rate @ 3 : {rm['hit_rate@3']:.4f} ({rm['hit_rate@3']*100:.1f}%)")
        print(f"  MRR          : {rm['mrr']:.4f} ({rm['mrr']*100:.1f}%)")
        hr3 = rm["hit_rate@3"]
        if hr3 >= 0.80:
            print("  [OK] Retrieval du dieu kien de chay Generation Eval!")
        else:
            print("  [WARN] Retrieval chua du tot - ket qua Generation co the bi anh huong!")

    # ---- Phase 2: Generation Benchmark ----
    print("\n[PHASE 2] Running Generation Benchmark (V1 vs V2)...")

    # Khởi tạo retrieval evaluator để tích hợp vào generation eval
    corpus_path = "data/document_corpus.json"
    retrieval_evaluator = None
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        vector_db = MockVectorDB(corpus)
        retrieval_evaluator = RetrievalEvaluator(vector_db=vector_db)

    v1_summary = await run_benchmark("Agent_V1_Base", retrieval_evaluator)
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", retrieval_evaluator
    )

    if not v1_summary or not v2_summary:
        print("[ERROR] Khong the chay Benchmark. Kiem tra lai data/golden_set.jsonl.")
        return

    # ---- Phase 3: Regression Analysis ----
    print("\n[PHASE 3] Regression Analysis (V1 vs V2)...")
    delta_score = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_hr = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]

    print(f"\n  {'Metric':<22} {'V1':>8} {'V2':>8} {'Delta':>8}")
    print(f"  {'-'*50}")
    for metric in ["avg_score", "hit_rate", "mrr", "agreement_rate",
                   "faithfulness", "relevancy", "avg_eval_cost_usd"]:
        v1_val = v1_summary["metrics"].get(metric, 0)
        v2_val = v2_summary["metrics"].get(metric, 0)
        d = v2_val - v1_val
        sign = "+" if d >= 0 else ""
        print(f"  {metric:<22} {v1_val:>8.4f} {v2_val:>8.4f} {sign}{d:>7.4f}")

    # Release Gate
    THRESHOLDS = {
        "avg_score_min": 4.0,
        "hit_rate_min": 0.80,
        "agreement_rate_min": 0.70,
    }

    gate_pass = (
        v2_summary["metrics"]["avg_score"] >= THRESHOLDS["avg_score_min"]
        and v2_summary["metrics"]["hit_rate"] >= THRESHOLDS["hit_rate_min"]
        and v2_summary["metrics"]["agreement_rate"] >= THRESHOLDS["agreement_rate_min"]
    )

    print(f"\n[RELEASE GATE]")
    print(f"  avg_score >= {THRESHOLDS['avg_score_min']} : {'PASS' if v2_summary['metrics']['avg_score'] >= THRESHOLDS['avg_score_min'] else 'FAIL'}")
    print(f"  hit_rate  >= {THRESHOLDS['hit_rate_min']} : {'PASS' if v2_summary['metrics']['hit_rate'] >= THRESHOLDS['hit_rate_min'] else 'FAIL'}")
    print(f"  agreement >= {THRESHOLDS['agreement_rate_min']} : {'PASS' if v2_summary['metrics']['agreement_rate'] >= THRESHOLDS['agreement_rate_min'] else 'FAIL'}")

    if gate_pass and delta_score >= 0:
        decision = "APPROVE - CHAP NHAN BAN CAP NHAT"
    elif gate_pass and delta_score < 0:
        decision = "APPROVE WITH WARNING - Score giam nhung van tren nguong"
    else:
        decision = "BLOCK RELEASE - Tu choi, khong dat nguong chat luong"

    print(f"\n[DECISION] {decision}")

    # ---- Save reports ----
    os.makedirs("reports", exist_ok=True)

    # Tích hợp retrieval metrics vào summary
    if retrieval_results:
        v2_summary["retrieval_eval"] = retrieval_results["retrieval_metrics"]
        v2_summary["retrieval_by_difficulty"] = retrieval_results["by_difficulty"]

    v2_summary["regression"] = {
        "v1_score": v1_summary["metrics"]["avg_score"],
        "v2_score": v2_summary["metrics"]["avg_score"],
        "delta_score": delta_score,
        "delta_hit_rate": delta_hr,
        "decision": decision,
        "thresholds": THRESHOLDS,
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Reports saved -> reports/summary.json")
    print(f"[OK] Results saved -> reports/benchmark_results.json")
    print("\n[DONE] Run 'python check_lab.py' to validate submission format.")


if __name__ == "__main__":
    asyncio.run(main())
