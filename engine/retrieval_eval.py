# -*- coding: utf-8 -*-
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
"""
Retrieval Evaluator — Lab 14: AI Evaluation Factory
=====================================================
Tính toán Hit Rate @ K và MRR (Mean Reciprocal Rank) thực sự
cho Vector DB Retrieval Stage.

Kiến trúc:
  - RetrievalEvaluator: tính metric cho từng test case
  - MockVectorDB: giả lập Vector DB với TF-IDF cosine similarity
  - evaluate_retrieval_pipeline(): chạy đầy đủ pipeline và in báo cáo

Chạy độc lập:
  python engine/retrieval_eval.py
"""

import json
import math
import os
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Mock Vector DB — Giả lập retrieval bằng keyword matching + scoring
# Trong production: thay bằng ChromaDB / Pinecone / Weaviate thực sự
# ---------------------------------------------------------------------------
class MockVectorDB:
    """
    Giả lập Vector DB dùng TF-IDF đơn giản + cosine similarity.
    Đủ để demo retrieval pipeline mà không cần API bên ngoài.
    """

    def __init__(self, corpus: Dict[str, str]):
        """
        corpus: {doc_id: document_text}
        """
        self.corpus = corpus
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase + split by whitespace + remove punctuation."""
        import re
        text = text.lower()
        tokens = re.findall(r'[a-zA-ZÀ-ỹ0-9]+', text)
        return tokens

    def _build_index(self):
        """Build inverted index và compute TF-IDF weights."""
        self.doc_tokens: Dict[str, List[str]] = {}
        self.tf: Dict[str, Dict[str, float]] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.N = len(self.corpus)

        for doc_id, text in self.corpus.items():
            tokens = self._tokenize(text)
            self.doc_tokens[doc_id] = tokens

            # Term Frequency
            freq: Dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self.tf[doc_id] = {
                t: count / len(tokens) for t, count in freq.items()
            }
            # Document Frequency
            for t in set(tokens):
                self.df[t] += 1

    def _tfidf_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Compute TF-IDF cosine similarity between query and document."""
        score = 0.0
        tf_doc = self.tf.get(doc_id, {})
        for token in query_tokens:
            if token in tf_doc:
                tf = tf_doc[token]
                idf = math.log((self.N + 1) / (self.df.get(token, 0) + 1)) + 1
                score += tf * idf
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top_k documents ranked by TF-IDF score.
        Returns: list of (doc_id, score) sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        scores = {}
        for doc_id in self.corpus:
            scores[doc_id] = self._tfidf_score(query_tokens, doc_id)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_doc_ids(self, query: str, top_k: int = 5) -> List[str]:
        """Return only the doc_ids (for compatibility with evaluator)."""
        return [doc_id for doc_id, _ in self.retrieve(query, top_k)]


# ---------------------------------------------------------------------------
# Retrieval Evaluator — Core Metrics
# ---------------------------------------------------------------------------
class RetrievalEvaluator:
    """
    Tính toán các Retrieval Metrics chuẩn:
      - Hit Rate @ K: tỷ lệ queries có ít nhất 1 relevant doc trong top-K
      - MRR (Mean Reciprocal Rank): đánh giá thứ hạng của doc liên quan đầu tiên
      - Precision @ K: tỷ lệ docs liên quan trong top-K kết quả
      - Recall @ K: tỷ lệ relevant docs được tìm thấy trong top-K
    """

    def __init__(self, vector_db: Optional[MockVectorDB] = None):
        self.vector_db = vector_db

    # ------------------------------------------------------------------
    # Per-query metrics
    # ------------------------------------------------------------------
    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3
    ) -> float:
        """
        Hit Rate @ K: 1.0 nếu ít nhất 1 expected_id nằm trong top_k retrieved.
        Công thức: HR@K = 1 nếu ∃ id ∈ expected_ids : id ∈ retrieved_ids[:K]
        """
        if not expected_ids:
            return 1.0  # out-of-scope: không cần retrieve gì
        top_retrieved = set(retrieved_ids[:top_k])
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str]
    ) -> float:
        """
        Reciprocal Rank: 1/rank của relevant document đầu tiên tìm thấy.
        MRR = 1/position (1-indexed). Nếu không tìm thấy → 0.
        Công thức: RR = 1 / rank_first_relevant
        """
        if not expected_ids:
            return 1.0  # out-of-scope: không cần retrieve
        expected_set = set(expected_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_set:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_precision_at_k(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3
    ) -> float:
        """
        Precision @ K = |relevant ∩ retrieved[:K]| / K
        """
        if not expected_ids:
            return 1.0
        expected_set = set(expected_ids)
        top_retrieved = retrieved_ids[:top_k]
        relevant_count = sum(1 for doc_id in top_retrieved if doc_id in expected_set)
        return relevant_count / top_k if top_k > 0 else 0.0

    def calculate_recall_at_k(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 5
    ) -> float:
        """
        Recall @ K = |relevant ∩ retrieved[:K]| / |relevant|
        """
        if not expected_ids:
            return 1.0
        expected_set = set(expected_ids)
        top_retrieved = set(retrieved_ids[:top_k])
        found = len(expected_set & top_retrieved)
        return found / len(expected_set)

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------
    async def evaluate_single(
        self,
        test_case: Dict,
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval for a single test case.
        Requires test_case to have 'ground_truth_doc_ids' and 'question'.
        """
        question = test_case.get("question", "")
        expected_ids = test_case.get("ground_truth_doc_ids", [])

        # Retrieve từ Vector DB (hoặc dùng danh sách có sẵn)
        if self.vector_db:
            retrieved_ids = self.vector_db.get_doc_ids(question, top_k=top_k)
            retrieval_scores = self.vector_db.retrieve(question, top_k=top_k)
            scores_dict = {doc_id: score for doc_id, score in retrieval_scores}
        else:
            # Fallback: dùng retrieved_ids từ test_case nếu có
            retrieved_ids = test_case.get("retrieved_ids", [])
            scores_dict = {}

        hit_rate_1 = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=1)
        hit_rate_3 = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        hit_rate_5 = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=5)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        precision_3 = self.calculate_precision_at_k(expected_ids, retrieved_ids, top_k=3)
        recall_5 = self.calculate_recall_at_k(expected_ids, retrieved_ids, top_k=5)

        # Kiểm tra ground truth có được retrieve không
        found_ids = [gid for gid in expected_ids if gid in retrieved_ids[:top_k]]
        missing_ids = [gid for gid in expected_ids if gid not in retrieved_ids[:top_k]]

        return {
            "test_id": test_case.get("id", "unknown"),
            "question": question[:80] + "..." if len(question) > 80 else question,
            "difficulty": test_case.get("difficulty", "unknown"),
            "type": test_case.get("type", "unknown"),
            "expected_doc_ids": expected_ids,
            "retrieved_doc_ids": retrieved_ids,
            "found_ids": found_ids,
            "missing_ids": missing_ids,
            "retrieval_scores": scores_dict,
            "metrics": {
                "hit_rate@1": hit_rate_1,
                "hit_rate@3": hit_rate_3,
                "hit_rate@5": hit_rate_5,
                "mrr": mrr,
                "precision@3": precision_3,
                "recall@5": recall_5,
            }
        }

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        top_k: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        Chạy retrieval eval cho toàn bộ dataset và tính aggregate metrics.
        """
        print(f"\n🔍 Running Retrieval Evaluation on {len(dataset)} test cases...")
        start_time = time.perf_counter()

        # Async concurrent evaluation
        tasks = [self.evaluate_single(case, top_k=top_k) for case in dataset]
        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time

        # Aggregate metrics
        total = len(results)
        if total == 0:
            return {"error": "No results"}

        # Tách out-of-scope (no ground truth) khỏi metrics chính
        retrieval_cases = [r for r in results if r["expected_doc_ids"]]
        oos_cases = [r for r in results if not r["expected_doc_ids"]]

        def avg(key, lst):
            if not lst:
                return 0.0
            return sum(r["metrics"][key] for r in lst) / len(lst)

        # Metrics chỉ tính trên cases có ground truth
        n_retrieval = len(retrieval_cases)

        aggregate = {
            "summary": {
                "total_cases": total,
                "retrieval_cases": n_retrieval,
                "out_of_scope_cases": len(oos_cases),
                "elapsed_seconds": round(elapsed, 3),
            },
            "retrieval_metrics": {
                "hit_rate@1": round(avg("hit_rate@1", retrieval_cases), 4),
                "hit_rate@3": round(avg("hit_rate@3", retrieval_cases), 4),
                "hit_rate@5": round(avg("hit_rate@5", retrieval_cases), 4),
                "mrr": round(avg("mrr", retrieval_cases), 4),
                "precision@3": round(avg("precision@3", retrieval_cases), 4),
                "recall@5": round(avg("recall@5", retrieval_cases), 4),
            },
            "by_difficulty": {},
            "failure_cases": [],
            "individual_results": results,
        }

        # Per-difficulty breakdown
        difficulties = set(r["difficulty"] for r in retrieval_cases)
        for diff in difficulties:
            diff_results = [r for r in retrieval_cases if r["difficulty"] == diff]
            aggregate["by_difficulty"][diff] = {
                "count": len(diff_results),
                "hit_rate@3": round(avg("hit_rate@3", diff_results), 4),
                "mrr": round(avg("mrr", diff_results), 4),
            }

        # Failure analysis: cases với hit_rate@5 = 0 (có GT nhưng không retrieve được)
        aggregate["failure_cases"] = [
            {
                "test_id": r["test_id"],
                "question": r["question"],
                "missing_ids": r["missing_ids"],
                "retrieved_top3": r["retrieved_doc_ids"][:3],
            }
            for r in retrieval_cases
            if r["metrics"]["hit_rate@5"] == 0.0
        ]

        return aggregate


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def print_report(aggregate: Dict):
    """In bao cao dep ra terminal."""
    print("\n" + "=" * 65)
    print("[REPORT] RETRIEVAL EVALUATION REPORT")
    print("=" * 65)

    s = aggregate["summary"]
    print(f"\n[Summary]")
    print(f"  Total test cases     : {s['total_cases']}")
    print(f"  Retrieval cases      : {s['retrieval_cases']} (co Ground Truth)")
    print(f"  Out-of-scope cases   : {s['out_of_scope_cases']} (khong can retrieve)")
    print(f"  Eval time            : {s['elapsed_seconds']:.3f}s")

    m = aggregate["retrieval_metrics"]
    print(f"\n[Metrics] Retrieval Metrics (tren {s['retrieval_cases']} cases):")
    print(f"  Hit Rate @ 1  : {m['hit_rate@1']:.4f}  ({m['hit_rate@1']*100:.1f}%)")
    print(f"  Hit Rate @ 3  : {m['hit_rate@3']:.4f}  ({m['hit_rate@3']*100:.1f}%)")
    print(f"  Hit Rate @ 5  : {m['hit_rate@5']:.4f}  ({m['hit_rate@5']*100:.1f}%)")
    print(f"  MRR           : {m['mrr']:.4f}  ({m['mrr']*100:.1f}%)")
    print(f"  Precision @ 3 : {m['precision@3']:.4f}  ({m['precision@3']*100:.1f}%)")
    print(f"  Recall @ 5    : {m['recall@5']:.4f}  ({m['recall@5']*100:.1f}%)")

    # Quality assessment
    hr3 = m["hit_rate@3"]
    mrr = m["mrr"]
    if hr3 >= 0.80 and mrr >= 0.70:
        verdict = "[EXCELLENT] Retrieval du tot de danh gia Generation"
    elif hr3 >= 0.60 and mrr >= 0.50:
        verdict = "[ACCEPTABLE] Retrieval can cai thien truoc khi production"
    else:
        verdict = "[POOR] Can fix Retrieval pipeline truoc khi danh gia Generation"
    print(f"\n[Assessment] {verdict}")

    print(f"\n[Breakdown] By Difficulty:")
    print(f"  {'Difficulty':<15} {'Count':>6} {'Hit@3':>8} {'MRR':>8}")
    print(f"  {'-'*40}")
    for diff, stats in aggregate["by_difficulty"].items():
        print(f"  {diff:<15} {stats['count']:>6} {stats['hit_rate@3']:>8.4f} {stats['mrr']:>8.4f}")

    failures = aggregate["failure_cases"]
    if failures:
        print(f"\n[FAILURES] {len(failures)} cases voi Hit Rate@5 = 0:")
        for fc in failures[:5]:  # Chi hien 5 failures dau
            print(f"  [{fc['test_id']}] {fc['question']}")
            print(f"    Missing : {fc['missing_ids']}")
            print(f"    Got Top3: {fc['retrieved_top3']}")
    else:
        print(f"\n[PERFECT] Khong co failure cases - Retrieval hoat dong hoan hao!")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def evaluate_retrieval_pipeline():
    """Full retrieval evaluation pipeline."""

    # 1. Load corpus
    corpus_path = "data/document_corpus.json"
    golden_set_path = "data/golden_set.jsonl"

    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus khong tim thay: {corpus_path}")
        print("   Hay chay: python data/synthetic_gen.py truoc")
        return None

    if not os.path.exists(golden_set_path):
        print(f"[ERROR] Golden set khong tim thay: {golden_set_path}")
        print("   Hay chay: python data/synthetic_gen.py truoc")
        return None

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"[OK] Loaded corpus: {len(corpus)} documents")
    print(f"[OK] Loaded dataset: {len(dataset)} test cases")

    # 2. Init Vector DB + Evaluator
    vector_db = MockVectorDB(corpus)
    evaluator = RetrievalEvaluator(vector_db=vector_db)

    # 3. Run evaluation
    results = await evaluator.evaluate_batch(dataset, top_k=5, verbose=True)

    # 4. Print report
    print_report(results)

    # 5. Save results
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/retrieval_eval_results.json"
    # Chỉ lưu summary và individual results (bỏ scores để gọn)
    save_data = {
        "summary": results["summary"],
        "retrieval_metrics": results["retrieval_metrics"],
        "by_difficulty": results["by_difficulty"],
        "failure_cases": results["failure_cases"],
        "individual_results": [
            {k: v for k, v in r.items() if k != "retrieval_scores"}
            for r in results["individual_results"]
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved -> {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(evaluate_retrieval_pipeline())
