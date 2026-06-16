# -*- coding: utf-8 -*-
"""
Unit Tests — engine/retrieval_eval.py
======================================
Kiểm tra tính đúng đắn của các Retrieval Metrics:
  - Hit Rate @ K
  - MRR (Mean Reciprocal Rank)
  - Precision @ K
  - Recall @ K
  - MockVectorDB TF-IDF ranking

Chạy: python -X utf8 tests/test_retrieval_eval.py
"""

import asyncio
import sys
import os

# Đảm bảo import từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.retrieval_eval import RetrievalEvaluator, MockVectorDB

PASS_SYMBOL = "[PASS]"
FAIL_SYMBOL = "[FAIL]"
_results = []


def assert_close(actual, expected, tol=1e-6, name=""):
    ok = abs(actual - expected) <= tol
    _results.append((name, ok, actual, expected))
    sym = PASS_SYMBOL if ok else FAIL_SYMBOL
    print(f"  {sym} {name}: got={actual:.6f}, expected={expected:.6f}")
    return ok


def assert_equal(actual, expected, name=""):
    ok = actual == expected
    _results.append((name, ok, actual, expected))
    sym = PASS_SYMBOL if ok else FAIL_SYMBOL
    print(f"  {sym} {name}: got={actual}, expected={expected}")
    return ok


# ---------------------------------------------------------------------------
# Test 1: Hit Rate
# ---------------------------------------------------------------------------
def test_hit_rate():
    print("\n[TEST] Hit Rate @ K")
    ev = RetrievalEvaluator()

    # Perfect hit at position 1
    assert_close(
        ev.calculate_hit_rate(["doc_a"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        1.0, name="hit@3 — relevant doc at pos 1"
    )
    # Hit at position 3, top_k=3
    assert_close(
        ev.calculate_hit_rate(["doc_c"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        1.0, name="hit@3 — relevant doc at pos 3"
    )
    # Miss: relevant doc NOT in top_k=3
    assert_close(
        ev.calculate_hit_rate(["doc_d"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        0.0, name="hit@3 — miss"
    )
    # top_k=1: only position 1 counts
    assert_close(
        ev.calculate_hit_rate(["doc_b"], ["doc_a", "doc_b", "doc_c"], top_k=1),
        0.0, name="hit@1 — relevant doc at pos 2, miss"
    )
    assert_close(
        ev.calculate_hit_rate(["doc_a"], ["doc_a", "doc_b", "doc_c"], top_k=1),
        1.0, name="hit@1 — relevant doc at pos 1, hit"
    )
    # Out-of-scope: no expected_ids
    assert_close(
        ev.calculate_hit_rate([], ["doc_a", "doc_b"], top_k=3),
        1.0, name="hit@3 — empty expected (out-of-scope) -> 1.0"
    )
    # Multiple expected ids: at least 1 in top-k
    assert_close(
        ev.calculate_hit_rate(["doc_x", "doc_b"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        1.0, name="hit@3 — multi-expected, partial hit"
    )


# ---------------------------------------------------------------------------
# Test 2: MRR
# ---------------------------------------------------------------------------
def test_mrr():
    print("\n[TEST] MRR (Mean Reciprocal Rank)")
    ev = RetrievalEvaluator()

    assert_close(
        ev.calculate_mrr(["doc_a"], ["doc_a", "doc_b", "doc_c"]),
        1.0, name="MRR — relevant at pos 1 => 1/1"
    )
    assert_close(
        ev.calculate_mrr(["doc_b"], ["doc_a", "doc_b", "doc_c"]),
        0.5, name="MRR — relevant at pos 2 => 1/2"
    )
    assert_close(
        ev.calculate_mrr(["doc_c"], ["doc_a", "doc_b", "doc_c"]),
        1 / 3, tol=1e-5, name="MRR — relevant at pos 3 => 1/3"
    )
    assert_close(
        ev.calculate_mrr(["doc_d"], ["doc_a", "doc_b", "doc_c"]),
        0.0, name="MRR — not found => 0"
    )
    assert_close(
        ev.calculate_mrr([], ["doc_a", "doc_b"]),
        1.0, name="MRR — empty expected (out-of-scope) -> 1.0"
    )
    # Multiple expected: take the first one found
    assert_close(
        ev.calculate_mrr(["doc_x", "doc_b"], ["doc_a", "doc_b", "doc_c"]),
        0.5, name="MRR — multi-expected, first match at pos 2"
    )


# ---------------------------------------------------------------------------
# Test 3: Precision @ K
# ---------------------------------------------------------------------------
def test_precision_at_k():
    print("\n[TEST] Precision @ K")
    ev = RetrievalEvaluator()

    # 2 relevant in top-3
    assert_close(
        ev.calculate_precision_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        2 / 3, tol=1e-5, name="P@3 — 2/3 relevant"
    )
    # 0 relevant
    assert_close(
        ev.calculate_precision_at_k(["doc_x"], ["doc_a", "doc_b", "doc_c"], top_k=3),
        0.0, name="P@3 — 0 relevant"
    )
    # All relevant
    assert_close(
        ev.calculate_precision_at_k(["doc_a", "doc_b", "doc_c"],
                                     ["doc_a", "doc_b", "doc_c"], top_k=3),
        1.0, name="P@3 — 3/3 relevant"
    )


# ---------------------------------------------------------------------------
# Test 4: Recall @ K
# ---------------------------------------------------------------------------
def test_recall_at_k():
    print("\n[TEST] Recall @ K")
    ev = RetrievalEvaluator()

    # 1 of 2 relevant found in top-5
    assert_close(
        ev.calculate_recall_at_k(["doc_a", "doc_z"], ["doc_a", "doc_b", "doc_c"], top_k=5),
        0.5, name="R@5 — 1 of 2 relevant found"
    )
    # All relevant found
    assert_close(
        ev.calculate_recall_at_k(["doc_a", "doc_b"], ["doc_a", "doc_b", "doc_c"], top_k=5),
        1.0, name="R@5 — all relevant found"
    )
    # None found
    assert_close(
        ev.calculate_recall_at_k(["doc_x", "doc_y"], ["doc_a", "doc_b", "doc_c"], top_k=5),
        0.0, name="R@5 — none found"
    )


# ---------------------------------------------------------------------------
# Test 5: MockVectorDB ranking
# ---------------------------------------------------------------------------
def test_mock_vector_db_ranking():
    print("\n[TEST] MockVectorDB TF-IDF Ranking")
    corpus = {
        "doc_password": "mat khau phai duoc thay doi moi 90 ngay va co it nhat 8 ky tu",
        "doc_leave": "nghi phep duoc dang ky truoc 3 ngay lam viec va toi da 12 ngay moi nam",
        "doc_vpn": "vpn la bat buoc khi truy cap he thong tu ngoai van phong cisco anyconnect",
        "doc_backup": "backup du lieu tu dong luc 2 gio sang va luu tru 90 ngay",
    }
    db = MockVectorDB(corpus)

    # Query liên quan đến VPN
    results = db.get_doc_ids("ket noi vpn cisco anyconnect", top_k=3)
    assert_equal("doc_vpn" in results, True, name="VectorDB — VPN query retrieves doc_vpn in top-3")

    # Query liên quan đến mật khẩu
    results = db.get_doc_ids("mat khau thay doi", top_k=3)
    assert_equal("doc_password" in results, True,
                 name="VectorDB — password query retrieves doc_password in top-3")

    # Scores giảm dần
    scored = db.retrieve("backup du lieu luu tru", top_k=4)
    scores = [s for _, s in scored]
    assert_equal(scores == sorted(scores, reverse=True), True,
                 name="VectorDB — scores are sorted descending")


# ---------------------------------------------------------------------------
# Test 6: Async evaluate_single
# ---------------------------------------------------------------------------
async def test_evaluate_single():
    print("\n[TEST] evaluate_single (async)")
    corpus = {
        "doc_policy_001": "mat khau phai duoc thay doi moi 90 ngay it nhat 8 ky tu",
        "doc_policy_002": "nghi phep 12 ngay co luong moi nam dang ky truoc 3 ngay",
    }
    db = MockVectorDB(corpus)
    ev = RetrievalEvaluator(vector_db=db)

    test_case = {
        "id": "test_001",
        "question": "mat khau thay doi bao lau mot lan",
        "ground_truth_doc_ids": ["doc_policy_001"],
        "difficulty": "easy",
        "type": "fact-check",
    }

    result = await ev.evaluate_single(test_case, top_k=5)
    assert_equal("metrics" in result, True, name="evaluate_single — returns metrics dict")
    assert_equal("hit_rate@3" in result["metrics"], True,
                 name="evaluate_single — has hit_rate@3")
    assert_equal("mrr" in result["metrics"], True, name="evaluate_single — has mrr")

    hr3 = result["metrics"]["hit_rate@3"]
    assert_equal(hr3 in (0.0, 1.0), True, name="evaluate_single — hit_rate@3 is 0 or 1")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests():
    print("=" * 55)
    print("[TESTS] engine/retrieval_eval.py — Unit Tests")
    print("=" * 55)

    test_hit_rate()
    test_mrr()
    test_precision_at_k()
    test_recall_at_k()
    test_mock_vector_db_ranking()
    asyncio.run(test_evaluate_single())

    print("\n" + "=" * 55)
    passed = sum(1 for _, ok, _, _ in _results if ok)
    total = len(_results)
    print(f"[RESULT] {passed}/{total} tests PASSED")

    if passed == total:
        print("[OK] All tests passed!")
    else:
        print("[FAIL] Some tests failed — check above for details.")
        failed = [(name, a, e) for name, ok, a, e in _results if not ok]
        for name, actual, expected in failed:
            print(f"  FAIL: {name}: got={actual}, expected={expected}")

    return passed == total


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
