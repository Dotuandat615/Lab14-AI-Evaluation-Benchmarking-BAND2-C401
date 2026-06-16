import unittest

from engine.regression_gate import evaluate_release_gate


def make_summary(
    version="Agent",
    avg_score=4.4,
    hit_rate=0.9,
    mrr=0.8,
    agreement_rate=0.8,
    cost=0.001,
    avg_latency=0.5,
    p95_latency=0.7,
    runtime=30.0,
):
    return {
        "metadata": {"version": version},
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "agreement_rate": agreement_rate,
            "faithfulness": 0.9,
            "relevancy": 0.85,
            "avg_eval_cost_usd": cost,
            "avg_latency_seconds": avg_latency,
            "p95_latency_seconds": p95_latency,
            "total_runtime_seconds": runtime,
        },
    }


class RegressionGateTests(unittest.TestCase):
    def test_release_when_candidate_is_stable_or_better(self):
        baseline = make_summary(version="Agent_V1", avg_score=4.2)
        candidate = make_summary(version="Agent_V2", avg_score=4.4, cost=0.0009)

        gate = evaluate_release_gate(baseline, candidate)

        self.assertEqual(gate["decision"], "Release")
        self.assertEqual(gate["blocking_failures"], [])

    def test_rollback_when_quality_regresses_too_much(self):
        baseline = make_summary(avg_score=4.5)
        candidate = make_summary(avg_score=4.2)

        gate = evaluate_release_gate(baseline, candidate)

        self.assertEqual(gate["decision"], "Rollback")
        self.assertTrue(
            any(f["name"] == "avg_score_regression" for f in gate["blocking_failures"])
        )

    def test_rollback_when_cost_increases_over_threshold(self):
        baseline = make_summary(cost=0.001)
        candidate = make_summary(cost=0.0013)

        gate = evaluate_release_gate(baseline, candidate)

        self.assertEqual(gate["decision"], "Rollback")
        self.assertTrue(
            any(
                f["name"] == "avg_eval_cost_usd_increase"
                for f in gate["blocking_failures"]
            )
        )

    def test_rollback_when_latency_increases_over_threshold(self):
        baseline = make_summary(avg_latency=0.5, p95_latency=0.7)
        candidate = make_summary(avg_latency=0.7, p95_latency=0.8)

        gate = evaluate_release_gate(baseline, candidate)

        self.assertEqual(gate["decision"], "Rollback")
        self.assertTrue(
            any(
                f["name"] == "avg_latency_seconds_increase"
                for f in gate["blocking_failures"]
            )
        )

    def test_zero_baseline_cost_does_not_crash(self):
        baseline = make_summary(cost=0.0)
        candidate = make_summary(cost=0.01)

        gate = evaluate_release_gate(baseline, candidate)

        self.assertEqual(gate["decision"], "Release")
        self.assertTrue(
            any(w["name"] == "avg_eval_cost_usd_increase_skipped" for w in gate["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
