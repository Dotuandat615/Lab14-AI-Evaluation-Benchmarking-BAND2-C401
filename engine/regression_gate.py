from copy import deepcopy
from typing import Any, Dict, Optional


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "avg_score_min": 4.0,
    "hit_rate_min": 0.80,
    "agreement_rate_min": 0.70,
    "max_avg_score_drop": 0.10,
    "max_hit_rate_drop": 0.02,
    "max_mrr_drop": 0.02,
    "max_cost_increase_pct": 0.20,
    "max_avg_latency_increase_pct": 0.25,
    "max_p95_latency_increase_pct": 0.25,
    "max_total_runtime_seconds": 120.0,
}


DELTA_METRICS = [
    "avg_score",
    "hit_rate",
    "mrr",
    "agreement_rate",
    "faithfulness",
    "relevancy",
    "avg_eval_cost_usd",
    "avg_latency_seconds",
    "p95_latency_seconds",
    "total_runtime_seconds",
]


def evaluate_release_gate(
    v1_summary: Dict[str, Any],
    v2_summary: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare V1 and V2 summaries and decide Release or Rollback."""
    active_thresholds = deepcopy(DEFAULT_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    checks = []
    warnings = []

    def add_check(
        name: str,
        passed: bool,
        baseline: Optional[float],
        candidate: Optional[float],
        threshold: float,
        message: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "baseline": baseline,
                "candidate": candidate,
                "threshold": threshold,
                "message": message,
            }
        )

    v1_metrics = v1_summary.get("metrics", {})
    v2_metrics = v2_summary.get("metrics", {})

    avg_score = _metric(v2_summary, "avg_score")
    hit_rate = _metric(v2_summary, "hit_rate")
    agreement_rate = _metric(v2_summary, "agreement_rate")

    add_check(
        "candidate_avg_score_min",
        avg_score >= active_thresholds["avg_score_min"],
        None,
        avg_score,
        active_thresholds["avg_score_min"],
        "Candidate average judge score must meet the minimum quality bar.",
    )
    add_check(
        "candidate_hit_rate_min",
        hit_rate >= active_thresholds["hit_rate_min"],
        None,
        hit_rate,
        active_thresholds["hit_rate_min"],
        "Candidate retrieval hit rate must meet the minimum quality bar.",
    )
    add_check(
        "candidate_agreement_rate_min",
        agreement_rate >= active_thresholds["agreement_rate_min"],
        None,
        agreement_rate,
        active_thresholds["agreement_rate_min"],
        "Multi-judge agreement must be high enough for a reliable release.",
    )

    _add_drop_check(
        checks,
        "avg_score_regression",
        v1_metrics,
        v2_metrics,
        "avg_score",
        active_thresholds["max_avg_score_drop"],
    )
    _add_drop_check(
        checks,
        "hit_rate_regression",
        v1_metrics,
        v2_metrics,
        "hit_rate",
        active_thresholds["max_hit_rate_drop"],
    )
    _add_drop_check(
        checks,
        "mrr_regression",
        v1_metrics,
        v2_metrics,
        "mrr",
        active_thresholds["max_mrr_drop"],
    )

    _add_increase_check(
        checks,
        warnings,
        "avg_eval_cost_usd_increase",
        v1_summary,
        v2_summary,
        "avg_eval_cost_usd",
        active_thresholds["max_cost_increase_pct"],
    )
    _add_increase_check(
        checks,
        warnings,
        "avg_latency_seconds_increase",
        v1_summary,
        v2_summary,
        "avg_latency_seconds",
        active_thresholds["max_avg_latency_increase_pct"],
    )
    _add_increase_check(
        checks,
        warnings,
        "p95_latency_seconds_increase",
        v1_summary,
        v2_summary,
        "p95_latency_seconds",
        active_thresholds["max_p95_latency_increase_pct"],
    )

    runtime = _metric(v2_summary, "total_runtime_seconds")
    add_check(
        "candidate_total_runtime_seconds",
        runtime <= active_thresholds["max_total_runtime_seconds"],
        _metric(v1_summary, "total_runtime_seconds"),
        runtime,
        active_thresholds["max_total_runtime_seconds"],
        "Candidate benchmark runtime must stay within the lab performance budget.",
    )

    blocking_failures = [check for check in checks if check["status"] == "fail"]
    decision = "Release" if not blocking_failures else "Rollback"

    return {
        "decision": decision,
        "baseline_version": v1_summary.get("metadata", {}).get("version", "unknown"),
        "candidate_version": v2_summary.get("metadata", {}).get("version", "unknown"),
        "deltas": _build_deltas(v1_summary, v2_summary),
        "checks": checks,
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "thresholds": active_thresholds,
    }


def _metric(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = summary.get("metrics", {})
    if key in metrics:
        return float(metrics.get(key) or default)
    if key == "avg_eval_cost_usd":
        return float(summary.get("cost", {}).get("avg_cost_per_eval_usd") or default)
    return default


def _add_drop_check(
    checks: list,
    name: str,
    v1_metrics: Dict[str, Any],
    v2_metrics: Dict[str, Any],
    metric: str,
    max_drop: float,
) -> None:
    baseline = float(v1_metrics.get(metric) or 0.0)
    candidate = float(v2_metrics.get(metric) or 0.0)
    delta = candidate - baseline
    checks.append(
        {
            "name": name,
            "status": "pass" if delta >= -max_drop else "fail",
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "threshold": -max_drop,
            "message": f"{metric} must not drop by more than {max_drop}.",
        }
    )


def _add_increase_check(
    checks: list,
    warnings: list,
    name: str,
    v1_summary: Dict[str, Any],
    v2_summary: Dict[str, Any],
    metric: str,
    max_increase_pct: float,
) -> None:
    baseline = _metric(v1_summary, metric)
    candidate = _metric(v2_summary, metric)
    delta = candidate - baseline

    if baseline <= 0:
        warnings.append(
            {
                "name": f"{name}_skipped",
                "message": (
                    f"Skipped percent check for {metric} because baseline is zero."
                ),
                "baseline": baseline,
                "candidate": candidate,
            }
        )
        checks.append(
            {
                "name": name,
                "status": "pass",
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "pct_change": None,
                "threshold": max_increase_pct,
                "message": f"{metric} percent increase check skipped safely.",
            }
        )
        return

    pct_change = delta / baseline
    checks.append(
        {
            "name": name,
            "status": "pass" if pct_change <= max_increase_pct else "fail",
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "pct_change": pct_change,
            "threshold": max_increase_pct,
            "message": (
                f"{metric} must not increase by more than "
                f"{max_increase_pct * 100:.1f}%."
            ),
        }
    )


def _build_deltas(
    v1_summary: Dict[str, Any],
    v2_summary: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    deltas = {}
    for metric in DELTA_METRICS:
        baseline = _metric(v1_summary, metric)
        candidate = _metric(v2_summary, metric)
        delta = candidate - baseline
        pct_change = None if baseline == 0 else delta / baseline
        deltas[metric] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "pct_change": pct_change,
        }
    return deltas
