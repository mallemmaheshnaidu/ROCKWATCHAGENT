"""
evaluation_tools.py - Model Quality Gate and Quiet Period Auditing Tools.

Plain Python functions for Model Evaluator & QA Agent.
Implements the Champion vs. Challenger validation gate with strict non-inferiority recall constraints.
"""

from typing import Any, Dict, List


def evaluate_model_gate(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    recall_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluates candidate retrained model against the frozen golden baseline.
    Enforces the Non-Inferiority Safety Rule:
    The challenger model CANNOT drop recall on confirmed events (Recall_challenger >= Recall_champion).
    """
    champ_recall = champion_metrics.get("recall", 1.0)
    chal_recall = challenger_metrics.get("recall", 0.0)

    champ_f1 = champion_metrics.get("f1_score", 0.0)
    chal_f1 = challenger_metrics.get("f1_score", 0.0)

    champ_precision = champion_metrics.get("precision", 0.0)
    chal_precision = challenger_metrics.get("precision", 0.0)

    # Strict safety gate: recall must not drop
    recall_delta = chal_recall - champ_recall
    f1_delta = chal_f1 - champ_f1
    precision_delta = chal_precision - champ_precision

    passed_recall_gate = recall_delta >= (-recall_tolerance)

    if not passed_recall_gate:
        status = "REJECTED_SAFETY_REGRESSION"
        verdict = f"Challenger rejected: recall dropped by {abs(recall_delta):.3f}. Safety recall cannot degrade."
        promoted = False
    elif chal_f1 < champ_f1:
        status = "REJECTED_LOWER_OVERALL_ACCURACY"
        verdict = f"Challenger rejected: F1 score ({chal_f1:.3f}) lower than champion ({champ_f1:.3f})."
        promoted = False
    else:
        status = "PASSED_BENCHMARK_GATE"
        verdict = (
            f"Challenger passed benchmark gate: Recall maintained ({chal_recall:.3f}), "
            f"Precision improved by {precision_delta:+.3f}. Ready for geotech deployment sign-off."
        )
        promoted = True

    return {
        "status": status,
        "promoted": promoted,
        "verdict_summary": verdict,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "recall_delta": round(recall_delta, 4),
        "f1_delta": round(f1_delta, 4),
        "precision_delta": round(precision_delta, 4),
    }


def sample_quiet_periods(
    total_operational_hours: int,
    alarm_timestamps: List[float],
    sample_size: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Samples un-alarmed quiet operational windows for human audit to prevent label bias.
    Ensures negatives are verified and labelled alongside positive alarms.
    """
    # Deterministic pseudo-random selection without external dependencies
    sampled = []
    step = max(1, total_operational_hours // (sample_size + 1))
    for i in range(1, sample_size + 1):
        target_hour = (i * step + seed) % total_operational_hours
        has_alarm = any(abs(target_hour - a) < 2.0 for a in alarm_timestamps)
        sampled.append({
            "window_id": f"QUIET-WIN-{target_hour:04d}",
            "operational_hour": target_hour,
            "alarm_present": has_alarm,
            "status": "QUEUED_FOR_GEOTECH_NEGATIVE_CONFIRMATION",
        })
    return sampled
