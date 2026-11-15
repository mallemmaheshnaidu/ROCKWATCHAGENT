"""
geotech_tools.py - Deterministic TARP Rules and Inverse Velocity Physics.

Plain Python functions for TARP Engine Agent.
"""

from typing import Any, Dict, List
from blackboard import TARP_SAFETY_FLOOR_VELOCITY_MM_HR


def evaluate_tarp(
    zone_id: str,
    velocity_mm_hr: float,
    rainfall_3h_mm: float = 0.0,
    sensor_health_degraded: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates deterministic TARP rules signed off by geotechnical engineers (docs/tarp_sop_v1.md).
    Enforces TARP-SOP-05 (sensor degradation tier bump) and TARP-SOP-06 (inviolable Red floor).
    """
    v = max(0.0, float(velocity_mm_hr))

    # Base TARP evaluation
    if v >= TARP_SAFETY_FLOOR_VELOCITY_MM_HR:
        base_level = "RED"
        action = "EVACUATE_PERSONNEL_AND_EQUIPMENT_200M"
        passage_id = "[TARP-SOP-04]"
    elif v >= 5.0:
        base_level = "ORANGE"
        action = "STOP_MINING_RESTRICT_ACCESS_BERM_EXCLUSION"
        passage_id = "[TARP-SOP-03]"
    elif v >= 1.0 or rainfall_3h_mm >= 15.0:
        base_level = "YELLOW"
        action = "HEIGHTENED_MONITORING_INCREASE_SCAN_RATE"
        passage_id = "[TARP-SOP-02]"
    else:
        base_level = "GREEN"
        action = "ROUTINE_MONITORING"
        passage_id = "[TARP-SOP-01]"

    final_level = base_level

    # Enforce TARP-SOP-05: Sensor degradation automatically raises tier by one level
    if sensor_health_degraded:
        if base_level == "GREEN":
            final_level = "YELLOW"
            action += " (ELEVATED_DUE_TO_SENSOR_DEGRADATION)"
        elif base_level == "YELLOW":
            final_level = "ORANGE"
            action += " (ELEVATED_DUE_TO_SENSOR_DEGRADATION)"

    return {
        "zone_id": zone_id,
        "base_level": base_level,
        "final_level": final_level,
        "velocity_mm_hr": v,
        "rainfall_3h_mm": rainfall_3h_mm,
        "mandatory_action": action,
        "rule_passage_id": passage_id,
        "is_evacuation_level": final_level == "RED",
    }


def compute_inverse_velocity(
    recent_velocities: List[float],
    time_step_hours: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes Fukuzono inverse-velocity trend (1/v) to screen for accelerating collapse.
    Returns estimated hours to failure if linear extrapolation trends toward zero.
    """
    valid_v = [max(0.01, float(v)) for v in recent_velocities if v is not None]
    if len(valid_v) < 2:
        return {
            "inv_v_trend": "INSUFFICIENT_DATA",
            "hours_to_failure": None,
            "is_accelerating": False,
        }

    inv_v = [1.0 / v for v in valid_v]
    # Simple linear slope across last points
    delta_inv_v = inv_v[-1] - inv_v[0]
    total_time = (len(inv_v) - 1) * time_step_hours
    slope = delta_inv_v / max(0.1, total_time)

    # Accelerating failure means v is increasing, so 1/v is decreasing (negative slope)
    is_accelerating = slope < -0.01
    estimated_hours = None
    if is_accelerating and abs(slope) > 1e-4:
        # Time when 1/v reaches 0: t_fail = current_1_v / (-slope)
        estimated_hours = round(inv_v[-1] / (-slope), 1)

    return {
        "current_velocity_mm_hr": valid_v[-1],
        "current_inv_v": round(inv_v[-1], 4),
        "inv_v_slope": round(slope, 5),
        "is_accelerating": is_accelerating,
        "estimated_hours_to_failure": estimated_hours,
        "fukuzono_critical": is_accelerating and (estimated_hours is not None and estimated_hours <= 12.0),
    }
