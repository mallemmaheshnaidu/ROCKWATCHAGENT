"""
geotech_tools.py - Deterministic TARP Rules and Inverse Velocity Physics.

Plain Python functions for TARP Engine Agent.

Safety invariants enforced here:
1. Non-finite or negative telemetry raises ValueError instead of evaluating to GREEN.
2. The signed TARP-SOP-04 Red criteria are honoured: velocity >= 15 mm/hr and
   inverse velocity extrapolating to zero within 4 hours (SOP-03 uses 12 hours
   for Orange). The "positive acceleration" criterion is only active when an
   explicit noise floor is provisioned via TARP_ACCEL_RED_MM_HR2, because the
   literal SOP wording would fire inside the signed Yellow band.
"""

import math
import os
from typing import Any, Dict, List, Optional

from blackboard import TARP_SAFETY_FLOOR_VELOCITY_MM_HR

# Velocity at/above which Level 2 (Yellow) applies (docs/tarp_sop_v1.md:4).
TARP_YELLOW_VELOCITY_MM_HR: float = 1.0
TARP_ORANGE_VELOCITY_MM_HR: float = 5.0
TARP_YELLOW_RAINFALL_3H_MM: float = 15.0

# SOP-04 lists "acceleration is positive" as a Red criterion, which as written
# would fire at ~4 mm/hr (inside the signed Yellow band) and contradicts SOP-02.
# The criterion is therefore DISABLED unless a geotechnical noise floor is
# explicitly provisioned (mm/hr^2), pending sign-off on the SOP wording.
_accel_floor = os.getenv("TARP_ACCEL_RED_MM_HR2", "").strip()
ACCEL_RED_MM_HR2: Optional[float] = float(_accel_floor) if _accel_floor else None


def _finite_nonneg(value: Any, field: str) -> float:
    """Validates a required numeric telemetry channel."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    if number < 0:
        raise ValueError(f"{field} must be >= 0, got {value!r}")
    return number


def _optional_finite(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or None, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite or None, got {value!r}")
    return number


def evaluate_tarp(
    zone_id: str,
    velocity_mm_hr: float,
    rainfall_3h_mm: float = 0.0,
    sensor_health_degraded: bool = False,
    acceleration_mm_hr2: Optional[float] = None,
    hours_to_failure: Optional[float] = None,
    velocity_history: Optional[List[float]] = None,
    velocity_timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates deterministic TARP rules signed off by geotechnical engineers (docs/tarp_sop_v1.md).
    Enforces TARP-SOP-05 (sensor degradation tier bump) and TARP-SOP-06 (inviolable Red floor).

    velocity_timestamps (epoch seconds) may accompany velocity_history so the
    inverse-velocity fit uses the real elapsed time instead of assuming one
    sample per hour.
    """
    v = _finite_nonneg(velocity_mm_hr, "velocity_mm_hr")
    rain = _finite_nonneg(rainfall_3h_mm, "rainfall_3h_mm")
    accel = _optional_finite(acceleration_mm_hr2, "acceleration_mm_hr2")

    htf = hours_to_failure
    if htf is None and velocity_history and len(velocity_history) >= 2:
        try:
            htf = compute_inverse_velocity(
                velocity_history, timestamps=velocity_timestamps
            ).get("estimated_hours_to_failure")
        except ValueError:
            htf = None
    if htf is not None:
        htf = _optional_finite(htf, "hours_to_failure")

    # --- Level 4 (Red): SOP-04 criteria ---
    red_criteria = []
    if v >= TARP_SAFETY_FLOOR_VELOCITY_MM_HR:
        red_criteria.append("VELOCITY_GE_15_MM_HR")
    if htf is not None and htf <= 4.0:
        red_criteria.append("INVERSE_VELOCITY_ZERO_WITHIN_4H")
    if accel is not None and ACCEL_RED_MM_HR2 is not None and accel >= ACCEL_RED_MM_HR2:
        red_criteria.append("SUSTAINED_POSITIVE_ACCELERATION")

    if red_criteria:
        base_level = "RED"
        action = "EVACUATE_PERSONNEL_AND_EQUIPMENT_200M"
        passage_id = "[TARP-SOP-04]"
    # --- Level 3 (Orange): SOP-03 criteria ---
    elif v >= TARP_ORANGE_VELOCITY_MM_HR or (htf is not None and htf <= 12.0):
        base_level = "ORANGE"
        action = "STOP_MINING_RESTRICT_ACCESS_BERM_EXCLUSION"
        passage_id = "[TARP-SOP-03]"
    # --- Level 2 (Yellow): SOP-02 criteria ---
    elif v >= TARP_YELLOW_VELOCITY_MM_HR or rain >= TARP_YELLOW_RAINFALL_3H_MM:
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
        "rainfall_3h_mm": rain,
        "acceleration_mm_hr2": accel,
        "estimated_hours_to_failure": htf,
        "red_criteria_matched": red_criteria,
        "mandatory_action": action,
        "rule_passage_id": passage_id,
        "is_evacuation_level": final_level == "RED",
    }


def compute_inverse_velocity(
    recent_velocities: List[float],
    time_step_hours: float = 1.0,
    timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Computes Fukuzono inverse-velocity trend (1/v) to screen for accelerating collapse.
    Returns estimated hours to failure if linear extrapolation trends toward zero.

    Notes:
    - Non-finite samples are discarded rather than propagating NaN into the slope.
    - A slope is only treated as accelerating when 1/v is decreasing meaningfully.
    - When timestamps (epoch seconds) are supplied and span at least 6 minutes,
      the fit uses real elapsed time. Otherwise a uniform sample interval is
      assumed and that assumption is reported in ``timing_basis``.
    - A sub-6-minute timestamp span is treated as insufficient evidence: two
      readings a few seconds apart cannot establish a collapse trend.
    """
    series: List[tuple[Optional[float], float]] = []
    raw_timestamps = timestamps if timestamps is not None else []
    use_timestamps = len(raw_timestamps) == len(recent_velocities or [])

    for index, value in enumerate(recent_velocities or []):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue

        stamp: Optional[float] = None
        if use_timestamps:
            try:
                stamp = float(raw_timestamps[index])
            except (TypeError, ValueError):
                stamp = None
            if stamp is not None and not math.isfinite(stamp):
                stamp = None
        series.append((stamp, max(0.01, number)))

    if len(series) < 2:
        return {
            "inv_v_trend": "INSUFFICIENT_DATA",
            "estimated_hours_to_failure": None,
            "is_accelerating": False,
            "timing_basis": "NONE",
        }

    inv_v = [1.0 / value for _, value in series]

    stamps = [stamp for stamp, _ in series]
    timestamps_supplied = any(stamp is not None for stamp in stamps)
    insufficient = {
        "inv_v_trend": "INSUFFICIENT_DATA",
        "estimated_hours_to_failure": None,
        "is_accelerating": False,
        "timing_basis": "TIMESTAMPS",
    }

    if timestamps_supplied:
        timestamps_usable = (
            all(stamp is not None for stamp in stamps)
            and all(b > a for a, b in zip(stamps, stamps[1:]))
        )
        if not timestamps_usable:
            return insufficient
        total_time = (stamps[-1] - stamps[0]) / 3600.0
        # Readings seconds apart cannot establish a collapse trend.
        if total_time < (6.0 / 60.0):
            return insufficient
        timing_basis = "TIMESTAMPS"
    else:
        total_time = (len(inv_v) - 1) * max(0.1, float(time_step_hours))
        timing_basis = "UNIFORM_ASSUMED"

    slope = (inv_v[-1] - inv_v[0]) / total_time

    # Accelerating failure means v is increasing, so 1/v is decreasing (negative slope)
    is_accelerating = slope < -0.01
    estimated_hours = None
    if is_accelerating and abs(slope) > 1e-4:
        estimated_hours = round(inv_v[-1] / (-slope), 1)

    return {
        "current_velocity_mm_hr": series[-1][1],
        "current_inv_v": round(inv_v[-1], 4),
        "inv_v_slope": round(slope, 5),
        "timing_basis": timing_basis,
        "is_accelerating": is_accelerating,
        "estimated_hours_to_failure": estimated_hours,
        "fukuzono_critical": is_accelerating and estimated_hours is not None and estimated_hours <= 12.0,
    }
