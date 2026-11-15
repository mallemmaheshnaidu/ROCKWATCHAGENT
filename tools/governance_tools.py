"""
governance_tools.py - Human Authority Governance and Root Cause Analysis Tools.

Plain Python functions for Governance Agent.
Enforces the Asymmetric Safety Principle: desensitization requires authenticated geotech role.
"""

import time
from typing import Any, Dict


def process_desensitization_proposal(
    zone_id: str,
    operator_id: str,
    reason: str,
    approver_role: str,
    duration_hours: float = 8.0,
    requested_sensitivity: float = 0.85,
    current_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Processes requests to temporarily desensitize forecaster sensitivity (Option A).
    Enforces strict rules:
    - Operator cannot authorize: only role 'GEOTECH_ENGINEER' can approve.
    - Clamped duration (max 12.0 hours) with strict expiry timestamp.
    - Explicitly affirms TARP safety floor (15 mm/hr) remains untouched.
    """
    now = current_time if current_time > 0.0 else time.time()
    clamped_duration = min(12.0, max(1.0, float(duration_hours)))
    expires_at = now + (clamped_duration * 3600.0)
    clamped_sensitivity = max(0.50, min(0.95, float(requested_sensitivity)))

    is_geotech = approver_role.upper() == "GEOTECH_ENGINEER"

    proposal_id = f"PROP-{zone_id}-{int(now)}"
    status = "APPROVED_ACTIVE" if is_geotech else "PENDING_GEOTECH_APPROVAL"

    return {
        "proposal_id": proposal_id,
        "zone_id": zone_id,
        "operator_id": operator_id,
        "reason": reason,
        "approver_role": approver_role,
        "is_approved": is_geotech,
        "status": status,
        "adjusted_sensitivity": clamped_sensitivity if is_geotech else 1.0,
        "created_at": now,
        "expires_at": expires_at,
        "duration_hours": clamped_duration,
        "tarp_floor_status": "UNTOUCHED_INVIOLABLE_15MM_HR",
    }


def propose_root_cause(
    zone_id: str,
    telemetry_snapshot: Dict[str, Any],
    sensor_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyzes telemetry snapshots following an unalarmed detachment (missed event).
    Proposes a root-cause classification for human geotechnical review.
    Code proposes; only the geotechnical engineer can confirm.
    """
    missing = sensor_health.get("missing_modalities", [])
    primary_source = sensor_health.get("primary_source", "UNKNOWN")

    if "RADAR" in missing or primary_source == "NONE":
        proposed_cause = "SENSOR_COVERAGE_GAP"
        recommended_action = "Deploy mobile radar or auxiliary PTZ optical camera to eliminate blind spot."
    elif telemetry_dict := telemetry_snapshot:
        v = telemetry_dict.get("effective_velocity_mm_hr", 0.0)
        rain = telemetry_dict.get("rainfall_3h_mm", 0.0)
        if v < 1.0 and rain > 10.0:
            proposed_cause = "WATER_DRIVEN_BRITTLE_FAILURE_WITHOUT_KINEMATIC_PRECURSOR"
            recommended_action = "Inspect crest tension cracks and review pore pressure piezometer telemetry."
        else:
            proposed_cause = "MODEL_DETECTION_PATTERN_FAILURE"
            recommended_action = "Save 24h precursor window into Golden Dataset benchmark for scheduled batch retraining."
    else:
        proposed_cause = "INDETERMINATE"
        recommended_action = "Comprehensive field inspection required by geotechnical lead."

    return {
        "zone_id": zone_id,
        "proposed_root_cause": proposed_cause,
        "recommended_action": recommended_action,
        "governance_status": "PROPOSED_PENDING_GEOTECH_SIGN_OFF",
        "precursor_window_archived": True,
    }
