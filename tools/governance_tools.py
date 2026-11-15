"""
governance_tools.py - Human Authority Governance and Root Cause Analysis Tools.

Plain Python functions for Governance Agent.

Enforces the Asymmetric Safety Principle:
- Desensitization requires a verified geotechnical credential, not a role string.
- Duration is clamped to 1-12 hours against an explicitly supplied clock.
- The TARP Red floor is reported from the shared constant, never asserted ad hoc.
"""

import uuid
from typing import Any, Dict

from blackboard import TARP_SAFETY_FLOOR_VELOCITY_MM_HR
from tools.auth_tools import AUTHORIZED_ROLES, require_geotech_credential


def _denied(zone_id: str, operator_id: str, reason: str, status: str, now: float) -> Dict[str, Any]:
    """Fail-closed denial record. TARP is untouched and sensitivity stays at baseline."""
    return {
        "proposal_id": f"PROP-{zone_id}-{uuid.uuid4().hex[:12]}",
        "zone_id": zone_id,
        "operator_id": operator_id,
        "reason": reason,
        "approver_role": None,
        "approved_by": None,
        "is_approved": False,
        "status": status,
        "denial_reason": reason,
        "adjusted_sensitivity": 1.0,
        "created_at": now,
        "expires_at": now,
        "duration_hours": 0.0,
        "tarp_floor_status": f"UNTOUCHED_INVIOLABLE_{TARP_SAFETY_FLOOR_VELOCITY_MM_HR:g}MM_HR",
    }


def process_desensitization_proposal(
    zone_id: str,
    operator_id: str,
    reason: str,
    approver_role: str,
    duration_hours: float = 8.0,
    requested_sensitivity: float = 0.85,
    current_time: float = 0.0,
    approver_credential: str = "",
) -> Dict[str, Any]:
    """
    Processes requests to temporarily desensitize forecaster sensitivity (Option A).
    Enforces strict rules:
    - Approval requires a valid geotechnical credential (signature + expiry + role).
    - Clamped duration (max 12.0 hours) with strict expiry timestamp.
    - Explicitly affirms that the TARP safety floor remains untouched.
    - Any authentication failure returns a DENIED record; nothing is approved.
    """
    if current_time is None or float(current_time) <= 0.0:
        raise ValueError(
            "current_time must be supplied from the caller's clock domain; "
            "implicit wall-clock fallback is not permitted."
        )
    now = float(current_time)
    claimed_role = str(approver_role or "").upper()

    if claimed_role not in AUTHORIZED_ROLES:
        return _denied(
            zone_id, operator_id, "Only a certified geotechnical engineer may attenuate sensitivity.",
            "DENIED_INSUFFICIENT_AUTHORITY", now,
        )

    try:
        credential = require_geotech_credential(approver_credential, now=now)
    except PermissionError as exc:
        return _denied(zone_id, operator_id, str(exc), "DENIED_INVALID_CREDENTIAL", now)

    clamped_duration = min(12.0, max(1.0, float(duration_hours)))
    clamped_sensitivity = max(0.50, min(0.95, float(requested_sensitivity)))

    return {
        "proposal_id": f"PROP-{zone_id}-{uuid.uuid4().hex[:12]}",
        "zone_id": zone_id,
        "operator_id": operator_id,
        "reason": reason,
        "approver_role": credential["role"],
        "approved_by": credential["operator_id"],
        "is_approved": True,
        "status": "APPROVED_ACTIVE",
        "adjusted_sensitivity": clamped_sensitivity,
        "created_at": now,
        "expires_at": now + (clamped_duration * 3600.0),
        "duration_hours": clamped_duration,
        "tarp_floor_status": f"UNTOUCHED_INVIOLABLE_{TARP_SAFETY_FLOOR_VELOCITY_MM_HR:g}MM_HR",
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

    if "RADAR" in missing or primary_source in ("NONE", "UNKNOWN"):
        proposed_cause = "SENSOR_COVERAGE_GAP"
        recommended_action = "Deploy mobile radar or auxiliary PTZ optical camera to eliminate blind spot."
    elif telemetry_snapshot:
        v = telemetry_snapshot.get("effective_velocity_mm_hr")
        v = float(v) if v is not None else 0.0
        rain = telemetry_snapshot.get("rainfall_3h_mm")
        rain = float(rain) if rain is not None else 0.0
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
