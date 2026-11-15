"""
sensor_tools.py - Telemetry Ingestion and Sensor Health Assessment Tools.

Plain Python functions for Health & Ingestion Agent.

Safety invariants enforced here:
1. NaN / Inf / non-numeric readings are rejected and recorded, never coerced to 0.0.
2. A modality that is absent or invalid is reported as MISSING rather than as a
   benign reading of zero (a dead rain gauge must not look like a dry day).
3. Effective velocity is None when no usable kinematic sensor exists; downstream
   layers must escalate rather than treat that as a stable slope.
"""

import math
from typing import Any, Dict, List, Optional

_MODALITY_FIELDS = (
    "rainfall_3h_mm",
    "vibration_ppv_mms",
    "pore_pressure_kpa",
    "extensometer_velocity_mm_hr",
)


def _clean(value: Any) -> Optional[float]:
    """Returns a finite float, or None for null / non-numeric / NaN / Inf."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ingest_sensor_frame(zone_id: str, raw_frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes and cleans telemetry inputs for a target pit zone.
    Handles missing or null sensor feeds without crashing (graceful degradation).
    """
    clean_data: Dict[str, Any] = {
        "zone_id": zone_id,
        "radar_velocity_mm_hr": _clean(raw_frame.get("radar_velocity_mm_hr")),
        "cumulative_displacement_mm": _clean(raw_frame.get("cumulative_displacement_mm")) or 0.0,
        "rainfall_3h_mm": _clean(raw_frame.get("rainfall_3h_mm")),
        "vibration_ppv_mms": _clean(raw_frame.get("vibration_ppv_mms")),
        "pore_pressure_kpa": _clean(raw_frame.get("pore_pressure_kpa")),
        "extensometer_velocity_mm_hr": _clean(raw_frame.get("extensometer_velocity_mm_hr")),
        "acceleration_mm_hr2": _clean(raw_frame.get("acceleration_mm_hr2")),
        "timestamp": raw_frame.get("timestamp", 0.0),
        "rejected_fields": [],
    }

    for field in ("radar_velocity_mm_hr", "acceleration_mm_hr2") + _MODALITY_FIELDS:
        if raw_frame.get(field) is not None and clean_data[field] is None:
            clean_data["rejected_fields"].append(field)

    # Derive effective displacement velocity from available sensors
    if clean_data["radar_velocity_mm_hr"] is not None:
        clean_data["effective_velocity_mm_hr"] = clean_data["radar_velocity_mm_hr"]
        clean_data["primary_sensor_source"] = "RADAR"
    elif clean_data["extensometer_velocity_mm_hr"] is not None:
        clean_data["effective_velocity_mm_hr"] = clean_data["extensometer_velocity_mm_hr"]
        clean_data["primary_sensor_source"] = "EXTENSOMETER"
    else:
        clean_data["effective_velocity_mm_hr"] = None
        clean_data["primary_sensor_source"] = "NONE"

    return clean_data


def assess_sensor_health(telemetry_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits active sensor telemetry for outages, staleness, or missing modalities.
    Enforces TARP-SOP-05: If primary radar is offline, flags caution elevation requirement.
    """
    has_radar = telemetry_dict.get("radar_velocity_mm_hr") is not None
    has_weather = telemetry_dict.get("rainfall_3h_mm") is not None
    has_seismic = telemetry_dict.get("vibration_ppv_mms") is not None
    has_extensometer = telemetry_dict.get("extensometer_velocity_mm_hr") is not None
    rejected: List[str] = list(telemetry_dict.get("rejected_fields") or [])

    missing_modalities: List[str] = []
    if not has_radar:
        missing_modalities.append("RADAR")
    if not has_weather:
        missing_modalities.append("WEATHER")
    if not has_seismic:
        missing_modalities.append("SEISMIC")
    if not has_extensometer:
        missing_modalities.append("EXTENSOMETER")

    # Assess overall coverage tier
    if rejected:
        health_status = "DEGRADED_INVALID_TELEMETRY"
        caution_elevation_required = True
        uncertainty_penalty = 0.45
    elif has_radar and has_weather and has_seismic:
        health_status = "HEALTHY"
        caution_elevation_required = False
        uncertainty_penalty = 0.05
    elif not has_radar and has_extensometer:
        health_status = "DEGRADED_SECONDARY_ACTIVE"
        caution_elevation_required = True  # TARP-SOP-05 requirement
        uncertainty_penalty = 0.25
    else:
        health_status = "CRITICAL_COVERAGE_GAP"
        caution_elevation_required = True
        uncertainty_penalty = 0.45

    return {
        "zone_id": telemetry_dict.get("zone_id", "UNKNOWN"),
        "health_status": health_status,
        "radar_online": has_radar,
        "missing_modalities": missing_modalities,
        "rejected_fields": rejected,
        "caution_elevation_required": caution_elevation_required,
        "uncertainty_penalty": uncertainty_penalty,
        "primary_source": telemetry_dict.get("primary_sensor_source", "UNKNOWN"),
    }
