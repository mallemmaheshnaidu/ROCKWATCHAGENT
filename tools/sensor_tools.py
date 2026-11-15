"""
sensor_tools.py - Telemetry Ingestion and Sensor Health Assessment Tools.

Plain Python functions for Health & Ingestion Agent.
"""

from typing import Any, Dict


def ingest_sensor_frame(zone_id: str, raw_frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes and cleans telemetry inputs for a target pit zone.
    Handles missing or null sensor feeds without crashing (graceful degradation).
    """
    clean_data = {
        "zone_id": zone_id,
        "radar_velocity_mm_hr": raw_frame.get("radar_velocity_mm_hr"),
        "cumulative_displacement_mm": raw_frame.get("cumulative_displacement_mm", 0.0),
        "rainfall_3h_mm": raw_frame.get("rainfall_3h_mm", 0.0),
        "vibration_ppv_mms": raw_frame.get("vibration_ppv_mms", 0.0),
        "pore_pressure_kpa": raw_frame.get("pore_pressure_kpa"),
        "extensometer_velocity_mm_hr": raw_frame.get("extensometer_velocity_mm_hr"),
        "timestamp": raw_frame.get("timestamp", 0.0),
    }

    # Derive effective displacement velocity from available sensors
    # If radar is missing, fallback to crack extensometer or zero
    if clean_data["radar_velocity_mm_hr"] is not None:
        clean_data["effective_velocity_mm_hr"] = float(clean_data["radar_velocity_mm_hr"])
        clean_data["primary_sensor_source"] = "RADAR"
    elif clean_data["extensometer_velocity_mm_hr"] is not None:
        clean_data["effective_velocity_mm_hr"] = float(clean_data["extensometer_velocity_mm_hr"])
        clean_data["primary_sensor_source"] = "EXTENSOMETER"
    else:
        clean_data["effective_velocity_mm_hr"] = 0.0
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

    missing_modalities = []
    if not has_radar:
        missing_modalities.append("RADAR")
    if not has_weather:
        missing_modalities.append("WEATHER")
    if not has_seismic:
        missing_modalities.append("SEISMIC")
    if not has_extensometer:
        missing_modalities.append("EXTENSOMETER")

    # Assess overall coverage tier
    if has_radar and has_weather and has_seismic:
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
        "caution_elevation_required": caution_elevation_required,
        "uncertainty_penalty": uncertainty_penalty,
        "primary_source": telemetry_dict.get("primary_sensor_source", "UNKNOWN"),
    }
