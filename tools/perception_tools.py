"""
perception_tools.py - Surface Change and Unalarmed Event Detection Tools.

Plain Python functions for Perception Agent.
"""

from typing import Any, Dict, List


def detect_surface_change(
    zone_id: str,
    baseline_scan_volume_m3: float,
    current_scan_volume_m3: float,
    detection_threshold_m3: float = 2.0,
) -> Dict[str, Any]:
    """
    Computes volumetric surface change from 3-D scan differencing (e.g. repeat LiDAR/SfM).
    Flags a physical detachment if volume reduction exceeds the noise threshold.
    """
    volume_diff = baseline_scan_volume_m3 - current_scan_volume_m3
    detached_volume = max(0.0, volume_diff)
    detachment_detected = detached_volume >= detection_threshold_m3

    return {
        "zone_id": zone_id,
        "detachment_detected": detachment_detected,
        "volume_detached_m3": round(detached_volume, 2),
        "threshold_m3": detection_threshold_m3,
        "change_type": "DETACHMENT" if detachment_detected else "STABLE",
    }


def find_unalarmed_changes(
    zone_id: str,
    detected_change: Dict[str, Any],
    recent_alerts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compares verified physical surface detachments against prior alarm history.
    If a detachment occurred without a preceding Level 2+ warning, flags a Missed Event.
    """
    if not detected_change.get("detachment_detected", False):
        return {
            "zone_id": zone_id,
            "is_unalarmed_miss": False,
            "reason": "NO_DETACHMENT",
        }

    # Check if any alert was issued for this zone in recent history
    zone_alerts = [
        a for a in recent_alerts
        if a.get("zone_id") == zone_id and a.get("alert_level") in ("YELLOW", "ORANGE", "RED")
    ]

    is_miss = len(zone_alerts) == 0

    return {
        "zone_id": zone_id,
        "is_unalarmed_miss": is_miss,
        "volume_m3": detected_change.get("volume_detached_m3", 0.0),
        "preceding_alerts_count": len(zone_alerts),
        "classification": "MISSED_EVENT" if is_miss else "WARNED_EVENT",
        "action_required": "INSTANT_CAUTION_ELEVATION" if is_miss else "STANDARD_REVIEW",
    }
