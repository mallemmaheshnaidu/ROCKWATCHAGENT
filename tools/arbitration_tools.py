"""
arbitration_tools.py - Decision Arbitration and Alert Package Builder Tools.

Plain Python functions for Arbiter & Responder Agent.
"""

from typing import Any, Dict, List

# Strict caution ranking
ALERT_RANK = {
    "GREEN": 1,
    "LOW": 1,
    "YELLOW": 2,
    "MODERATE": 2,
    "ORANGE": 3,
    "RED": 4,
    "HIGH": 4,
}


def arbitrate_decision(
    tarp_level: str,
    forecast_tier: str,
    sensor_health_status: str = "HEALTHY",
) -> Dict[str, Any]:
    """
    Arbitrates between deterministic TARP rules and ML risk forecast.
    Enforces two strict principles:
    1. Asymmetric Caution: Always selects the higher risk tier between TARP and Model.
    2. Discrepancy Flagging: Any divergence between rule and model triggers 'REVIEW_NEEDED'.
    """
    tarp_clean = tarp_level.upper()
    forecast_clean = forecast_tier.upper()

    tarp_rank = ALERT_RANK.get(tarp_clean, 1)
    forecast_rank = ALERT_RANK.get(forecast_clean, 1)

    # Detect divergence
    discrepancy = tarp_rank != forecast_rank
    max_rank = max(tarp_rank, forecast_rank)

    # Map rank back to standard operational level
    rank_to_level = {1: "GREEN", 2: "YELLOW", 3: "ORANGE", 4: "RED"}
    effective_level = rank_to_level[max_rank]

    # Additional bump if sensors are degraded and level is still GREEN
    if "DEGRADED" in sensor_health_status and effective_level == "GREEN":
        effective_level = "YELLOW"
        discrepancy = True

    review_needed = discrepancy or (effective_level in ("ORANGE", "RED"))

    return {
        "tarp_level": tarp_clean,
        "forecast_tier": forecast_clean,
        "effective_alert_level": effective_level,
        "discrepancy_detected": discrepancy,
        "review_needed": review_needed,
        "governing_source": "TARP" if tarp_rank >= forecast_rank else "FORECAST_ELEVATED",
        "rationale": (
            "Model and TARP agree." if not discrepancy
            else f"Discrepancy detected (TARP: {tarp_clean}, Forecast: {forecast_clean}). Enforced higher caution: {effective_level}."
        ),
    }


def build_alert_package(
    zone_id: str,
    arbitrated_result: Dict[str, Any],
    geofenced_assets: List[Dict[str, Any]],
    sop_citation: str = "[TARP-SOP-01]",
) -> Dict[str, Any]:
    """
    Constructs a structured, human-readable alert package for mine dispatch and audit log.
    Identifies vehicles/personnel inside the hazard geofence without issuing autonomous stops.
    """
    level = arbitrated_result.get("effective_alert_level", "GREEN")
    threatened_assets = [
        asset for asset in geofenced_assets
        if asset.get("zone_id") == zone_id
    ]

    action_map = {
        "GREEN": "Normal operations. Maintain 15-min scan cadence.",
        "YELLOW": "Heightened caution. Notify pit supervisor; restrict non-essential foot traffic.",
        "ORANGE": "Operations suspended in zone. Park machinery outside 1.5x bench height.",
        "RED": "IMMEDIATE EVACUATION ADVISORY: Audible alarm, clear 200m danger polygon, barricade ramps.",
    }

    return {
        "zone_id": zone_id,
        "alert_level": level,
        "advisory_directive": action_map.get(level, "Monitor"),
        "threatened_assets_count": len(threatened_assets),
        "threatened_asset_ids": [a.get("asset_id") for a in threatened_assets],
        "governing_source": arbitrated_result.get("governing_source"),
        "review_needed": arbitrated_result.get("review_needed", False),
        "sop_citation": sop_citation,
        "human_authorization_required": level in ("ORANGE", "RED"),
    }
