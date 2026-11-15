"""
arbitration_tools.py - Decision Arbitration and Alert Package Builder Tools.

Plain Python functions for Arbiter & Responder Agent.

Safety invariants enforced here:
1. Alert levels are parsed against a CLOSED vocabulary. Anything unrecognised raises
   UnknownAlertLevel instead of silently defaulting to the lowest rank.
2. A TARP Level 4 (Red) input can never be arbitrated below Red.
"""

import re
from typing import Any, Dict, List

# Single source of truth for the evacuation geometry (docs/tarp_sop_v1.md, TARP-SOP-04).
EVACUATION_RADIUS_M = 200.0


class UnknownAlertLevel(ValueError):
    """Raised when an alert level is outside the signed operational vocabulary."""


# TARP levels (deterministic rule engine output).
_TARP_RANK: Dict[str, int] = {
    "GREEN": 1, "LEVEL 1": 1, "LEVEL 1 GREEN": 1, "L1": 1,
    "YELLOW": 2, "LEVEL 2": 2, "LEVEL 2 YELLOW": 2, "L2": 2,
    "ORANGE": 3, "LEVEL 3": 3, "LEVEL 3 ORANGE": 3, "L3": 3,
    "RED": 4, "LEVEL 4": 4, "LEVEL 4 RED": 4, "L4": 4,
}

# ML forecast tiers.
_FORECAST_RANK: Dict[str, int] = {
    "LOW": 1, "GREEN": 1,
    "MODERATE": 2, "MEDIUM": 2, "YELLOW": 2,
    "HIGH": 4, "SEVERE": 4, "CRITICAL": 4, "RED": 4,
}

_RANK_TO_LEVEL = {1: "GREEN", 2: "YELLOW", 3: "ORANGE", 4: "RED"}

# Kept for backwards compatibility with external imports.
ALERT_RANK = {
    "GREEN": 1, "LOW": 1,
    "YELLOW": 2, "MODERATE": 2,
    "ORANGE": 3,
    "RED": 4, "HIGH": 4,
}


def _normalise_key(value: str) -> str:
    """Uppercases, strips punctuation, and collapses whitespace."""
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", str(value).upper())
    return re.sub(r"\s+", " ", cleaned).strip()


def _rank(value: str, table: Dict[str, int], axis: str) -> int:
    if not isinstance(value, str) or not value.strip():
        raise UnknownAlertLevel(f"{axis} level must be a non-empty string, got {value!r}")
    key = _normalise_key(value)
    if key in table:
        return table[key]
    raise UnknownAlertLevel(f"Unrecognized {axis} level: {value!r}")


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
    tarp_rank = _rank(tarp_level, _TARP_RANK, "TARP")
    forecast_rank = _rank(forecast_tier, _FORECAST_RANK, "forecast")

    discrepancy = tarp_rank != forecast_rank
    effective_level = _RANK_TO_LEVEL[max(tarp_rank, forecast_rank)]

    # INVARIANT: TARP-SOP-06 - a Red rule outcome can never be arbitrated down.
    if tarp_rank == 4 and effective_level != "RED":
        raise AssertionError(
            f"Inviolable Red floor violated: TARP={tarp_level!r} -> {effective_level!r}"
        )

    # Additional bump if sensors are degraded and level is still GREEN
    if "DEGRADED" in str(sensor_health_status).upper() and effective_level == "GREEN":
        effective_level = "YELLOW"
        discrepancy = True

    review_needed = discrepancy or (effective_level in ("ORANGE", "RED"))

    return {
        "tarp_level": tarp_level,
        "forecast_tier": forecast_tier,
        "effective_alert_level": effective_level,
        "discrepancy_detected": discrepancy,
        "review_needed": review_needed,
        "governing_source": "TARP" if tarp_rank >= forecast_rank else "FORECAST_ELEVATED",
        "rationale": (
            "Model and TARP agree." if not discrepancy
            else f"Discrepancy detected (TARP: {tarp_level}, Forecast: {forecast_tier}). "
                 f"Enforced higher caution: {effective_level}."
        ),
    }


def build_alert_package(
    zone_id: str,
    arbitrated_result: Dict[str, Any],
    geofenced_assets: List[Dict[str, Any]],
    sop_citation: str = "[TARP-SOP-01]",
    evacuation_radius_m: float = EVACUATION_RADIUS_M,
) -> Dict[str, Any]:
    """
    Constructs a structured, human-readable alert package for mine dispatch and audit log.
    Identifies vehicles/personnel inside the hazard geofence without issuing autonomous stops.

    An asset is threatened when it is in the same zone AND (when a distance is recorded)
    inside the evacuation radius. Assets with no distance, or an unparseable distance,
    are treated conservatively.
    """
    # Fail safe: a malformed arbitration result is escalated, never shown as GREEN.
    level = arbitrated_result.get("effective_alert_level", "RED")
    threatened_assets = []
    for asset in geofenced_assets:
        if asset.get("zone_id") != zone_id:
            continue
        distance = asset.get("distance_m")
        try:
            inside = distance is None or float(distance) <= evacuation_radius_m
        except (TypeError, ValueError):
            inside = True
        if inside:
            threatened_assets.append(asset)

    action_map = {
        "GREEN": "Normal operations. Maintain 15-min scan cadence.",
        "YELLOW": "Heightened caution. Notify pit supervisor; restrict non-essential foot traffic.",
        "ORANGE": "Operations suspended in zone. Park machinery outside 1.5x bench height.",
        "RED": (
            f"IMMEDIATE EVACUATION ADVISORY: Audible alarm, clear {int(evacuation_radius_m)}m "
            "danger polygon, barricade ramps."
        ),
    }

    return {
        "zone_id": zone_id,
        "alert_level": level,
        "advisory_directive": action_map.get(
            level, "Unrecognized alert level. Escalate to the geotechnical engineer; do not assume safe."
        ),
        "evacuation_radius_m": evacuation_radius_m if level == "RED" else None,
        "threatened_assets_count": len(threatened_assets),
        "threatened_asset_ids": [a.get("asset_id") for a in threatened_assets],
        "governing_source": arbitrated_result.get("governing_source"),
        "review_needed": arbitrated_result.get("review_needed", False),
        "sop_citation": sop_citation,
        "human_authorization_required": level in ("ORANGE", "RED"),
    }
