"""
risk_tiers.py - Shared risk-tier vocabulary and attenuation floor.

Kept dependency-free so both the ML fusion engine and the deterministic
fallback path can share one definition of what a tier means and how far an
approved desensitization is allowed to lower it.
"""

from __future__ import annotations

MIN_SENSITIVITY_MULTIPLIER: float = 0.50
HIGH_TIER_CUT: float = 0.75
MODERATE_TIER_CUT: float = 0.40

TIER_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
RANK_TIER = {rank: tier for tier, rank in TIER_RANK.items()}


def tier_from_probability(probability: float) -> str:
    if probability >= HIGH_TIER_CUT:
        return "HIGH"
    if probability >= MODERATE_TIER_CUT:
        return "MODERATE"
    return "LOW"


def clamp_sensitivity(multiplier: float) -> float:
    """Bounds an approved sensitivity multiplier to the signed 0.50-1.00 range."""
    return max(MIN_SENSITIVITY_MULTIPLIER, min(1.0, float(multiplier)))


def apply_attenuation_floor(raw_probability: float, adjusted_probability: float) -> str:
    """
    Returns the tier for an attenuated probability, guaranteeing that approved
    desensitization can remove at most one tier of severity.
    """
    raw_tier = tier_from_probability(raw_probability)
    adjusted_tier = tier_from_probability(adjusted_probability)
    if TIER_RANK[adjusted_tier] < TIER_RANK[raw_tier] - 1:
        return RANK_TIER[TIER_RANK[raw_tier] - 1]
    return adjusted_tier
