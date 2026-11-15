"""
test_safety_regressions.py - Regression tests for the 2026 safety audit findings.

Each test pins a defect that shipped in the audited build:
- fail-open alert parsing and NaN telemetry
- self-approving governance and mixed clock domains
- wrong RAG citation for level queries
- broken audit ledger and shared mutable blackboard state
- unbounded sensitivity attenuation, wrong evacuation radius
"""

import hashlib
import json

import pytest

from blackboard import FORECASTER_BASELINE_SENSITIVITY, MineBlackboard
from controller import RockWatchController
from ml.risk_tiers import apply_attenuation_floor, tier_from_probability
from tools.arbitration_tools import (
    EVACUATION_RADIUS_M,
    UnknownAlertLevel,
    arbitrate_decision,
    build_alert_package,
)
from tools.auth_tools import mint_credential, verify_credential
from tools.geotech_tools import evaluate_tarp
from tools.governance_tools import process_desensitization_proposal
from tools.rag_tools import LEVEL_PASSAGE, get_passage_for_level, retrieve_sop
from tools.sensor_tools import assess_sensor_health, ingest_sensor_frame


@pytest.fixture(autouse=True)
def auth_backend(monkeypatch):
    monkeypatch.setenv("ROCKWATCH_AUTH_SECRET", "test-secret-key")
    monkeypatch.setenv(
        "ROCKWATCH_GEOTECH_PASSPHRASE_SHA256",
        hashlib.sha256(b"test-passphrase").hexdigest(),
    )


@pytest.fixture
def temp_log(tmp_path):
    return str(tmp_path / "audit_log.jsonl")


# ---------------------------------------------------------------------------
# C1 - alert level parsing must never silently downgrade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["LEVEL 4 RED", " RED ", "level 4 (red)", "L4", "red"])
def test_red_aliases_are_parsed_as_red(value):
    result = arbitrate_decision(value, "LOW")
    assert result["effective_alert_level"] == "RED"
    assert result["governing_source"] == "TARP"


@pytest.mark.parametrize("value", ["PURPLE", "", "   ", None, 4])
def test_unknown_levels_raise_instead_of_defaulting_to_green(value):
    with pytest.raises(UnknownAlertLevel):
        arbitrate_decision(value, "LOW")


def test_model_high_still_elevates_to_red():
    assert arbitrate_decision("GREEN", "HIGH")["effective_alert_level"] == "RED"


def test_red_floor_is_asserted_not_assumed():
    decision = arbitrate_decision("RED", "LOW")
    assert decision["governing_source"] == "TARP"
    assert decision["effective_alert_level"] == "RED"


# ---------------------------------------------------------------------------
# C2 - non-finite telemetry must never evaluate as GREEN / HEALTHY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "abc", None, -1.0])
def test_evaluate_tarp_rejects_non_finite_velocity(bad):
    with pytest.raises(ValueError):
        evaluate_tarp("ZONE_A", bad)


def test_nan_radar_is_rejected_and_flags_degradation():
    clean = ingest_sensor_frame(
        "ZONE_A",
        {"radar_velocity_mm_hr": float("nan"), "rainfall_3h_mm": 1.0, "vibration_ppv_mms": 1.0},
    )
    assert clean["radar_velocity_mm_hr"] is None
    assert clean["effective_velocity_mm_hr"] is None
    assert "radar_velocity_mm_hr" in clean["rejected_fields"]

    health = assess_sensor_health(clean)
    assert health["radar_online"] is False
    assert health["health_status"] == "DEGRADED_INVALID_TELEMETRY"
    assert health["caution_elevation_required"] is True


def test_missing_weather_is_reported_missing_not_zero():
    clean = ingest_sensor_frame("ZONE_A", {"radar_velocity_mm_hr": 0.5, "vibration_ppv_mms": 1.0})
    assert clean["rainfall_3h_mm"] is None
    health = assess_sensor_health(clean)
    assert "WEATHER" in health["missing_modalities"]


def test_unusable_telemetry_never_produces_green(temp_log):
    bb = MineBlackboard(log_path=temp_log)
    ctrl = RockWatchController(blackboard=bb)
    result = ctrl.run_telemetry_cycle("ZONE_X", {"rainfall_3h_mm": 1.0, "vibration_ppv_mms": 1.0})
    assert result["effective_level"] != "GREEN"
    assert bb.read("tarp_status")["ZONE_X"]["data_quality_flag"] == "TELEMETRY_UNUSABLE"


# ---------------------------------------------------------------------------
# C3 / C4 - authorization and clock domain
# ---------------------------------------------------------------------------

def test_operator_cannot_approve():
    result = process_desensitization_proposal(
        "ZONE_A", "OP_BOB", "false alarm", "OPERATOR", current_time=1000.0
    )
    assert result["is_approved"] is False
    assert result["status"] == "DENIED_INSUFFICIENT_AUTHORITY"
    assert result["adjusted_sensitivity"] == 1.0


def test_engineer_role_without_credential_is_denied():
    result = process_desensitization_proposal(
        "ZONE_A", "OP_BOB", "thermal drift", "GEOTECH_ENGINEER",
        current_time=1000.0, approver_credential="",
    )
    assert result["is_approved"] is False
    assert result["status"] == "DENIED_INVALID_CREDENTIAL"


def test_engineer_with_valid_credential_is_approved():
    token = mint_credential("GEO-7714", "GEOTECH_ENGINEER", issued_at=1000.0)
    result = process_desensitization_proposal(
        "ZONE_A", "OP_BOB", "thermal drift", "GEOTECH_ENGINEER",
        duration_hours=4.0, current_time=1000.0, approver_credential=token,
    )
    assert result["is_approved"] is True
    assert result["adjusted_sensitivity"] == 0.85
    assert result["expires_at"] == 1000.0 + 4 * 3600.0


def test_credential_expiry_and_tampering_are_rejected():
    token = mint_credential("GEO-7714", "GEOTECH_ENGINEER", ttl_seconds=10, issued_at=1000.0)
    assert verify_credential(token, now=1005.0) is not None
    assert verify_credential(token, now=1011.0) is None

    parts = token.split("|")
    parts[0] = "GEO-0000"
    assert verify_credential("|".join(parts), now=1005.0) is None


def test_proposal_expires_on_the_shared_simulation_clock(temp_log):
    simulated = {"now": 10_000.0}

    def clock():
        return simulated["now"]

    bb = MineBlackboard(log_path=temp_log, time_provider=clock)
    ctrl = RockWatchController(blackboard=bb, time_provider=clock)

    ctrl.submit_desensitization_proposal(
        zone_id="ZONE_C",
        operator_id="OP_BOB",
        reason="thermal drift",
        approver_role="GEOTECH_ENGINEER",
        duration_hours=4.0,
        requested_sensitivity=0.80,
        approver_credential=mint_credential("GEO-7714", "GEOTECH_ENGINEER", issued_at=simulated["now"]),
    )
    assert bb.get_effective_forecaster_sensitivity("ZONE_C", clock()) == 0.80

    simulated["now"] += 5 * 3600.0
    assert bb.get_effective_forecaster_sensitivity("ZONE_C", clock()) == FORECASTER_BASELINE_SENSITIVITY


# ---------------------------------------------------------------------------
# C5 - grounded citation mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level,expected", list(LEVEL_PASSAGE.items()))
def test_level_resolves_to_signed_passage(level, expected):
    result = get_passage_for_level(level)
    assert result["found"] is True
    assert result["passage_id"] == expected


def test_bare_level_keyword_does_not_ground_a_citation():
    result = retrieve_sop("RED")
    assert result["found"] is False
    assert result["passage_id"] is None


def test_level_query_never_returns_an_incident_case():
    result = retrieve_sop("TARP Level 4 Red evacuation actions")
    assert result["found"] is True
    assert result["passage_id"] == "[TARP-SOP-04]"


def test_unknown_level_citation_fails_safe_to_the_red_passage():
    assert get_passage_for_level("PURPLE")["passage_id"] == "[TARP-SOP-04]"


# ---------------------------------------------------------------------------
# C6 / m - evacuation geometry and UI copy
# ---------------------------------------------------------------------------

def test_alert_package_uses_200m_polygon_and_filters_assets():
    decision = arbitrate_decision("RED", "LOW")
    assets = [
        {"asset_id": "TRUCK-1", "zone_id": "ZONE_A", "distance_m": 150.0},
        {"asset_id": "TRUCK-2", "zone_id": "ZONE_A", "distance_m": 400.0},
        {"asset_id": "TRUCK-3", "zone_id": "ZONE_B", "distance_m": 10.0},
    ]
    package = build_alert_package("ZONE_A", decision, assets)
    assert EVACUATION_RADIUS_M == 200.0
    assert "200m" in package["advisory_directive"]
    assert package["threatened_asset_ids"] == ["TRUCK-1"]


def test_alert_package_fails_safe_when_arbitration_result_is_malformed():
    package = build_alert_package("ZONE_A", {}, [])
    assert package["alert_level"] == "RED"
    assert "200m" in package["advisory_directive"]
    assert package["evacuation_radius_m"] == 200.0


def test_unparseable_asset_distance_is_treated_as_inside_the_fence():
    decision = arbitrate_decision("RED", "LOW")
    assets = [{"asset_id": "TRUCK-X", "zone_id": "ZONE_A", "distance_m": "unknown"}]
    package = build_alert_package("ZONE_A", decision, assets)
    assert package["threatened_asset_ids"] == ["TRUCK-X"]


def test_ui_copy_contains_no_conflicting_radius_or_threshold():
    with open("app.py", "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "500m" not in source
    assert "1.5 mm/h" not in source
    assert "submit_desensitization_request" not in source
    assert "retrieve_sop(sec_level)" not in source


# ---------------------------------------------------------------------------
# C8 / M5 - ledger integrity and partition isolation
# ---------------------------------------------------------------------------

def test_chain_is_valid_after_writes_and_detects_tampering(temp_log):
    bb = MineBlackboard(log_path=temp_log)
    bb.write("clean_telemetry", {"z1": {"v": 1.0}}, "TEST")
    bb.write("clean_telemetry", {"z2": {"v": 2.0}}, "TEST")

    report = MineBlackboard.verify_chain(temp_log)
    assert report["valid"] is True
    assert report["entries"] == 3  # genesis + 2 writes

    lines = []
    with open(temp_log, "r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    lines[1]["payload"]["malicious"] = True
    with open(temp_log, "w", encoding="utf-8") as handle:
        for entry in lines:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    report = MineBlackboard.verify_chain(temp_log)
    assert report["valid"] is False
    assert any(f["reason"] == "HASH_MISMATCH" for f in report["failures"])


def test_corrupt_ledger_is_archived_rather_than_appended_to(temp_log, tmp_path):
    bb = MineBlackboard(log_path=temp_log)
    bb.write("clean_telemetry", {"z1": {"v": 1.0}}, "TEST")

    with open(temp_log, "r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    lines[-1]["payload"]["tampered"] = True
    with open(temp_log, "w", encoding="utf-8") as handle:
        for entry in lines:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    resumed = MineBlackboard(log_path=temp_log)
    assert MineBlackboard.verify_chain(temp_log)["valid"] is True
    assert resumed.chain_status["valid"] is False
    assert list(tmp_path.glob("audit_log.jsonl.corrupt-*"))


def test_read_returns_an_isolated_copy(temp_log):
    bb = MineBlackboard(log_path=temp_log)
    bb.write("golden_pool", {"pending": {"count": 1}}, "TEST")

    snapshot = bb.read("golden_pool")
    snapshot["pending"]["count"] = 99
    assert bb.read("golden_pool")["pending"]["count"] == 1


# ---------------------------------------------------------------------------
# M3 - attenuation may remove at most one tier
# ---------------------------------------------------------------------------

def test_attenuation_floor_limits_tier_loss():
    assert tier_from_probability(0.9855) == "HIGH"
    # Raw HIGH at 0.5x would compute MODERATE; the floor permits one tier, not two.
    assert apply_attenuation_floor(0.9855, 0.4928) == "MODERATE"
    # A raw HIGH attenuated to LOW must be held at MODERATE.
    assert apply_attenuation_floor(0.90, 0.30) == "MODERATE"
    # No attenuation => unchanged tier.
    assert apply_attenuation_floor(0.50, 0.50) == "MODERATE"


# ---------------------------------------------------------------------------
# M1 - inverse velocity and acceleration are now live TARP criteria
# ---------------------------------------------------------------------------

def test_inverse_velocity_within_four_hours_triggers_red():
    result = evaluate_tarp("ZONE_A", 2.0, hours_to_failure=3.0)
    assert result["final_level"] == "RED"
    assert "INVERSE_VELOCITY_ZERO_WITHIN_4H" in result["red_criteria_matched"]


def test_inverse_velocity_within_twelve_hours_triggers_orange():
    result = evaluate_tarp("ZONE_A", 2.0, hours_to_failure=9.0)
    assert result["final_level"] == "ORANGE"


def test_sustained_acceleration_triggers_red(monkeypatch):
    import tools.geotech_tools as geotech_tools

    monkeypatch.setattr(geotech_tools, "ACCEL_RED_MM_HR2", 0.5)
    result = evaluate_tarp("ZONE_A", 2.0, acceleration_mm_hr2=1.2)
    assert result["final_level"] == "RED"


def test_acceleration_is_disabled_by_default_to_avoid_contradicting_sop02():
    # Row #103 of the shipped dataset is 3.92 mm/hr with +1.33 mm/hr^2 accel:
    # signed SOP-02 says Yellow. The over-broad "positive acceleration" RED
    # criterion must stay off until a noise floor is signed off.
    result = evaluate_tarp("ZONE_A", 3.92, acceleration_mm_hr2=1.33)
    assert result["final_level"] == "YELLOW"


def test_velocity_history_is_used_when_supplied():
    history = [0.5, 1.0, 3.0, 20.0]  # strongly accelerating
    result = evaluate_tarp("ZONE_A", history[-1], velocity_history=history)
    assert result["final_level"] == "RED"


# ---------------------------------------------------------------------------
# M1b - inverse velocity must use real elapsed time, not sample count
# ---------------------------------------------------------------------------

def test_same_instant_samples_cannot_fake_a_collapse_trend():
    from tools.geotech_tools import compute_inverse_velocity

    # Two readings two seconds apart (a UI rerun) are not a collapse trend.
    result = compute_inverse_velocity([0.604, 6.179], timestamps=[1000.0, 1002.0])
    assert result["estimated_hours_to_failure"] is None
    assert result["timing_basis"] == "TIMESTAMPS"


def test_real_hourly_acceleration_still_detects_collapse():
    from tools.geotech_tools import compute_inverse_velocity

    result = compute_inverse_velocity(
        [0.5, 1.0, 3.0, 20.0],
        timestamps=[0.0, 3600.0, 7200.0, 10800.0],
    )
    assert result["timing_basis"] == "TIMESTAMPS"
    assert result["estimated_hours_to_failure"] is not None
    assert result["estimated_hours_to_failure"] <= 12.0


def test_scenario_switch_does_not_inherit_previous_velocity_trend(temp_log):
    clock = {"now": 10_000.0}

    def now():
        return clock["now"]

    bb = MineBlackboard(log_path=temp_log, time_provider=now)
    ctrl = RockWatchController(blackboard=bb, time_provider=now)

    ctrl.run_telemetry_cycle("ZONE_A", {"radar_velocity_mm_hr": 0.604})
    # Loading a different dataset row is a new regime: reset the baseline.
    ctrl.reset_velocity_history("ZONE_A")
    clock["now"] += 5.0
    result = ctrl.run_telemetry_cycle("ZONE_A", {"radar_velocity_mm_hr": 6.179})

    tarp = result["trace"][2]["output"]
    assert tarp["inverse_velocity_analysis"]["estimated_hours_to_failure"] is None
    assert "INVERSE_VELOCITY_ZERO_WITHIN_4H" not in tarp["red_criteria_matched"]


def test_duplicate_samples_at_one_instant_replace_instead_of_append(temp_log):
    clock = {"now": 10_000.0}

    def now():
        return clock["now"]

    bb = MineBlackboard(log_path=temp_log, time_provider=now)
    ctrl = RockWatchController(blackboard=bb, time_provider=now)

    ctrl.run_telemetry_cycle("ZONE_A", {"radar_velocity_mm_hr": 0.6})
    ctrl.run_telemetry_cycle("ZONE_A", {"radar_velocity_mm_hr": 0.7})
    assert len(ctrl.velocity_history["ZONE_A"]) == 1
    assert ctrl.velocity_history["ZONE_A"][-1][1] == 0.7
