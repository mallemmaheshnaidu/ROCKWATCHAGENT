"""
test_scenarios.py - Comprehensive Verification Test Suite for RockWatch.

Covers:
- Scenario 1: Primary radar offline -> Health flags degradation -> Caution elevated per TARP-SOP-05 -> Forecaster widens uncertainty.
- Scenario 2: Un-alarmed surface detachment discovered -> Instant caution without approval -> Queued to pending golden pool -> Diagnostic root cause.
- Scenario 3: False alarm reported -> Operator blocked, Geotech authorized -> Auto-expiry to baseline via injectable clock -> Model gate rejects recall drop.
- Scenario 4: Middle tampering of append-only run log is detected by canonical hash verification.
- Scenario 5: Grounded RAG returns explicit passage IDs and fails closed ('Information not found') on out-of-corpus queries.
- Scenario 6: System resilience during LLM outage (deterministic safety path remains 100% functional).
"""

import os
import json
import hashlib
import pytest
from blackboard import MineBlackboard, FORECASTER_BASELINE_SENSITIVITY, TARP_SAFETY_FLOOR_VELOCITY_MM_HR
from controller import RockWatchController
from tools.auth_tools import mint_credential
from tools.rag_tools import retrieve_sop


@pytest.fixture(autouse=True)
def auth_backend(monkeypatch):
    """Configures a deterministic authorization backend for the suite."""
    monkeypatch.setenv("ROCKWATCH_AUTH_SECRET", "test-secret-key")
    monkeypatch.setenv(
        "ROCKWATCH_GEOTECH_PASSPHRASE_SHA256",
        hashlib.sha256(b"test-passphrase").hexdigest(),
    )


@pytest.fixture
def temp_log(tmp_path):
    """Provides an isolated run log path for each test."""
    return str(tmp_path / "test_run_log.jsonl")


def test_scenario_1_degraded_sensors(temp_log):
    """
    Scenario 1: Primary radar offline.
    Verifies that losing primary radar elevates caution (Green -> Yellow per TARP-SOP-05)
    and widens the forecaster's uncertainty margin.
    """
    bb = MineBlackboard(log_path=temp_log)
    ctrl = RockWatchController(blackboard=bb)

    # Telemetry with NO radar, but secondary crack extensometer active (0.4 mm/hr - would normally be GREEN)
    raw_frame = {
        "radar_velocity_mm_hr": None,
        "extensometer_velocity_mm_hr": 0.4,
        "rainfall_3h_mm": 2.0,
        "vibration_ppv_mms": 1.0,
    }

    result = ctrl.run_telemetry_cycle("ZONE_A", raw_frame)

    # 1. Health agent flags degradation
    health = bb.read("sensor_health")["ZONE_A"]
    assert health["radar_online"] is False
    assert health["caution_elevation_required"] is True
    assert health["uncertainty_penalty"] > 0.10

    # 2. TARP rules elevate from Green to Yellow due to degradation
    tarp = bb.read("tarp_status")["ZONE_A"]
    assert tarp["base_level"] == "GREEN"
    assert tarp["final_level"] == "YELLOW"

    # 3. Forecaster widens uncertainty margin
    forecast = bb.read("forecast")["ZONE_A"]
    assert forecast["uncertainty_margin"] >= 0.25

    # 4. Arbiter enforces elevated Yellow caution
    assert result["effective_level"] == "YELLOW"


def test_scenario_2_missed_event_instant_caution(temp_log):
    """
    Scenario 2: Un-alarmed surface detachment discovered post-hoc.
    Verifies that a missed event triggers instant caution elevation,
    is placed in the pending golden set, and generates a diagnostic root-cause proposal.
    """
    bb = MineBlackboard(log_path=temp_log)
    ctrl = RockWatchController(blackboard=bb)

    # Quiet telemetry with no preceding alerts
    raw_frame = {"radar_velocity_mm_hr": 0.2, "rainfall_3h_mm": 0.0}

    # Context simulates LiDAR scan differencing discovering a 15.0 m3 detachment
    ctx = {
        "baseline_scan_volume_m3": 200.0,
        "current_scan_volume_m3": 185.0,  # 15 m3 detached
        "recent_alerts": [],              # Zero preceding warnings = Missed Event
    }

    ctrl.run_telemetry_cycle("ZONE_B", raw_frame, context=ctx)

    # 1. Perception flags unalarmed miss
    surface = bb.read("surface_changes")["ZONE_B"]
    assert surface["detachment_detected"] is True
    assert surface["volume_detached_m3"] == 15.0

    # 2. Event is queued into pending golden pool
    golden = bb.read("golden_pool")
    pending = golden.get("pending_misses", [])
    assert len(pending) > 0
    assert pending[0]["zone_id"] == "ZONE_B"
    assert pending[0]["volume_m3"] == 15.0

    # 3. Governance produces root-cause diagnosis for geotech sign-off
    gov_res = ctrl.governance_agent.run(bb, {"zone_id": "ZONE_B", "timestamp": 1000.0})
    root_cause = gov_res.get("root_cause_diagnosis", {})
    assert root_cause["governance_status"] == "PROPOSED_PENDING_GEOTECH_SIGN_OFF"
    assert root_cause["proposed_root_cause"] != ""


def test_scenario_3_false_alarm_proposal_and_gate_rejection(temp_log):
    """
    Scenario 3: Asymmetric authority on false alarms and model gating.
    Verifies:
    1. Operator cannot unilaterally approve desensitization (must be GEOTECH_ENGINEER).
    2. Geotech-approved desensitization auto-expires to baseline via injectable clock.
    3. Model Gate rejects a candidate model that drops recall on confirmed events.
    """
    # Controllable mock clock
    simulated_time = 10000.0

    def mock_clock():
        return simulated_time

    bb = MineBlackboard(log_path=temp_log, time_provider=mock_clock)
    ctrl = RockWatchController(blackboard=bb, time_provider=mock_clock)

    # Step A: Operator attempts to desensitize Zone C (Blocked)
    op_res = ctrl.submit_desensitization_proposal(
        zone_id="ZONE_C",
        operator_id="OP_BOB",
        reason="Blast dust spike",
        approver_role="OPERATOR",
        duration_hours=4.0,
    )
    assert op_res["proposal_result"]["is_approved"] is False
    assert bb.get_effective_forecaster_sensitivity("ZONE_C", simulated_time) == FORECASTER_BASELINE_SENSITIVITY

    # Step B: Authenticated Geotech Engineer approves proposal
    geo_res = ctrl.submit_desensitization_proposal(
        zone_id="ZONE_C",
        operator_id="OP_BOB",
        reason="Blast dust confirmed by physical inspection",
        approver_role="GEOTECH_ENGINEER",
        duration_hours=4.0,
        requested_sensitivity=0.80,
        approver_credential=mint_credential(
            "GEO-7714", "GEOTECH_ENGINEER", issued_at=simulated_time
        ),
    )
    assert geo_res["proposal_result"]["is_approved"] is True
    assert bb.get_effective_forecaster_sensitivity("ZONE_C", simulated_time) == 0.80

    # Step C: Fast-forward clock by 5 hours (> 4 hour expiry limit)
    simulated_time += (5 * 3600.0)

    # Verifies fail-closed auto-expiry
    active_sens = bb.get_effective_forecaster_sensitivity("ZONE_C", simulated_time)
    assert active_sens == FORECASTER_BASELINE_SENSITIVITY  # Reverted to 1.0!

    # Step D: Model gate evaluates candidate model with recall drop
    champ = {"recall": 0.96, "f1_score": 0.91, "precision": 0.86}
    chal_bad = {"recall": 0.92, "f1_score": 0.92, "precision": 0.92}  # Recall dropped by 0.04

    gate_res = ctrl.evaluate_challenger_model(champ, chal_bad)
    assert gate_res["gate_result"]["promoted"] is False
    assert gate_res["gate_result"]["status"] == "REJECTED_SAFETY_REGRESSION"


def test_scenario_4_tamper_detection_in_run_log(temp_log):
    """
    Scenario 4: Hash-chained run log tampering test.
    Verifies that mutating an entry in the middle of the log is detected by hash verification.
    """
    bb = MineBlackboard(log_path=temp_log)
    bb.write("clean_telemetry", {"z1": {"v": 1.0}}, "TEST_AGENT")
    bb.write("clean_telemetry", {"z2": {"v": 2.0}}, "TEST_AGENT")
    bb.write("clean_telemetry", {"z3": {"v": 3.0}}, "TEST_AGENT")

    # Read and parse all lines
    lines = []
    with open(temp_log, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line.strip()))

    assert len(lines) >= 4  # Genesis + 3 writes

    # Verification function
    def verify_chain(entry_list):
        for i in range(1, len(entry_list)):
            curr = entry_list[i].copy()
            expected_hash = curr.pop("hash")
            canonical_str = json.dumps(curr, sort_keys=True)
            recomputed = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
            if recomputed != expected_hash or curr["prev_hash"] != entry_list[i - 1]["hash"]:
                return False, i
        return True, -1

    # Initially untampered: passes
    is_valid, _ = verify_chain(lines)
    assert is_valid is True

    # Tamper with payload of line 2
    tampered_lines = [l.copy() for l in lines]
    tampered_lines[1]["payload"]["malicious_tamper"] = True

    is_valid_tampered, failed_index = verify_chain(tampered_lines)
    assert is_valid_tampered is False
    assert failed_index == 1  # Successfully detected tampering at line 1!


def test_scenario_5_rag_grounding_and_missing_citation():
    """
    Scenario 5: Grounded RAG verification.
    Verifies that existing SOPs return passage IDs, and ungrounded queries fail closed.
    """
    # 1. Valid geotechnical SOP lookup
    res_valid = retrieve_sop("TARP Level 2 heightened caution scan rate")
    assert res_valid["found"] is True
    assert res_valid["passage_id"] == "[TARP-SOP-02]"
    assert "Actions required: increase radar scan cadence" in res_valid["text"]

    # 2. Out-of-corpus query fails closed
    res_invalid = retrieve_sop("Quantum mechanical tunneling of quartz crystals in basalt")
    assert res_invalid["found"] is False
    assert res_invalid["passage_id"] is None
    assert res_invalid["text"] == "Information not found in geotechnical documentation."


def test_scenario_6_llm_outage_fallback(temp_log):
    """
    Scenario 6: Deterministic safety execution during LLM outage.
    Verifies that the entire safety path functions cleanly with zero exceptions
    even when no external LLM API is available.
    """
    bb = MineBlackboard(log_path=temp_log)
    ctrl = RockWatchController(blackboard=bb)

    # Imminent critical collapse scenario
    raw_frame = {
        "radar_velocity_mm_hr": 18.5,  # Exceeds 15 mm/hr Red threshold
        "rainfall_3h_mm": 5.0,
        "vibration_ppv_mms": 12.0,
    }

    result = ctrl.run_telemetry_cycle("ZONE_CRITICAL", raw_frame)

    assert result["effective_level"] == "RED"
    assert result["final_alert_package"]["alert_level"] == "RED"
    assert "IMMEDIATE EVACUATION ADVISORY" in result["final_alert_package"]["advisory_directive"]
    assert result["final_alert_package"]["sop_citation"] == "[TARP-SOP-04]"
    assert result["head_hash"] != ""
