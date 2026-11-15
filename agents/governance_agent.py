"""
governance_agent.py - Human Authority Governance & Root Cause Analysis Agent.

Role:
Manages desensitization proposals under the Asymmetric Safety Principle (requires GEOTECH_ENGINEER),
enforces time-limited auto-expiry, and analyzes root causes of missed events for human sign-off.
Powered by LLMRouter for diagnostic failure analysis.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard, DesensitizationProposal
from sk_compat import kernel_function
from tools.governance_tools import process_desensitization_proposal, propose_root_cause
from tools.llm_router import get_llm_router


class GovernanceAgent(BaseAgent):
    """Agent 7: Human Authority Governance & Root Cause Analysis."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_GOVERNANCE",
            role="Human Authority Governance & Root Cause Analysis",
            plugin_name="GovernancePlugin",
        )

    @kernel_function(name="review_proposal", description="Evaluates sensitivity proposal against geotech authorization.")
    def review_proposal(
        self,
        zone_id: str,
        operator_id: str,
        reason: str,
        approver_role: str,
        duration_hours: float = 8.0,
        requested_sensitivity: float = 0.85,
        current_time: float = 0.0,
    ) -> Dict[str, Any]:
        return process_desensitization_proposal(
            zone_id,
            operator_id,
            reason,
            approver_role,
            duration_hours,
            requested_sensitivity,
            current_time,
        )

    @kernel_function(name="diagnose_miss", description="Proposes root cause for an unalarmed detachment.")
    def diagnose_miss(
        self,
        zone_id: str,
        telemetry_snapshot: Dict[str, Any],
        sensor_health: Dict[str, Any],
    ) -> Dict[str, Any]:
        return propose_root_cause(zone_id, telemetry_snapshot, sensor_health)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads proposal or diagnostic triggers, processes them,
        and writes to 'proposals' and 'golden_pool'.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        current_time = context.get("timestamp", 0.0)

        results = {}

        # 1. Handle desensitization proposal request if present in context
        if prop_req := context.get("desensitization_request"):
            proposal_res = self.review_proposal(
                zone_id=zone_id,
                operator_id=prop_req.get("operator_id", "OPERATOR_UNKNOWN"),
                reason=prop_req.get("reason", "False alarm report"),
                approver_role=prop_req.get("approver_role", "OPERATOR"),
                duration_hours=prop_req.get("duration_hours", 8.0),
                requested_sensitivity=prop_req.get("requested_sensitivity", 0.85),
                current_time=current_time,
            )

            # Store as DesensitizationProposal in blackboard
            prop_obj = DesensitizationProposal(
                proposal_id=proposal_res["proposal_id"],
                zone_id=zone_id,
                reason=proposal_res["reason"],
                requested_by=proposal_res["operator_id"],
                created_at=proposal_res["created_at"],
                expires_at=proposal_res["expires_at"],
                approved_by=prop_req.get("approver_role") if proposal_res["is_approved"] else None,
                is_approved=proposal_res["is_approved"],
                adjusted_sensitivity=proposal_res["adjusted_sensitivity"],
            )

            # Explicit Blackboard Write
            blackboard.write("proposals", {zone_id: prop_obj.to_dict()}, self.agent_id)
            results["proposal_result"] = proposal_res

        # 2. Handle root-cause diagnosis if unalarmed miss was detected
        surface_changes = blackboard.read("surface_changes").get(zone_id, {})
        if surface_changes.get("detachment_detected"):
            clean_telem = blackboard.read("clean_telemetry").get(zone_id, {})
            health = blackboard.read("sensor_health").get(zone_id, {})
            root_cause = self.diagnose_miss(zone_id, clean_telem, health)

            # Multi-Agent LLM Call: Generate failure analysis if requested
            if context.get("enable_llm_governance", False):
                router = get_llm_router()
                llm_diag = router.generate_governance_root_cause(
                    zone_id=zone_id,
                    detached_volume_m3=surface_changes.get("volume_detached_m3", 10.0),
                    telemetry=clean_telem,
                    sensor_health=health,
                )
                if llm_diag:
                    root_cause["diagnostic_analysis"] = llm_diag

            # Write root cause to golden_pool for geotech sign-off
            golden_pool = blackboard.read("golden_pool")
            golden_pool["last_root_cause_diagnosis"] = root_cause
            blackboard.write("golden_pool", golden_pool, self.agent_id)
            results["root_cause_diagnosis"] = root_cause

        return results
