"""
arbiter_agent.py - Decision Arbiter & Alert Package Responder Agent.

Role:
Arbitrates between deterministic TARP rules and ML risk forecast,
enforces asymmetric caution and discrepancy flagging, and builds dispatch alert packages.
Enriched with LLMRouter for plain-language arbitration justifications.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.arbitration_tools import arbitrate_decision, build_alert_package
from tools.llm_router import get_llm_router


class ArbiterAgent(BaseAgent):
    """Agent 6: Decision Arbiter & Alert Responder."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_ARBITER",
            role="Decision Arbiter & Response Dispatcher",
            plugin_name="ArbiterPlugin",
        )

    @kernel_function(name="arbitrate", description="Resolves TARP vs Model divergence and sets effective level.")
    def arbitrate(
        self,
        tarp_level: str,
        forecast_tier: str,
        sensor_health_status: str = "HEALTHY",
    ) -> Dict[str, Any]:
        return arbitrate_decision(tarp_level, forecast_tier, sensor_health_status)

    @kernel_function(name="build_package", description="Builds structured dispatch package with geofenced assets.")
    def build_package(
        self,
        zone_id: str,
        arbitrated_result: Dict[str, Any],
        geofenced_assets: List[Dict[str, Any]],
        sop_citation: str = "[TARP-SOP-01]",
    ) -> Dict[str, Any]:
        return build_alert_package(zone_id, arbitrated_result, geofenced_assets, sop_citation)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'tarp_status', 'forecast', 'sensor_health', and 'advisor_notes',
        invokes tools, and writes 'arbitration' to the shared blackboard.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        tarp_status = blackboard.read("tarp_status").get(zone_id, {})
        forecast = blackboard.read("forecast").get(zone_id, {})
        sensor_health = blackboard.read("sensor_health").get(zone_id, {})
        advisor_notes = blackboard.read("advisor_notes").get(zone_id, {})

        tarp_level = tarp_status.get("final_level", "GREEN")
        forecast_tier = forecast.get("risk_tier", "LOW")
        health_status = sensor_health.get("health_status", "HEALTHY")
        citation = advisor_notes.get("primary_citation", "[TARP-SOP-01]")
        geofenced_assets = context.get("geofenced_assets", [])

        # Tool 1: Arbitrate decision (Asymmetric caution + discrepancy detection)
        decision = self.arbitrate(tarp_level, forecast_tier, health_status)

        # Multi-Agent LLM Call: Explain asymmetric elevation if divergence detected
        if decision.get("discrepancy_detected") and context.get("enable_llm_arbiter", False):
            router = get_llm_router()
            llm_just = router.generate_arbiter_explanation(
                zone_id=zone_id,
                tarp_level=tarp_level,
                forecast_tier=forecast_tier,
                effective_level=decision.get("effective_alert_level", "GREEN"),
                discrepancy=decision.get("discrepancy_detected", False),
            )
            if llm_just:
                decision["llm_justification"] = llm_just

        # Tool 2: Build alert dispatch package
        alert_pkg = self.build_package(zone_id, decision, geofenced_assets, citation)

        combined_result = {
            "decision": decision,
            "alert_package": alert_pkg,
        }

        # Explicit Blackboard Write
        blackboard.write("arbitration", {zone_id: combined_result}, self.agent_id)

        return combined_result
