"""
forecaster_agent.py - Calibrated Horizon Risk Forecaster Agent.

Role:
Computes zone-level rockfall probability over a 6-hour horizon,
modulates risk via approved sensitivity proposals, quantifies sensor uncertainty bounds,
and enriches forecasts with LLM physical correlation narratives via LLMRouter.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.forecast_tools import compute_zone_risk, create_provenance_record
from tools.llm_router import get_llm_router


class ForecasterAgent(BaseAgent):
    """Agent 4: Calibrated Horizon Risk Forecaster."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_FORECASTER",
            role="Calibrated Risk Forecaster & Uncertainty Assessor",
            plugin_name="ForecasterPlugin",
        )

    @kernel_function(name="estimate_risk", description="Estimates 6-hour rockfall probability and uncertainty interval.")
    def estimate_risk(
        self,
        zone_id: str,
        velocity_mm_hr: float,
        rainfall_3h_mm: float = 0.0,
        vibration_ppv: float = 0.0,
        sensitivity_multiplier: float = 1.0,
        sensor_uncertainty_penalty: float = 0.05,
    ) -> Dict[str, Any]:
        return compute_zone_risk(
            zone_id,
            velocity_mm_hr,
            rainfall_3h_mm,
            vibration_ppv,
            sensitivity_multiplier,
            sensor_uncertainty_penalty,
        )

    @kernel_function(name="log_provenance", description="Creates tamper-evident provenance audit record.")
    def log_provenance(self, zone_id: str, risk_payload: Dict[str, Any]) -> Dict[str, Any]:
        return create_provenance_record(zone_id, risk_payload)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'clean_telemetry' and 'sensor_health', queries fail-closed sensitivity,
        invokes tools, and writes 'forecast' to the shared blackboard.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        clean_telemetry = blackboard.read("clean_telemetry").get(zone_id, {})
        sensor_health = blackboard.read("sensor_health").get(zone_id, {})

        v = clean_telemetry.get("effective_velocity_mm_hr", 0.0)
        rain = clean_telemetry.get("rainfall_3h_mm", 0.0)
        vib = clean_telemetry.get("vibration_ppv_mms", 0.0)
        penalty = sensor_health.get("uncertainty_penalty", 0.05)

        # Fail-closed sensitivity lookup: reverts to baseline (1.0) on expiry or error
        current_time = context.get("timestamp")
        active_sensitivity = blackboard.get_effective_forecaster_sensitivity(zone_id, current_time)

        # Tool 1: Compute calibrated horizon risk
        risk_data = self.estimate_risk(zone_id, v, rain, vib, active_sensitivity, penalty)

        # Tool 2: Build provenance record
        provenance = self.log_provenance(zone_id, risk_data)

        # Multi-Agent LLM Call: Generate cross-sensor correlation narrative if requested
        p_risk = risk_data.get("risk_probability", 0.0)
        tier = risk_data.get("risk_tier", "LOW")
        narrative = f"6-hour calibrated rockfall probability is {p_risk:.1%} under tier {tier}."

        if context.get("enable_llm_narrative", False):
            router = get_llm_router()
            llm_narrative = router.generate_forecaster_narrative(
                zone_id=zone_id,
                p_rockfall=p_risk,
                risk_tier=tier,
                uncertainty=risk_data.get("uncertainty_margin", 0.05),
                telemetry=clean_telemetry,
            )
            if llm_narrative:
                narrative = llm_narrative

        combined_forecast = {
            **risk_data,
            "provenance": provenance,
            "narrative": narrative,
        }

        # Explicit Blackboard Write
        blackboard.write("forecast", {zone_id: combined_forecast}, self.agent_id)

        return combined_forecast
