"""
tarp_agent.py - Deterministic TARP Engine & Inverse Velocity Physics Agent.

Role:
Executes deterministic geotechnical safety thresholds (docs/tarp_sop_v1.md)
and computes Fukuzono inverse-velocity acceleration trends.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.geotech_tools import evaluate_tarp, compute_inverse_velocity


class TARPAgent(BaseAgent):
    """Agent 3: Deterministic TARP Engine."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_TARP",
            role="Deterministic TARP Safety Engine & Kinematic Physics",
            plugin_name="TARPPlugin",
        )

    @kernel_function(name="evaluate_rules", description="Evaluates signed geotechnical TARP rules.")
    def evaluate_rules(
        self,
        zone_id: str,
        velocity_mm_hr: float,
        rainfall_3h_mm: float = 0.0,
        sensor_health_degraded: bool = False,
    ) -> Dict[str, Any]:
        return evaluate_tarp(zone_id, velocity_mm_hr, rainfall_3h_mm, sensor_health_degraded)

    @kernel_function(name="analyze_inverse_velocity", description="Computes Fukuzono inverse velocity acceleration.")
    def analyze_inverse_velocity(
        self,
        recent_velocities: List[float],
        time_step_hours: float = 1.0,
    ) -> Dict[str, Any]:
        return compute_inverse_velocity(recent_velocities, time_step_hours)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'clean_telemetry' and 'sensor_health', invokes tools,
        and writes 'tarp_status' to the shared blackboard.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        clean_telemetry = blackboard.read("clean_telemetry").get(zone_id, {})
        sensor_health = blackboard.read("sensor_health").get(zone_id, {})

        effective_velocity = clean_telemetry.get("effective_velocity_mm_hr", 0.0)
        rainfall = clean_telemetry.get("rainfall_3h_mm", 0.0)
        is_degraded = sensor_health.get("caution_elevation_required", False)

        velocity_history = context.get("velocity_history", [effective_velocity])

        # Tool 1: Evaluate deterministic TARP rules
        tarp_result = self.evaluate_rules(zone_id, effective_velocity, rainfall, is_degraded)

        # Tool 2: Compute Fukuzono inverse velocity
        inv_v_result = self.analyze_inverse_velocity(velocity_history)

        combined_tarp = {
            **tarp_result,
            "inverse_velocity_analysis": inv_v_result,
        }

        # Explicit Blackboard Write
        blackboard.write("tarp_status", {zone_id: combined_tarp}, self.agent_id)

        return combined_tarp
