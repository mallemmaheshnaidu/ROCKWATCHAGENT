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
from tools.geotech_tools import compute_inverse_velocity, evaluate_tarp


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
        acceleration_mm_hr2: float = None,
        hours_to_failure: float = None,
        velocity_history: List[float] = None,
        velocity_timestamps: List[float] = None,
    ) -> Dict[str, Any]:
        return evaluate_tarp(
            zone_id,
            velocity_mm_hr,
            rainfall_3h_mm,
            sensor_health_degraded,
            acceleration_mm_hr2=acceleration_mm_hr2,
            hours_to_failure=hours_to_failure,
            velocity_history=velocity_history,
            velocity_timestamps=velocity_timestamps,
        )

    @kernel_function(name="analyze_inverse_velocity", description="Computes Fukuzono inverse velocity acceleration.")
    def analyze_inverse_velocity(
        self,
        recent_velocities: List[float],
        time_step_hours: float = 1.0,
        timestamps: List[float] = None,
    ) -> Dict[str, Any]:
        return compute_inverse_velocity(recent_velocities, time_step_hours, timestamps)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'clean_telemetry' and 'sensor_health', invokes tools,
        and writes 'tarp_status' to the blackboard.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        clean_telemetry = blackboard.read("clean_telemetry").get(zone_id, {})
        sensor_health = blackboard.read("sensor_health").get(zone_id, {})

        effective_velocity = clean_telemetry.get("effective_velocity_mm_hr")
        rainfall = clean_telemetry.get("rainfall_3h_mm")
        acceleration = clean_telemetry.get("acceleration_mm_hr2")
        is_degraded = bool(sensor_health.get("caution_elevation_required", False))

        # Unusable kinematics must never be evaluated as a stable slope.
        telemetry_unusable = effective_velocity is None
        if telemetry_unusable:
            effective_velocity = 0.0
            is_degraded = True

        velocity_history = context.get("velocity_history") or [effective_velocity]
        velocity_timestamps = context.get("velocity_timestamps")

        # Tool 1: Evaluate deterministic TARP rules
        tarp_result = self.evaluate_rules(
            zone_id,
            effective_velocity,
            0.0 if rainfall is None else rainfall,
            is_degraded,
            acceleration_mm_hr2=acceleration,
            velocity_history=velocity_history,
            velocity_timestamps=velocity_timestamps,
        )

        # Tool 2: Compute Fukuzono inverse velocity
        inv_v_result = self.analyze_inverse_velocity(
            velocity_history, timestamps=velocity_timestamps
        )

        combined_tarp = {
            **tarp_result,
            "inverse_velocity_analysis": inv_v_result,
            "data_quality_flag": "TELEMETRY_UNUSABLE" if telemetry_unusable else "OK",
        }

        # Explicit Blackboard Write
        blackboard.write("tarp_status", {zone_id: combined_tarp}, self.agent_id)

        return combined_tarp
