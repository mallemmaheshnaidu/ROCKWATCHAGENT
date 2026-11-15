"""
health_agent.py - Sensor Ingestion & Health Auditing Agent.

Role:
Ingests raw telemetry frames, handles missing sensor modalities gracefully,
and audits sensor health to enforce TARP-SOP-05 caution elevation.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.sensor_tools import ingest_sensor_frame, assess_sensor_health


class HealthAgent(BaseAgent):
    """Agent 1: Sensor Ingestion & Health Auditing."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_HEALTH",
            role="Sensor Ingestion & Health Auditing",
            plugin_name="HealthPlugin",
        )

    @kernel_function(name="ingest_telemetry", description="Ingests and standardizes raw sensor telemetry.")
    def ingest_telemetry(self, zone_id: str, raw_frame: Dict[str, Any]) -> Dict[str, Any]:
        return ingest_sensor_frame(zone_id, raw_frame)

    @kernel_function(name="audit_health", description="Assesses sensor health and flags coverage degradation.")
    def audit_health(self, telemetry_dict: Dict[str, Any]) -> Dict[str, Any]:
        return assess_sensor_health(telemetry_dict)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'raw_telemetry', invokes tools, and writes 'clean_telemetry' and 'sensor_health'.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        raw_telemetry = blackboard.read("raw_telemetry").get(zone_id, context.get("raw_frame", {}))

        # Tool 1: Clean and standardize telemetry
        clean_data = self.ingest_telemetry(zone_id, raw_telemetry)

        # Tool 2: Assess health and degradation
        health_data = self.audit_health(clean_data)

        # Explicit Blackboard Writes
        blackboard.write("clean_telemetry", {zone_id: clean_data}, self.agent_id)
        blackboard.write("sensor_health", {zone_id: health_data}, self.agent_id)

        return {"clean_telemetry": clean_data, "sensor_health": health_data}
