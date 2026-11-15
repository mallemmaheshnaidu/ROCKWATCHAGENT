"""
perception_agent.py - 3-D Surface Change & Unalarmed Miss Discovery Agent.

Role:
Performs volumetric 3-D scan differencing, detects physical detachments,
and identifies unalarmed events (misses) to eliminate label bias.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.perception_tools import detect_surface_change, find_unalarmed_changes


class PerceptionAgent(BaseAgent):
    """Agent 2: Surface Change & Missed Event Perception."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_PERCEPTION",
            role="Surface Change Detection & Unalarmed Event Discovery",
            plugin_name="PerceptionPlugin",
        )

    @kernel_function(name="detect_change", description="Computes volumetric scan differences from 3-D scans.")
    def detect_change(
        self,
        zone_id: str,
        baseline_scan_volume_m3: float,
        current_scan_volume_m3: float,
    ) -> Dict[str, Any]:
        return detect_surface_change(zone_id, baseline_scan_volume_m3, current_scan_volume_m3)

    @kernel_function(name="check_misses", description="Compares detachments against alert history to catch missed events.")
    def check_misses(
        self,
        zone_id: str,
        detected_change: Dict[str, Any],
        recent_alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return find_unalarmed_changes(zone_id, detected_change, recent_alerts)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads scan data from context, reads 'clean_telemetry' and 'arbitration' from blackboard,
        invokes tools, and writes 'surface_changes' and 'golden_pool'.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        baseline_vol = context.get("baseline_scan_volume_m3", 100.0)
        current_vol = context.get("current_scan_volume_m3", 100.0)
        recent_alerts = context.get("recent_alerts", [])

        # Tool 1: Compute volumetric surface change
        change_result = self.detect_change(zone_id, baseline_vol, current_vol)

        # Tool 2: Cross-reference with alert history to catch missed events
        miss_check = self.check_misses(zone_id, change_result, recent_alerts)

        # Explicit Blackboard Writes
        blackboard.write("surface_changes", {zone_id: change_result}, self.agent_id)

        # If a missed event is discovered, queue into golden_pool for batch retraining review
        if miss_check.get("is_unalarmed_miss", False):
            golden_pool = blackboard.read("golden_pool")
            pending_misses = golden_pool.get("pending_misses", [])
            pending_misses.append({
                "zone_id": zone_id,
                "volume_m3": change_result.get("volume_detached_m3"),
                "detected_at": context.get("timestamp", 0.0),
                "status": "PENDING_GEOTECH_VERIFICATION",
            })
            blackboard.write("golden_pool", {"pending_misses": pending_misses}, self.agent_id)

        return {"surface_change": change_result, "miss_check": miss_check}
