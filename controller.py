"""
controller.py - Deterministic Event Router & Semantic Kernel Orchestrator.

Enforces:
- Deterministic routing of domain events through the 8 specialized agents.
- Hosts the Microsoft Semantic Kernel and registers all agents as KernelPlugins.
- Bridges data across the shared MineBlackboard.
- Supports clock injection for deterministic simulation & testing.
- Safe error recovery: LLM outages or tool faults never compromise the safety path.
"""

from __future__ import annotations
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from blackboard import MineBlackboard
from sk_compat import Kernel
from agents.health_agent import HealthAgent
from agents.perception_agent import PerceptionAgent
from agents.tarp_agent import TARPAgent
from agents.forecaster_agent import ForecasterAgent
from agents.advisor_agent import AdvisorAgent
from agents.arbiter_agent import ArbiterAgent
from agents.governance_agent import GovernanceAgent
from agents.evaluator_agent import EvaluatorAgent


class RockWatchController:
    """
    Coordinates multi-agent execution across the shared MineBlackboard.
    Decides agent execution sequences deterministically according to event type.
    """

    def __init__(
        self,
        blackboard: Optional[MineBlackboard] = None,
        docs_dir: str = "docs",
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.time_provider = time_provider or time.time
        self.blackboard = blackboard or MineBlackboard(time_provider=self.time_provider)

        # Bounded per-zone velocity history feeds the Fukuzono inverse-velocity screen.
        # Samples carry their epoch timestamp so the fit uses real elapsed time.
        self.velocity_history: Dict[str, Deque[tuple]] = {}
        self.velocity_history_maxlen = 12
        self.velocity_min_sample_gap_s = 15 * 60.0
        self.velocity_reset_gap_s = 12 * 3600.0

        # Initialize Microsoft Semantic Kernel host
        self.kernel = Kernel()

        # Instantiate the 8 domain agents
        self.health_agent = HealthAgent()
        self.perception_agent = PerceptionAgent()
        self.tarp_agent = TARPAgent()
        self.forecaster_agent = ForecasterAgent()
        self.advisor_agent = AdvisorAgent(docs_dir=docs_dir)
        self.arbiter_agent = ArbiterAgent()
        self.governance_agent = GovernanceAgent()
        self.evaluator_agent = EvaluatorAgent()

        # Register all 8 agents as Semantic Kernel Plugins
        self.kernel.add_plugin(self.health_agent.to_plugin(), "HealthPlugin")
        self.kernel.add_plugin(self.perception_agent.to_plugin(), "PerceptionPlugin")
        self.kernel.add_plugin(self.tarp_agent.to_plugin(), "TARPPlugin")
        self.kernel.add_plugin(self.forecaster_agent.to_plugin(), "ForecasterPlugin")
        self.kernel.add_plugin(self.advisor_agent.to_plugin(), "AdvisorPlugin")
        self.kernel.add_plugin(self.arbiter_agent.to_plugin(), "ArbiterPlugin")
        self.kernel.add_plugin(self.governance_agent.to_plugin(), "GovernancePlugin")
        self.kernel.add_plugin(self.evaluator_agent.to_plugin(), "EvaluatorPlugin")

    def run_telemetry_cycle(self, zone_id: str, raw_frame: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the primary monitoring cycle across all operational agents:
        Health -> Perception -> TARP -> Forecaster -> Advisor -> Arbiter
        """
        ctx = context.copy() if context else {}
        ctx["zone_id"] = zone_id
        ctx["raw_frame"] = raw_frame
        if "timestamp" not in ctx:
            ctx["timestamp"] = self.time_provider()

        # Write raw telemetry into blackboard partition
        self.blackboard.write("raw_telemetry", {zone_id: raw_frame}, "CONTROLLER")

        trace: List[Dict[str, Any]] = []

        # 1. Health & Ingestion
        res_health = self.health_agent.run(self.blackboard, ctx)
        trace.append({"step": 1, "agent": self.health_agent.agent_id, "output": res_health})

        # Maintain the kinematic history used by the inverse-velocity screen.
        effective_v = (self.blackboard.read("clean_telemetry").get(zone_id) or {}).get(
            "effective_velocity_mm_hr"
        )
        sample_time = float(ctx.get("timestamp", self.time_provider()))
        if effective_v is not None:
            history = self.velocity_history.setdefault(
                zone_id, deque(maxlen=self.velocity_history_maxlen)
            )
            if not history:
                history.append((sample_time, float(effective_v)))
            else:
                last_time, _ = history[-1]
                gap = sample_time - last_time
                if gap <= 0:
                    # Same instant (UI rerun): replace the newest reading rather
                    # than inventing a time step between duplicate samples.
                    history[-1] = (sample_time, float(effective_v))
                elif gap > self.velocity_reset_gap_s:
                    # A long outage invalidates the trend baseline.
                    history.clear()
                    history.append((sample_time, float(effective_v)))
                elif gap >= self.velocity_min_sample_gap_s:
                    history.append((sample_time, float(effective_v)))
                # Sub-interval samples are ignored: the trend needs real elapsed time.
        ctx["velocity_history"] = [value for _, value in self.velocity_history.get(zone_id, [])]
        ctx["velocity_timestamps"] = [stamp for stamp, _ in self.velocity_history.get(zone_id, [])]

        # 2. Perception (Surface Change & Miss Detection)
        res_perception = self.perception_agent.run(self.blackboard, ctx)
        trace.append({"step": 2, "agent": self.perception_agent.agent_id, "output": res_perception})

        # 3. TARP Engine (Deterministic Rules)
        res_tarp = self.tarp_agent.run(self.blackboard, ctx)
        trace.append({"step": 3, "agent": self.tarp_agent.agent_id, "output": res_tarp})

        # 4. Risk Forecaster (Calibrated Horizon Probability)
        res_forecaster = self.forecaster_agent.run(self.blackboard, ctx)
        trace.append({"step": 4, "agent": self.forecaster_agent.agent_id, "output": res_forecaster})

        # 5. Geotechnical Advisor (Grounded RAG Explanations)
        res_advisor = self.advisor_agent.run(self.blackboard, ctx)
        trace.append({"step": 5, "agent": self.advisor_agent.agent_id, "output": res_advisor})

        # 6. Arbiter & Responder (Asymmetric Caution & Alert Package)
        res_arbiter = self.arbiter_agent.run(self.blackboard, ctx)
        trace.append({"step": 6, "agent": self.arbiter_agent.agent_id, "output": res_arbiter})

        return {
            "zone_id": zone_id,
            "final_alert_package": res_arbiter.get("alert_package", {}),
            "effective_level": res_arbiter.get("decision", {}).get("effective_alert_level", "RED"),
            "discrepancy_detected": res_arbiter.get("decision", {}).get("discrepancy_detected", True),
            "trace": trace,
            "head_hash": self.blackboard.get_head_hash(),
        }

    def reset_velocity_history(self, zone_id: Optional[str] = None) -> None:
        """
        Clears the kinematic trend baseline for a zone (or all zones).
        Callers should use this when the telemetry source changes discontinuously
        (for example, loading a different dataset scenario) so unrelated readings
        are never fitted as one accelerating series.
        """
        if zone_id is None:
            self.velocity_history.clear()
        else:
            self.velocity_history.pop(zone_id, None)

    def submit_desensitization_proposal(
        self,
        zone_id: str,
        operator_id: str,
        reason: str,
        approver_role: str,
        duration_hours: float = 8.0,
        requested_sensitivity: float = 0.85,
        approver_credential: str = "",
    ) -> Dict[str, Any]:
        """Routes human proposal to Agent 7 (Governance)."""
        ctx = {
            "zone_id": zone_id,
            "timestamp": self.time_provider(),
            "desensitization_request": {
                "operator_id": operator_id,
                "reason": reason,
                "approver_role": approver_role,
                "duration_hours": duration_hours,
                "requested_sensitivity": requested_sensitivity,
                "approver_credential": approver_credential,
            },
        }
        return self.governance_agent.run(self.blackboard, ctx)

    def evaluate_challenger_model(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
        recall_tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        """Routes candidate model validation to Agent 8 (Evaluator)."""
        ctx = {
            "timestamp": self.time_provider(),
            "model_evaluation_request": {
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
                "recall_tolerance": recall_tolerance,
            },
        }
        return self.evaluator_agent.run(self.blackboard, ctx)

    def audit_quiet_shifts(
        self,
        total_operational_hours: int = 720,
        alarm_timestamps: Optional[List[float]] = None,
        sample_size: int = 5,
    ) -> Dict[str, Any]:
        """Routes negative sample auditing to Agent 8 (Evaluator)."""
        ctx = {
            "timestamp": self.time_provider(),
            "quiet_sampling_request": {
                "total_operational_hours": total_operational_hours,
                "alarm_timestamps": alarm_timestamps or [],
                "sample_size": sample_size,
            },
        }
        return self.evaluator_agent.run(self.blackboard, ctx)
