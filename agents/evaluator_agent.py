"""
evaluator_agent.py - Model Quality Gate & Benchmark QA Agent.

Role:
Manages the frozen golden benchmark dataset, executes the Champion vs. Challenger
non-inferiority recall validation gate, and audits quiet non-alarm shifts to counteract label bias.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.evaluation_tools import evaluate_model_gate, sample_quiet_periods


class EvaluatorAgent(BaseAgent):
    """Agent 8: Model Quality Gate & Benchmark QA."""

    def __init__(self):
        super().__init__(
            agent_id="AGENT_EVALUATOR",
            role="Model Quality Gate & Benchmark QA",
            plugin_name="EvaluatorPlugin",
        )

    @kernel_function(name="evaluate_gate", description="Evaluates challenger model against champion on frozen golden set.")
    def evaluate_gate(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
        recall_tolerance: float = 0.0,
    ) -> Dict[str, Any]:
        return evaluate_model_gate(champion_metrics, challenger_metrics, recall_tolerance)

    @kernel_function(name="audit_quiet_periods", description="Samples un-alarmed quiet shifts to prevent negative label bias.")
    def audit_quiet_periods(
        self,
        total_operational_hours: int,
        alarm_timestamps: List[float],
        sample_size: int = 5,
    ) -> List[Dict[str, Any]]:
        return sample_quiet_periods(total_operational_hours, alarm_timestamps, sample_size)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads model benchmark candidates or audit requests from context,
        invokes tools, and writes 'golden_pool' updates to the shared blackboard.
        """
        results = {}

        # 1. Evaluate Candidate Model Gate if provided
        if model_eval_req := context.get("model_evaluation_request"):
            champ = model_eval_req.get("champion_metrics", {"recall": 0.95, "f1_score": 0.90, "precision": 0.85})
            chal = model_eval_req.get("challenger_metrics", {"recall": 0.92, "f1_score": 0.91, "precision": 0.90})
            tol = model_eval_req.get("recall_tolerance", 0.0)

            gate_result = self.evaluate_gate(champ, chal, tol)

            golden_pool = blackboard.read("golden_pool")
            golden_pool["last_model_gate_result"] = gate_result
            blackboard.write("golden_pool", golden_pool, self.agent_id)
            results["gate_result"] = gate_result

        # 2. Audit Quiet Periods if requested
        if quiet_req := context.get("quiet_sampling_request"):
            hours = quiet_req.get("total_operational_hours", 720)
            alarms = quiet_req.get("alarm_timestamps", [12.0, 48.0, 110.0])
            size = quiet_req.get("sample_size", 5)

            quiet_samples = self.audit_quiet_periods(hours, alarms, size)

            golden_pool = blackboard.read("golden_pool")
            golden_pool["audited_quiet_periods"] = quiet_samples
            blackboard.write("golden_pool", golden_pool, self.agent_id)
            results["quiet_samples"] = quiet_samples

        return results
