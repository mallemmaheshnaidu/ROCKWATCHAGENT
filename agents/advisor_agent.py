"""
advisor_agent.py - Geotechnical Advisor & Grounded RAG Agent.

Role:
Retrieves authoritative geotechnical SOPs and precedent case studies via local RAG,
cites explicit passage IDs, and formats plain-language explanations for operators.
Powered by LLMRouter (LiteLLM proxy / 11-key Gemini pool) with automatic deterministic fallback.
"""

from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent
from blackboard import MineBlackboard
from sk_compat import kernel_function
from tools.rag_tools import retrieve_sop, find_similar_incidents
from tools.llm_router import get_llm_router


class AdvisorAgent(BaseAgent):
    """Agent 5: Geotechnical Advisor & RAG Explainer."""

    def __init__(self, docs_dir: str = "docs"):
        super().__init__(
            agent_id="AGENT_ADVISOR",
            role="Geotechnical Advisor & Grounded RAG Explainer",
            plugin_name="AdvisorPlugin",
        )
        self.docs_dir = docs_dir

    @kernel_function(name="lookup_sop", description="Retrieves verified SOP passage by keyword search.")
    def lookup_sop(self, query: str) -> Dict[str, Any]:
        return retrieve_sop(query, docs_dir=self.docs_dir)

    @kernel_function(name="lookup_incidents", description="Finds historical mine incident case precedents.")
    def lookup_incidents(self, keywords: List[str]) -> List[Dict[str, Any]]:
        return find_similar_incidents(keywords, docs_dir=self.docs_dir)

    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads 'tarp_status' and 'forecast', retrieves matching SOPs & cases,
        synthesizes a grounded explanation via LLM Router or deterministic fallback,
        and writes to 'advisor_notes'.
        """
        zone_id = context.get("zone_id", "ZONE_1")
        tarp_status = blackboard.read("tarp_status").get(zone_id, {})
        clean_telem = blackboard.read("clean_telemetry").get(zone_id, {})

        tarp_level = tarp_status.get("final_level", "GREEN")
        query = f"TARP Level {tarp_level} actions"

        # Tool 1: Grounded RAG retrieval of mandatory SOP
        sop_doc = self.lookup_sop(query)

        # Tool 2: Precedent case retrieval
        keywords = ["blast", "dust"] if context.get("is_blast_event") else ["rainfall", "detachment"]
        incidents = self.lookup_incidents(keywords)

        # Baseline deterministic explanation
        citation = sop_doc.get("passage_id") or "[NO_PASSAGE_FOUND]"
        if sop_doc.get("found"):
            deterministic_explanation = (
                f"Operational Advisory for {zone_id}: Evaluated as {tarp_level}. "
                f"Mandatory protocol governed by {citation}: {sop_doc.get('text', '')[:180]}..."
            )
        else:
            deterministic_explanation = f"Operational Advisory for {zone_id}: Information not found in geotechnical documentation."

        explanation = deterministic_explanation
        llm_synthesized = False
        model_source = "DETERMINISTIC_RULES"

        # Multi-Agent LLM Call: Query LiteLLM / 11-key Gemini pool only if explicitly enabled
        if context.get("enable_llm", False) and sop_doc.get("found"):
            router = get_llm_router()
            llm_res = router.generate_advisor_explanation(
                zone_id=zone_id,
                tarp_level=tarp_level,
                citation=citation,
                sop_text=sop_doc.get("text", ""),
                incidents_count=len(incidents),
                telemetry=clean_telem,
            )
            if llm_res:
                explanation = llm_res
                llm_synthesized = True
                model_source = router.get_status().get("active_endpoint", "LLM_ROUTER")

        advisor_data = {
            "zone_id": zone_id,
            "tarp_level": tarp_level,
            "primary_citation": citation,
            "sop_passage": sop_doc,
            "matching_incidents_count": len(incidents),
            "explanation": explanation,
            "llm_synthesized": llm_synthesized,
            "model_source": model_source,
        }

        # Explicit Blackboard Write
        blackboard.write("advisor_notes", {zone_id: advisor_data}, self.agent_id)

        return advisor_data
