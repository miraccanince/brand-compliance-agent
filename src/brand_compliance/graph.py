import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama

from brand_compliance.config import OLLAMA_MODEL

from brand_compliance.agents.rule_extractor import extract_rules
from brand_compliance.agents.compliance_checker import check_compliance
from brand_compliance.agents.correction_suggester import suggest_corrections
from brand_compliance.models import BrandRule, ComplianceReport

logger = logging.getLogger(__name__)


class BrandState(TypedDict):
    pdf_path:          str
    ad_text:           str
    session_id:        str
    brand_rules:       list[dict[str, Any]]
    compliance_report: dict[str, Any]
    corrections:       list[dict[str, Any]]
    corrections_error: str
    error:             str


def make_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")


def build_graph() -> Any:
    llm = make_llm()

    def node_extract_rules(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=extract_rules | session={state['session_id']}")
        if state["brand_rules"]:
            logger.info(f"[GRAPH] node=extract_rules skipped, using {len(state['brand_rules'])} supplied rules")
            return {}
        try:
            rules = extract_rules(state["pdf_path"], llm)
            logger.info(f"[GRAPH] node=extract_rules done | rules={len(rules)}")
            return {"brand_rules": [r.model_dump() for r in rules]}
        except Exception as e:
            logger.error(f"[GRAPH] node=extract_rules failed | error={e}")
            return {"error": f"Rule extraction failed: {e}"}

    def node_check_compliance(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=check_compliance | session={state['session_id']}")
        try:
            rules = [BrandRule(**r) for r in state["brand_rules"]]
            report = check_compliance(state["ad_text"], rules)
            logger.info(f"[GRAPH] node=check_compliance done | score={report.score} | violations={len(report.violations)}")
            return {"compliance_report": report.model_dump()}
        except Exception as e:
            logger.error(f"[GRAPH] node=check_compliance failed | error={e}")
            return {"error": f"Compliance check failed: {e}"}

    def node_suggest_corrections(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=suggest_corrections | session={state['session_id']}")
        try:
            report = ComplianceReport(**state["compliance_report"])
            corrections = suggest_corrections(state["ad_text"], report, llm)
            logger.info(f"[GRAPH] node=suggest_corrections done | corrections={len(corrections)}")
            return {"corrections": [c.model_dump() for c in corrections]}
        except Exception as e:
            # The confirmed compliance result is kept; the failure is reported separately.
            logger.error(f"[GRAPH] node=suggest_corrections failed | error={e}")
            return {"corrections": [], "corrections_error": f"Correction suggestion failed: {e}"}

    graph = StateGraph(BrandState)
    graph.add_node("extract_rules",       node_extract_rules)
    graph.add_node("check_compliance",    node_check_compliance)
    graph.add_node("suggest_corrections", node_suggest_corrections)

    graph.set_entry_point("extract_rules")
    graph.add_edge("extract_rules",       "check_compliance")
    graph.add_edge("check_compliance",    "suggest_corrections")
    graph.add_edge("suggest_corrections", END)

    return graph.compile()
