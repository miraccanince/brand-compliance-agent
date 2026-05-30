import logging
from typing import Any, TypedDict

from typing import Any

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL

from agents.rule_extractor import extract_rules
from agents.content_analyzer import analyze_content
from agents.compliance_checker import check_compliance
from agents.correction_suggester import suggest_corrections
from models import ComplianceReport
from vector_store import RuleVectorStore

logger = logging.getLogger(__name__)


class BrandState(TypedDict):
    pdf_path:          str
    ad_text:           str
    session_id:        str
    brand_rules:       list[dict[str, Any]]
    content_analysis:  dict[str, Any]
    compliance_report: dict[str, Any]
    ml_scores:         dict[str, float]
    corrections:       list[dict[str, Any]]
    error:             str


def build_graph(vector_store: RuleVectorStore) -> Any:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")

    def node_extract_rules(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=extract_rules | session={state['session_id']}")
        try:
            rules = extract_rules(state["pdf_path"], llm)
            vector_store.store_rules(rules, state["session_id"])
            logger.info(f"[GRAPH] node=extract_rules done | rules={len(rules)}")
            return {"brand_rules": [r.model_dump() for r in rules]}
        except Exception as e:
            logger.error(f"[GRAPH] node=extract_rules failed | error={e}")
            return {"error": f"Rule extraction failed: {e}"}

    def node_analyze_content(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=analyze_content | session={state['session_id']}")
        try:
            analysis = analyze_content(state["ad_text"], llm)
            logger.info("[GRAPH] node=analyze_content done")
            return {"content_analysis": analysis}
        except Exception as e:
            logger.error(f"[GRAPH] node=analyze_content failed | error={e}")
            return {"error": f"Content analysis failed: {e}"}

    def node_check_compliance(state: BrandState) -> dict:
        if state.get("error"):
            return {}
        logger.info(f"[GRAPH] node=check_compliance | session={state['session_id']}")
        try:
            rules = vector_store.search_relevant_rules(state["ad_text"], state["session_id"])
            logger.info(f"[GRAPH] relevant rules fetched from Qdrant | count={len(rules)}")
            report, ml_scores = check_compliance(state["ad_text"], rules, state["content_analysis"], llm)
            logger.info(f"[GRAPH] node=check_compliance done | score={report.score} | violations={len(report.violations)}")
            return {"compliance_report": report.model_dump(), "ml_scores": ml_scores}
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
            logger.error(f"[GRAPH] node=suggest_corrections failed | error={e}")
            return {"error": f"Correction suggestion failed: {e}"}

    graph = StateGraph(BrandState)
    graph.add_node("extract_rules",       node_extract_rules)
    graph.add_node("analyze_content",     node_analyze_content)
    graph.add_node("check_compliance",    node_check_compliance)
    graph.add_node("suggest_corrections", node_suggest_corrections)

    graph.set_entry_point("extract_rules")
    graph.add_edge("extract_rules",       "analyze_content")
    graph.add_edge("analyze_content",     "check_compliance")
    graph.add_edge("check_compliance",    "suggest_corrections")
    graph.add_edge("suggest_corrections", END)

    return graph.compile()
