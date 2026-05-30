import json
import logging
import time

from langchain_core.messages import HumanMessage

from config import ML_THRESHOLD, USE_ML_FILTER
from models import BrandRule, ComplianceReport
from utils import parse_json_response

logger = logging.getLogger(__name__)


def check_compliance(
    ad_text: str,
    rules: list[BrandRule],
    content_analysis: dict,
    llm,
) -> tuple[ComplianceReport, dict[str, float]]:
    """
    Agent 3: Kurallara uygunluk raporu üretir.

    Returns: (ComplianceReport, ml_scores)
    ml_scores is empty dict if ML filter is disabled.
    """
    logger.info(f"Agent 3 started | rules_count={len(rules)} | use_ml={USE_ML_FILTER}")
    t0 = time.time()

    ml_scores: dict[str, float] = {}
    llm_rules = rules

    if USE_ML_FILTER and len(rules) > 2:
        from agents.zero_shot_filter import filter_rules
        ml_scores, filtered = filter_rules(ad_text, rules, threshold=ML_THRESHOLD)
        # Fall back to all rules if ML finds nothing (avoid false negatives)
        llm_rules = filtered if filtered else rules
        logger.info(f"ML pre-filter applied | {len(rules)} → {len(llm_rules)} rules sent to LLM")

    rules_text = "\n".join(
        f"- [{r.category.upper()}] {r.rule}" + (f" → {r.value}" if r.value else "")
        for r in llm_rules
    )

    prompt = f"""Check this advertisement against the brand guidelines and identify all violations.

BRAND RULES:
{rules_text}

ADVERTISEMENT TEXT:
{ad_text}

Return a JSON object:
{{
  "compliant": true or false,
  "score": 0-100 (100 = fully compliant),
  "violations": [
    {{
      "rule": {{"category": "...", "rule": "...", "value": null}},
      "violation": "specific description of what is wrong",
      "severity": "critical or warning"
    }}
  ]
}}

Return only valid JSON, no other text."""

    logger.info("Calling LLM for compliance check...")
    response = llm.invoke([HumanMessage(content=prompt)])
    data = parse_json_response(response.content)
    report = ComplianceReport(**data)

    logger.info(
        f"Agent 3 complete | compliant={report.compliant} | score={report.score} "
        f"| violations={len(report.violations)} | elapsed={time.time()-t0:.1f}s"
    )
    for v in report.violations:
        logger.warning(f"  [{v.severity.upper()}] {v.rule.category} — {v.violation}")

    return report, ml_scores
