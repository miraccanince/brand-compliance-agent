import logging
import time

from langchain_core.messages import HumanMessage

from brand_compliance.models import ComplianceReport, CorrectionSuggestion
from brand_compliance.utils import extract_list, parse_json_response

logger = logging.getLogger(__name__)


def suggest_corrections(ad_text: str, report: ComplianceReport, llm) -> list[CorrectionSuggestion]:
    """Agent 3: İhlaller için düzeltme önerir."""
    if report.compliant or not report.violations:
        logger.info("Agent 3 skipped | no violations to fix")
        return []

    logger.info(f"Agent 3 started | violations={len(report.violations)}")
    t0 = time.time()

    violations_text = "\n".join(
        f"- [{v.severity.upper()}] {v.violation}" for v in report.violations
    )

    prompt = f"""Fix the following advertisement text to comply with brand guidelines.

ORIGINAL ADVERTISEMENT:
{ad_text}

VIOLATIONS TO FIX:
{violations_text}

Return a JSON object with a "corrections" key containing an array. Each item must have:
- "original": the exact problematic text
- "corrected": the fixed replacement
- "reason": which violation this fixes

Return only valid JSON."""

    logger.info("Calling LLM for correction suggestions...")
    response = llm.invoke([HumanMessage(content=prompt)])
    data = parse_json_response(response.content)
    items = extract_list(data)

    corrections = [CorrectionSuggestion(**item) for item in items]
    logger.info(f"Agent 3 complete | corrections={len(corrections)} | elapsed={time.time()-t0:.1f}s")
    return corrections
