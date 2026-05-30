import logging
import time

from langchain_core.messages import HumanMessage

from utils import parse_json_response

logger = logging.getLogger(__name__)


def analyze_content(ad_text: str, llm) -> dict:
    """Agent 2: Reklam metnini analiz eder."""
    logger.info(f"Agent 2 started | ad_text_len={len(ad_text)}")
    t0 = time.time()

    prompt = f"""Analyze this advertisement text and return a JSON object with:
- "words": list of all significant words used
- "colors": any color references (hex codes or color names)
- "slogans": any slogans or taglines detected
- "tone": overall tone (formal/informal/aggressive/friendly/neutral)
- "sensitive_words": words that might violate brand language rules

Ad text:
{ad_text}

Return only valid JSON, no other text."""

    logger.info("Calling LLM for content analysis...")
    response = llm.invoke([HumanMessage(content=prompt)])
    result = parse_json_response(response.content)

    sensitive = result.get("sensitive_words") or []
    logger.info(f"Agent 2 complete | tone={result.get('tone', 'unknown')} | sensitive_words={len(sensitive)} | elapsed={time.time()-t0:.1f}s")
    return result
