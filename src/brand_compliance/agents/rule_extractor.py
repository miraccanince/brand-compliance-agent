import logging
import time

import fitz
from langchain_core.messages import HumanMessage

from brand_compliance.config import PDF_TEXT_LIMIT
from brand_compliance.models import BrandRule, RuleCheck
from brand_compliance.utils import extract_list, parse_json_response

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def validate_check(raw, document_text: str) -> RuleCheck | None:
    """
    Return the check if it is well formed and its words or text appear in the document.

    The LLM copies parameters from the PDF. If it invents or alters them, the check is
    dropped and the rule is judged by the compliance LLM instead.
    """
    if not raw:
        return None
    try:
        check = RuleCheck(**raw)
    except Exception as e:
        logger.warning(f"Check dropped, invalid structure | check={raw!r} | error={e}")
        return None

    document = _normalize(document_text)
    if check.type == "forbidden_words":
        words = [w for w in check.words if w.strip()]
        if not words or not all(_normalize(w) in document for w in words):
            logger.warning(f"Check dropped, words missing from document | check={raw!r}")
            return None
        return RuleCheck(type=check.type, words=words)
    if check.type == "required_text":
        if not check.text or _normalize(check.text) not in document:
            logger.warning(f"Check dropped, text missing from document | check={raw!r}")
            return None
        return RuleCheck(type=check.type, text=check.text, position=check.position)
    return RuleCheck(type=check.type)


def extract_rules(pdf_path: str, llm) -> list[BrandRule]:
    """Agent 1: PDF'den marka kurallarını çıkarır."""
    logger.info(f"Agent 1 started | pdf={pdf_path}")
    t0 = time.time()

    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    text = "".join(page.get_text() for page in doc)
    doc.close()
    logger.info(f"PDF read | pages={page_count} | chars={len(text)}")

    prompt = f"""Extract all brand guidelines from this document.
Return a JSON object with a "rules" key containing an array. Each item must have:
- "category": one of logo, color, typography, language, slogan, other
- "rule": clear description of the rule
- "value": the specific value or constraint (null if not applicable)
- "check": null, or an object when the rule can be verified exactly from the advertisement text:
    {{"type": "forbidden_words", "words": ["..."]}}   words or phrases the text must not contain
    {{"type": "required_text", "text": "...", "position": "anywhere" or "last_line"}}   exact text the advertisement must contain
    {{"type": "no_all_caps"}}   no full line may be written entirely in capital letters
  Copy words and text exactly as written in the document. Use null for every other rule,
  including rules about design, colors, fonts, sizes or tone.

Document:
{text[:PDF_TEXT_LIMIT]}

Return only valid JSON."""

    logger.info("Calling LLM for rule extraction...")
    response = llm.invoke([HumanMessage(content=prompt)])
    data = parse_json_response(response.content)
    items = extract_list(data)

    rules = []
    for item in items:
        item["check"] = validate_check(item.get("check"), text)
        rules.append(BrandRule(**item))
    logger.info(f"Agent 1 complete | rules={len(rules)} | elapsed={time.time()-t0:.1f}s")
    return rules
