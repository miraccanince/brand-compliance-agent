import logging
import time

import fitz
from langchain_core.messages import HumanMessage

from config import PDF_TEXT_LIMIT
from models import BrandRule
from utils import extract_list, parse_json_response

logger = logging.getLogger(__name__)


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

Document:
{text[:PDF_TEXT_LIMIT]}

Return only valid JSON."""

    logger.info("Calling LLM for rule extraction...")
    response = llm.invoke([HumanMessage(content=prompt)])
    data = parse_json_response(response.content)
    items = extract_list(data)

    rules = [BrandRule(**item) for item in items]
    logger.info(f"Agent 1 complete | rules={len(rules)} | elapsed={time.time()-t0:.1f}s")
    return rules
