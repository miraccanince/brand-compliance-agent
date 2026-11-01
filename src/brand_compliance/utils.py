import json
import logging
import re

from json_repair import repair_json

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> list | dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from LLM, attempting repair...")
        repaired = repair_json(text)
        return json.loads(repaired)


def extract_list(data: list | dict, fallback: list | None = None) -> list:
    """Return the first list found in a dict, or data itself if already a list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    logger.warning(f"Could not extract list from response: {type(data)}")
    return fallback if fallback is not None else []
