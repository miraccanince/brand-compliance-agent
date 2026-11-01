import json
from pathlib import Path

import pytest

from brand_compliance.agents.rule_extractor import extract_rules, validate_check

SAMPLE_PDF = str(Path(__file__).parent.parent / "samples" / "test_brand_guidelines.pdf")
DOCUMENT = 'Forbidden words: cheap, free, buy now.\nRequired tagline: "Innovation.   Trust. Excellence."'


def test_valid_checks_are_kept():
    assert validate_check({"type": "forbidden_words", "words": ["Cheap", "buy now"]}, DOCUMENT).words == ["Cheap", "buy now"]
    required = validate_check({"type": "required_text", "text": "Innovation. Trust. Excellence.", "position": "last_line"}, DOCUMENT)
    assert (required.type, required.position) == ("required_text", "last_line")  # whitespace in the PDF is normalized
    assert validate_check({"type": "no_all_caps"}, DOCUMENT).type == "no_all_caps"


@pytest.mark.parametrize("raw", [
    None,
    {},
    {"type": "regex", "words": ["x"]},                                  # unknown type
    {"type": "forbidden_words", "words": []},                           # no parameters
    {"type": "forbidden_words", "words": ["  "]},
    {"type": "forbidden_words", "words": ["cheap", "invented"]},        # one word is not in the document
    {"type": "required_text", "text": "Innovation. Quality."},          # text that is not in the document
    {"type": "required_text", "text": ""},
    {"type": "required_text", "text": "Innovation. Trust. Excellence.", "position": "top"},
    "forbidden_words",
])
def test_invalid_or_unsupported_checks_are_dropped(raw):
    assert validate_check(raw, DOCUMENT) is None


def test_extract_rules_keeps_valid_checks_and_falls_back_for_invalid_ones():
    reply = json.dumps({"rules": [
        {"category": "language", "rule": "Forbidden words: cheap, free", "value": None,
         "check": {"type": "forbidden_words", "words": ["cheap", "free"]}},
        {"category": "slogan", "rule": "Tagline must be last", "value": None,
         "check": {"type": "required_text", "text": "Made-up tagline", "position": "last_line"}},
        {"category": "logo", "rule": "Minimum logo size: 50px width", "value": "50px", "check": None},
        {"category": "language", "rule": "Always use formal, professional tone"},   # no check key at all
    ]})

    class Llm:
        def invoke(self, messages):
            return type("R", (), {"content": reply})()

    rules = extract_rules(SAMPLE_PDF, Llm())

    assert [r.check.type if r.check else None for r in rules] == ["forbidden_words", None, None, None]
    assert rules[2].value == "50px"
