import importlib.util
import json
import re
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "evaluation"
TAGLINE = "Innovation. Trust. Excellence."

spec = importlib.util.spec_from_file_location("run_eval", EVAL / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

GOLD = json.loads((EVAL / "gold_rules.json").read_text(encoding="utf-8"))["rules"]
CASES = json.loads((EVAL / "cases.json").read_text(encoding="utf-8"))["cases"]
SCORED = run_eval.scored_ids()


def pdf_bullets():
    text = "".join(page.get_text() for page in fitz.open(ROOT / "samples" / "test_brand_guidelines.pdf"))
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]


def in_scope(case, rule_id):
    return rule_id not in case["dont_care"]


def test_gold_rules_match_the_pdf_bullets_in_order():
    assert [g["pdf_text"] for g in GOLD] == pdf_bullets()
    assert len(GOLD) == 23


def test_dataset_shape_and_counts():
    assert [c["id"] for c in CASES] == [f"E{i:02d}" for i in range(1, 13)]
    for case in CASES:
        assert set(case["expected"]) <= SCORED
        assert set(case["dont_care"]) <= SCORED
        assert not set(case["expected"]) & set(case["dont_care"])
    positives = sum(len(c["expected"]) for c in CASES)
    dont_care = sum(len(c["dont_care"]) for c in CASES)
    assert (positives, len(CASES) * len(SCORED) - dont_care) == (12, 74)
    assert {i for c in CASES for i in c["expected"]} == SCORED  # every scored rule has a positive case


def test_existing_sample_ad_is_case_e06():
    sample = (ROOT / "samples" / "test.txt").read_text(encoding="utf-8").strip()
    assert next(c for c in CASES if c["id"] == "E06")["text"] == sample


def test_lexical_labels_follow_the_pdf_rules():
    forbidden_rule = next(g["pdf_text"] for g in GOLD if g["score_id"] == "FORBIDDEN_WORDS")
    words = [w.strip() for w in forbidden_rule.split(":", 1)[1].split(",")]

    for case in CASES:
        text, lines = case["text"], case["text"].splitlines()
        has_tagline_text = "innovation. trust" in text.lower()
        derived = {
            "FORBIDDEN_WORDS": any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words),
            "ALL_CAPS": any(len(l.split()) >= 2 and l == l.upper() and re.search("[A-Z]", l) for l in lines),
            "TAGLINE_REQUIRED": not has_tagline_text,
            "TAGLINE_POSITION": has_tagline_text and "innovation. trust" not in lines[-1].lower(),
            "TAGLINE_WORDING": has_tagline_text and TAGLINE not in text,
            "FINANCIAL_DISCLAIMER": "loan" in text.lower(),
        }
        for rule_id, violated in derived.items():
            if in_scope(case, rule_id):
                assert (rule_id in case["expected"]) == violated, (case["id"], rule_id)


def test_frozen_rule_mapping_is_complete_and_valid():
    rules, mapping = run_eval.load_rules()  # exits if any extracted rule is unmapped
    assert set(mapping.values()) <= SCORED | {"VISUAL", "EXCLUDED"}
    assert SCORED <= {mapping[run_eval.norm(r.rule)] for r in rules}


def test_classify_uses_exact_text_only():
    mapping = {"forbidden words: cheap": "FORBIDDEN_WORDS"}
    result = run_eval.classify(["Forbidden  words: Cheap", "Forbidden words: cheap!"], mapping)
    assert result["ids"] == ["FORBIDDEN_WORDS"]  # case and whitespace only
    assert result["unmapped"] == ["Forbidden words: cheap!"]  # near match is not accepted


def test_score_case_counts_hits_misses_and_extras():
    case = {"expected": ["FORBIDDEN_WORDS", "TAGLINE_REQUIRED"], "dont_care": ["TONE"]}
    s = run_eval.score_case(case, ["FORBIDDEN_WORDS", "ALL_CAPS", "TONE", "VISUAL", "EXCLUDED"], SCORED)
    assert s["true_positive"] == ["FORBIDDEN_WORDS"]
    assert s["missed"] == ["TAGLINE_REQUIRED"]
    assert s["false_positive"] == ["ALL_CAPS"]  # TONE is "don't care"
    assert s["unsupported_visual"] == ["VISUAL"]
    twice = run_eval.score_case(case, ["VISUAL", "VISUAL"], SCORED)
    assert len(twice["unsupported_visual"]) == 2  # every report is counted
    assert s["unscored_excluded"] == ["EXCLUDED"]
    assert s["pairs_in_scope"] == 6


def test_call_with_timeout_returns_result_and_raises_on_overrun():
    import time

    assert run_eval.call_with_timeout(lambda: "ok", 5) == "ok"
    with pytest.raises(run_eval.CallTimeout):
        run_eval.call_with_timeout(lambda: time.sleep(5), 0.2)
    assert run_eval.call_with_timeout(lambda: "still works", 5) == "still works"  # alarm was cleared


def test_score_case_only_counts_expected_violations_inside_the_given_scope():
    case = {"expected": ["FORBIDDEN_WORDS", "TONE"], "dont_care": []}
    exact_only = run_eval.score_case(case, ["FORBIDDEN_WORDS"], {"FORBIDDEN_WORDS"})
    assert (exact_only["true_positive"], exact_only["missed"]) == (["FORBIDDEN_WORDS"], [])  # TONE is outside this scope
    advisory = run_eval.score_case(case, [], {"TONE"})
    assert (advisory["true_positive"], advisory["missed"]) == ([], ["TONE"])
