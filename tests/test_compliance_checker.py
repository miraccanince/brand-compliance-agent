import pytest

from brand_compliance.agents.compliance_checker import check_compliance, run_check
from brand_compliance.models import BrandRule, RuleCheck

TAGLINE = "Innovation. Trust. Excellence."
FORBIDDEN = RuleCheck(type="forbidden_words", words=["cheap", "free", "buy now", "limited time"])
TAGLINE_ANYWHERE = RuleCheck(type="required_text", text=TAGLINE)
TAGLINE_LAST = RuleCheck(type="required_text", text=TAGLINE, position="last_line")
NO_CAPS = RuleCheck(type="no_all_caps")


# ---- exact checks ---------------------------------------------------------

@pytest.mark.parametrize("text, found", [
    ("Get it FREE now!", "free"),
    ("Please buy now and save", "buy now"),
    ("Limited time offer", "limited time"),
])
def test_forbidden_words_are_found_whole_word_case_insensitive(text, found):
    assert f"Contains forbidden word(s): {found}" == run_check(FORBIDDEN, text)


@pytest.mark.parametrize("text", [
    "Enjoy the freedom of unlimited storage.",       # "free" inside "freedom", "limited" inside "unlimited"
    "Read our buying guide for details.",             # "buy" without "buy now"
    "A cheaper option is not listed, nor is cheapest.",  # whole words only
])
def test_forbidden_words_ignore_look_alikes(text):
    assert run_check(FORBIDDEN, text) is None


def test_required_text_anywhere_and_last_line():
    assert run_check(TAGLINE_ANYWHERE, f"Hello\n{TAGLINE}") is None
    assert run_check(TAGLINE_ANYWHERE, f"{TAGLINE}\nHello") is None
    assert run_check(TAGLINE_LAST, f"Hello\n{TAGLINE}\n\n") is None   # trailing blank lines ignored
    assert "last line" in run_check(TAGLINE_LAST, f"{TAGLINE}\nHello")
    assert "not found" in run_check(TAGLINE_ANYWHERE, "Hello\nInnovation. Trust.")  # abbreviated
    assert run_check(TAGLINE_LAST, "") is not None


def test_no_all_caps_flags_full_capital_lines_only():
    assert run_check(NO_CAPS, "URGENT NOTICE: SAVE ON ALL SERVICES THIS MONTH.") is not None
    assert run_check(NO_CAPS, "Get it FREE now! URGENT news.") is None  # single capital words are fine
    assert run_check(NO_CAPS, f"We announce our platform.\n{TAGLINE}") is None
    assert run_check(NO_CAPS, "FAQ") is None  # one word is not a shouted line


# ---- report -----------------------------------------------------------------

FORBIDDEN_RULE = BrandRule(category="language", rule="Forbidden words: cheap, free", check=FORBIDDEN)
TAGLINE_RULES = [
    BrandRule(category="slogan", rule="Always include the tagline", check=TAGLINE_ANYWHERE),
    BrandRule(category="slogan", rule="Tagline must be at the bottom", check=TAGLINE_LAST),
]
TONE_RULE = BrandRule(category="language", rule="Always use formal, professional tone")
VISUAL_RULE = BrandRule(category="logo", rule="Logo must appear on white")


def test_report_comes_only_from_exact_checks():
    rules = [FORBIDDEN_RULE, *TAGLINE_RULES, TONE_RULE, VISUAL_RULE]

    report = check_compliance(f"Get it free\n{TAGLINE}", rules)

    assert [v.rule for v in report.violations] == [FORBIDDEN_RULE]
    assert report.violations[0].violation == "Contains forbidden word(s): free"
    assert report.checks_evaluated == 3  # the tone and logo rules have no exact check and are not evaluated
    assert (report.compliant, report.score) == (False, 66.7)  # 2 of 3 checks passed


def test_all_checks_passing_is_compliant_with_full_score():
    report = check_compliance(f"A formal announcement.\n{TAGLINE}", [FORBIDDEN_RULE, *TAGLINE_RULES, TONE_RULE])
    assert (report.compliant, report.score, report.violations, report.checks_evaluated) == (True, 100.0, [], 3)


def test_tagline_presence_wording_and_position_are_exact_checks():
    rules = TAGLINE_RULES + [BrandRule(category="slogan", rule="Do not modify the tagline", check=TAGLINE_ANYWHERE)]

    top = check_compliance(f"{TAGLINE}\nWe announce new hours.", rules)
    assert [v.rule.rule for v in top.violations] == ["Tagline must be at the bottom"]

    abbreviated = check_compliance("We announce new hours.\nInnovation. Trust.", rules)
    assert len(abbreviated.violations) == 3

    clean = check_compliance(f"We announce new hours.\n{TAGLINE}", rules)
    assert (clean.compliant, clean.score) == (True, 100.0)


def test_visual_category_rule_with_an_exact_check_is_still_checked():
    rule = BrandRule(category="color", rule="Do not use red", check=RuleCheck(type="forbidden_words", words=["red"]))
    report = check_compliance("Our red banner", [rule])
    assert [v.violation for v in report.violations] == ["Contains forbidden word(s): red"]


def test_no_exact_check_is_an_error_not_a_perfect_score():
    with pytest.raises(ValueError, match="nothing could be checked"):
        check_compliance("Any text", [TONE_RULE, VISUAL_RULE])
    with pytest.raises(ValueError, match="nothing could be checked"):
        check_compliance("Any text", [])
