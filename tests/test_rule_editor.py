import pytest

from brand_compliance.agents.compliance_checker import check_compliance
from brand_compliance.models import BrandRule, RuleCheck
from brand_compliance.rule_editor import rows_to_rules, rules_to_rows


def row(**overrides):
    base = {"category": "slogan", "rule": "Always include the tagline", "value": "", "check": "none",
            "words": "", "text": "", "position": "anywhere"}
    return {**base, **overrides}


def test_extracted_rules_round_trip_through_rows():
    rules = [
        BrandRule(category="language", rule="Forbidden words", check=RuleCheck(type="forbidden_words", words=["cheap", "buy now"])),
        BrandRule(category="slogan", rule="Tagline last", value="v", check=RuleCheck(type="required_text", text="Think Bigger", position="last_line")),
        BrandRule(category="language", rule="No shouting", check=RuleCheck(type="no_all_caps")),
        BrandRule(category="language", rule="Be formal"),
    ]
    assert rows_to_rules(rules_to_rows(rules)) == rules


def test_user_can_supply_a_required_text_check_without_any_brand_values_in_the_code():
    rows = [row(check="required_text", text="Think Bigger", position="last_line")]
    (rule,) = rows_to_rules(rows)
    assert rule.check == RuleCheck(type="required_text", text="Think Bigger", position="last_line")

    # and the check is enforced on advertisement text
    report = check_compliance("Our news\nThink Bigger", [rule])
    assert report.violations == []
    report = check_compliance("Think Bigger\nOur news", [rule])
    assert [v.rule.rule for v in report.violations] == ["Always include the tagline"]


def test_words_are_split_on_commas_and_trimmed():
    (rule,) = rows_to_rules([row(check="forbidden_words", words=" cheap ,free,, buy now ")])
    assert rule.check.words == ["cheap", "free", "buy now"]


def test_blank_rows_and_missing_values_are_handled():
    rows = [row(category="", rule="", check=None), {"rule": "Be formal", "category": None, "check": float("nan")}]
    (rule,) = rows_to_rules(rows)
    assert (rule.category, rule.rule, rule.check, rule.value) == ("other", "Be formal", None, None)


def test_every_problem_is_reported_together_and_nothing_is_silently_dropped():
    rows = [
        row(rule="", check="no_all_caps"),
        row(check="forbidden_words", words=" , "),
        row(check="required_text", text=""),
        row(check="required_text", text="x", position="top"),
        row(check="regex"),
    ]
    with pytest.raises(ValueError) as error:
        rows_to_rules(rows)
    message = str(error.value)
    for expected in ("Row 1: the rule text is empty", "Row 2: forbidden_words needs", "Row 3: required_text needs",
                     "Row 4: position must be", "Row 5: unknown check type"):
        assert expected in message
