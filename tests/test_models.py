import pytest
from pydantic import ValidationError

from brand_compliance.models import BrandRule, ComplianceReport, ComplianceViolation


def test_brand_rule_coerces_value_to_string():
    assert BrandRule(category="color", rule="Use blue", value=123).value == "123"


def test_brand_rule_value_defaults_to_none():
    assert BrandRule(category="logo", rule="Keep clear space").value is None


@pytest.mark.parametrize("raw, expected", [(150, 100.0), (-5, 0.0), ("72.5", 72.5)])
def test_report_score_is_clamped(raw, expected):
    report = ComplianceReport(compliant=True, violations=[], score=raw)
    assert report.score == expected


def test_violation_severity_defaults_to_warning():
    rule = BrandRule(category="language", rule="No superlatives")
    assert ComplianceViolation(rule=rule, violation="uses 'best'").severity == "warning"


def test_violation_rejects_unknown_severity():
    rule = BrandRule(category="language", rule="No superlatives")
    with pytest.raises(ValidationError):
        ComplianceViolation(rule=rule, violation="x", severity="minor")


def test_brand_rule_check_is_optional_and_typed():
    from brand_compliance.models import RuleCheck

    assert BrandRule(category="language", rule="r").check is None
    rule = BrandRule(category="language", rule="r", check={"type": "forbidden_words", "words": ["free"]})
    assert rule.check == RuleCheck(type="forbidden_words", words=["free"])
    assert BrandRule(**rule.model_dump()) == rule  # survives the graph state round trip


def test_rule_check_rejects_unknown_type_and_position():
    from brand_compliance.models import RuleCheck

    with pytest.raises(ValidationError):
        RuleCheck(type="regex", words=["x"])
    with pytest.raises(ValidationError):
        RuleCheck(type="required_text", text="x", position="top")

