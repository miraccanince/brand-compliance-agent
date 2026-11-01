import json

from brand_compliance.models import BrandRule, ComplianceReport, ComplianceViolation
from brand_compliance.report_generator import generate_json, generate_pdf

RULE = BrandRule(category="language", rule="No forbidden words")
REPORT = ComplianceReport(
    compliant=False, score=50.0, checks_evaluated=2,
    violations=[ComplianceViolation(rule=RULE, violation="Contains forbidden word(s): free")],
)


def test_json_export_reports_exact_check_counts_and_makes_no_overall_compliance_claim():
    summary = json.loads(generate_json("ad", REPORT, [], [RULE]))["summary"]
    assert summary["exact_checks_evaluated"] == 2
    assert summary["exact_checks_failed"] == 1
    assert "compliant" not in summary


def test_pdf_export_builds():
    pdf = generate_pdf("ad", REPORT, [], [RULE])
    assert bytes(pdf)[:4] == b"%PDF"
