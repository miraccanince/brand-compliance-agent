import logging
import re
import time

from brand_compliance.models import BrandRule, ComplianceReport, ComplianceViolation, RuleCheck

logger = logging.getLogger(__name__)


def _squash(text: str) -> str:
    return " ".join(text.split())


def run_check(check: RuleCheck, ad_text: str) -> str | None:
    """Return the reason the text breaks the check, or None if it passes."""
    lines = [line.strip() for line in ad_text.strip().splitlines() if line.strip()]

    if check.type == "forbidden_words":
        found = [
            w.strip() for w in check.words
            if re.search(rf"(?<!\w){re.escape(w.strip())}(?!\w)", ad_text, re.IGNORECASE)
        ]
        return f"Contains forbidden word(s): {', '.join(found)}" if found else None

    if check.type == "required_text":
        searched = _squash(lines[-1] if lines else "") if check.position == "last_line" else _squash(ad_text)
        if _squash(check.text) in searched:
            return None
        where = "on the last line" if check.position == "last_line" else "in the text"
        return f'Required text "{check.text}" not found {where}'

    shouting = [line for line in lines if len(line.split()) >= 2 and line.isupper()]
    return f"Entire line is in capitals: {shouting[0]!r}" if shouting else None


def check_compliance(ad_text: str, rules: list[BrandRule]) -> ComplianceReport:
    """
    Agent 2: Kurallara uygunluk raporu üretir.

    Only rules with an exact check (see RuleCheck) are evaluated. Rules without one are
    not evaluated and say nothing about the advertisement.

    score = 100 * (checks passed) / (checks evaluated). It is a simple proportion, not
    calibrated model confidence. `compliant` only means that no evaluated check failed.
    Raises ValueError if no rule has an exact check, so a result is never reported
    without something having been checked.
    """
    logger.info(f"Agent 2 started | rules_count={len(rules)}")
    t0 = time.time()

    exact = [(rule, run_check(rule.check, ad_text)) for rule in rules if rule.check]
    if not exact:
        raise ValueError("No rule has an exact check, so nothing could be checked")

    violations = [
        ComplianceViolation(rule=rule, violation=reason) for rule, reason in exact if reason is not None
    ]
    report = ComplianceReport(
        compliant=not violations,
        violations=violations,
        score=round(100 * (len(exact) - len(violations)) / len(exact), 1),
        checks_evaluated=len(exact),
    )

    logger.info(
        f"Agent 2 complete | checks={report.checks_evaluated} | not_evaluated={len(rules) - len(exact)} "
        f"| violations={len(violations)} | score={report.score} | elapsed={time.time()-t0:.1f}s"
    )
    for v in violations:
        logger.warning(f"  [{v.severity.upper()}] {v.rule.category} — {v.violation}")
    return report
