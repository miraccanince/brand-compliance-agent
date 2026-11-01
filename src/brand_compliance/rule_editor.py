"""Convert brand rules to and from editable table rows (used by the dashboard review step)."""
from brand_compliance.models import BrandRule, RuleCheck

CHECK_TYPES = ["none", "forbidden_words", "required_text", "no_all_caps"]
POSITIONS = ["anywhere", "last_line"]


def rules_to_rows(rules: list[BrandRule]) -> list[dict]:
    rows = []
    for rule in rules:
        check = rule.check
        rows.append({
            "category": rule.category,
            "rule": rule.rule,
            "value": rule.value or "",
            "check": check.type if check else "none",
            "words": ", ".join(check.words) if check else "",
            "text": (check.text or "") if check else "",
            "position": check.position if check else "anywhere",
        })
    return rows


def _clean(value) -> str:
    return "" if value is None or value != value else str(value).strip()  # None and NaN become ""


def rows_to_rules(rows: list[dict]) -> list[BrandRule]:
    """
    Build rules from edited rows. Blank rows are skipped.
    Raises ValueError listing every problem, so the user can fix them in one pass.
    """
    rules, problems = [], []
    for n, row in enumerate(rows, 1):
        category, rule, value = (_clean(row.get(k)) for k in ("category", "rule", "value"))
        kind = _clean(row.get("check")) or "none"
        words, text = _clean(row.get("words")), _clean(row.get("text"))
        position = _clean(row.get("position")) or "anywhere"
        if not (category or rule or value or words or text) and kind == "none":
            continue

        check = None
        if not rule:
            problems.append(f"Row {n}: the rule text is empty")
        if kind == "forbidden_words":
            word_list = [w.strip() for w in words.split(",") if w.strip()]
            if word_list:
                check = RuleCheck(type=kind, words=word_list)
            else:
                problems.append(f"Row {n}: forbidden_words needs at least one word")
        elif kind == "required_text":
            if not text:
                problems.append(f"Row {n}: required_text needs the required text")
            elif position not in POSITIONS:
                problems.append(f"Row {n}: position must be one of {', '.join(POSITIONS)}")
            else:
                check = RuleCheck(type=kind, text=text, position=position)
        elif kind == "no_all_caps":
            check = RuleCheck(type=kind)
        elif kind != "none":
            problems.append(f"Row {n}: unknown check type {kind!r}")

        rules.append(BrandRule(category=category or "other", rule=rule, value=value or None, check=check))
    if problems:
        raise ValueError("; ".join(problems))
    return rules
