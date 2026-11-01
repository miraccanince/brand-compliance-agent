"""Small evaluation of the exact compliance checks against the frozen advertisements.

Run from the repository root after `python -m pip install -e ".[dev]"`:

    python evaluation/run_eval.py extract   # freeze extracted rules (one LLM call)
    python evaluation/run_eval.py compare   # check every advertisement (no LLM) and score
    python evaluation/run_eval.py score     # re-score the saved compare results

Reported violations are mapped to gold IDs by exact text match against
rule_mapping.json. Unmatched text is listed for review and is never scored.
Only IDs whose rules have an exact check are scored; the others are not evaluated.
"""
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PDF = HERE.parent / "samples" / "test_brand_guidelines.pdf"
RESULTS = HERE / "results"
EXTRACT_TIMEOUT_SECONDS = 180


def load_json(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def scored_ids() -> set[str]:
    return {r["score_id"] for r in load_json("gold_rules.json")["rules"] if r["score_id"]}


def load_rules():
    """Frozen extracted rules and their reviewed mapping to gold IDs."""
    from brand_compliance.models import BrandRule

    rules = [BrandRule(**r) for r in load_json("extracted_rules.json")["rules"]]
    mapping = {norm(k): v for k, v in load_json("rule_mapping.json")["mapping"].items()}
    missing = [r.rule for r in rules if norm(r.rule) not in mapping]
    if missing:
        sys.exit(f"Rules without a reviewed mapping: {missing}")
    return rules, mapping


def classify(rule_texts: list[str], mapping: dict[str, str]) -> dict:
    """Split reported rule texts into gold IDs and unmapped text."""
    ids, unmapped = [], []
    for text in rule_texts:
        if norm(text) in mapping:
            ids.append(mapping[norm(text)])
        else:
            unmapped.append(text)
    return {"ids": ids, "unmapped": unmapped}


def score_case(case: dict, reported_ids: list[str], scope: set[str]) -> dict:
    """Compare reported IDs with expected IDs inside `scope`, leaving out "don't care" pairs."""
    in_scope = scope - set(case["dont_care"])
    expected = set(case["expected"]) & in_scope
    reported = set(reported_ids) & in_scope
    return {
        "true_positive": sorted(expected & reported),
        "missed": sorted(expected - reported),
        "false_positive": sorted(reported - expected),
        "unsupported_visual": [i for i in reported_ids if i == "VISUAL"],
        "unscored_excluded": [i for i in reported_ids if i == "EXCLUDED"],
        "pairs_in_scope": len(in_scope),
    }


class CallTimeout(Exception):
    pass


def call_with_timeout(fn, seconds: float):
    """Run fn() and raise CallTimeout after `seconds` of wall-clock time (main thread, Unix)."""
    def on_alarm(signum, frame):
        raise CallTimeout(f"no result after {seconds}s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def cmd_extract():
    from brand_compliance.agents.rule_extractor import extract_rules
    from brand_compliance.config import OLLAMA_MODEL
    from brand_compliance.graph import make_llm

    path = HERE / "extracted_rules.json"
    if path.exists():
        sys.exit(f"{path.name} is frozen. Delete it deliberately to extract again.")
    t0 = time.time()
    rules = call_with_timeout(lambda: extract_rules(str(PDF), make_llm()), EXTRACT_TIMEOUT_SECONDS)
    out = {
        "model": OLLAMA_MODEL,
        "temperature": 0,
        "elapsed_seconds": round(time.time() - t0, 1),
        "rules": [r.model_dump() for r in rules],
    }
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Extracted {len(rules)} rules in {out['elapsed_seconds']}s -> {path.name}")
    for r in rules:
        print(f"  [{r.category}] {r.rule}" + (f" -> {r.value}" if r.value else "")
              + (f" | check: {r.check.model_dump(exclude_defaults=True)}" if r.check else ""))


def cmd_compare():
    from brand_compliance.agents.compliance_checker import check_compliance

    rules, _ = load_rules()
    entries = []
    for case in load_json("cases.json")["cases"]:
        entry = {"case": case["id"]}
        t0 = time.time()
        try:
            report = check_compliance(case["text"], rules)
            entry["checks_evaluated"] = report.checks_evaluated
            entry["score"] = report.score
            entry["violations"] = [{"rule": v.rule.rule, "violation": v.violation} for v in report.violations]
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        entry["check_seconds"] = round(time.time() - t0, 4)
        entries.append(entry)
    RESULTS.mkdir(exist_ok=True)
    settings = {"rules": len(rules), "rules_with_exact_check": sum(1 for r in rules if r.check)}
    (RESULTS / "compare_raw.json").write_text(
        json.dumps({"settings": settings, "entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd_score()


def cmd_score():
    rules, mapping = load_rules()
    cases = {c["id"]: c for c in load_json("cases.json")["cases"]}
    exact_ids = {mapping[norm(r.rule)] for r in rules if r.check}
    scope = scored_ids() & exact_ids
    entries = json.loads((RESULTS / "compare_raw.json").read_text(encoding="utf-8"))["entries"]
    print(f"Exact-check IDs scored: {sorted(scope)}\n")

    totals = {"true_positive": 0, "missed": 0, "false_positive": 0, "false_positive_on_compliant_ads": 0,
              "unsupported_visual": 0, "unscored_excluded": 0, "pairs_in_scope": 0}
    unmapped, failed = [], []
    for entry in entries:
        if "error" in entry:
            failed.append(f"{entry['case']}: {entry['error']}")
            continue
        case = cases[entry["case"]]
        reported = classify([v["rule"] for v in entry["violations"]], mapping)
        unmapped += [f"{entry['case']}: {u}" for u in reported["unmapped"]]
        s = score_case(case, reported["ids"], scope)
        for key in ("true_positive", "missed", "false_positive", "unsupported_visual", "unscored_excluded"):
            totals[key] += len(s[key])
        totals["pairs_in_scope"] += s["pairs_in_scope"]
        if not case["expected"]:
            totals["false_positive_on_compliant_ads"] += len(s["false_positive"])
        print(f"{entry['case']} expected={case['expected']} TP={s['true_positive']} "
              f"missed={s['missed']} FP={s['false_positive']}")
    print("\n" + json.dumps(totals, indent=2))
    print("Checks that failed to run:", failed or "none")
    print("Unmapped rule text needing review:", unmapped or "none")


COMMANDS = {"extract": cmd_extract, "compare": cmd_compare, "score": cmd_score}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]]()
