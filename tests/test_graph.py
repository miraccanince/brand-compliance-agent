from brand_compliance import graph
from brand_compliance.models import BrandRule, ComplianceReport, ComplianceViolation


def initial_state():
    return {
        "pdf_path": "unused.pdf",
        "ad_text": "Buy now!",
        "session_id": "s1",
        "brand_rules": [],
        "compliance_report": {},
        "corrections": [],
        "corrections_error": "",
        "error": "",
    }


def patch_agents(monkeypatch, calls, fail_at=None):
    """Replace the three agent functions with fakes that record call order."""
    report = ComplianceReport(compliant=False, violations=[], score=50)

    def make(name, result):
        def fake(*args, **kwargs):
            calls.append(name)
            if name == fail_at:
                raise RuntimeError("boom")
            return result
        return fake

    monkeypatch.setattr(graph, "extract_rules", make("extract_rules", [BrandRule(category="logo", rule="r")]))
    monkeypatch.setattr(graph, "check_compliance", make("check_compliance", report))
    monkeypatch.setattr(graph, "suggest_corrections", make("suggest_corrections", []))


def test_nodes_run_in_expected_order(monkeypatch):
    calls = []
    patch_agents(monkeypatch, calls)

    result = graph.build_graph().invoke(initial_state())

    assert calls == ["extract_rules", "check_compliance", "suggest_corrections"]
    assert result["error"] == ""
    assert result["brand_rules"] == [{"category": "logo", "rule": "r", "value": None, "check": None}]
    assert result["compliance_report"]["score"] == 50.0


def test_check_compliance_receives_the_extracted_rules(monkeypatch):
    received = []
    monkeypatch.setattr(graph, "extract_rules", lambda path, llm: [BrandRule(category="language", rule="Be formal")])
    monkeypatch.setattr(
        graph, "check_compliance",
        lambda ad, rules: received.append(rules) or ComplianceReport(compliant=True, violations=[], score=100),
    )
    monkeypatch.setattr(graph, "suggest_corrections", lambda ad, report, llm: [])

    graph.build_graph().invoke(initial_state())

    assert received == [[BrandRule(category="language", rule="Be formal")]]


def test_graph_has_three_nodes():
    nodes = set(graph.build_graph().get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"extract_rules", "check_compliance", "suggest_corrections"}


def test_validation_error_in_compliance_check_uses_the_error_path(monkeypatch):
    calls = []
    patch_agents(monkeypatch, calls)
    monkeypatch.setattr(graph, "check_compliance", lambda *a, **k: (_ for _ in ()).throw(ValueError("Invalid compliance response: no verdict for R2")))

    result = graph.build_graph().invoke(initial_state())

    assert result["error"] == "Compliance check failed: Invalid compliance response: no verdict for R2"
    assert result["compliance_report"] == {}
    assert "suggest_corrections" not in calls


def test_supplied_rules_skip_extraction(monkeypatch):
    calls = []
    patch_agents(monkeypatch, calls)
    state = initial_state()
    state["brand_rules"] = [BrandRule(category="language", rule="r", check=None).model_dump()]

    result = graph.build_graph().invoke(state)

    assert calls == ["check_compliance", "suggest_corrections"]  # extract_rules was not called
    assert result["error"] == ""


def test_correction_failure_keeps_the_confirmed_report_and_reports_the_error(monkeypatch):
    violation = ComplianceViolation(rule=BrandRule(category="language", rule="No free"), violation="Contains free")
    report = ComplianceReport(compliant=False, violations=[violation], score=50, checks_evaluated=2)
    calls = []
    patch_agents(monkeypatch, calls)
    monkeypatch.setattr(graph, "check_compliance", lambda *a, **k: report)
    monkeypatch.setattr(graph, "suggest_corrections", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("model server down")))

    result = graph.build_graph().invoke(initial_state())

    assert result["error"] == ""  # not a failed analysis
    assert result["compliance_report"]["violations"][0]["violation"] == "Contains free"
    assert result["corrections"] == []
    assert result["corrections_error"] == "Correction suggestion failed: model server down"
