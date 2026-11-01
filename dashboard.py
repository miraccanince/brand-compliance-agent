import logging
import os
import tempfile
import uuid

import streamlit as st

from brand_compliance.logging_config import setup_logging
setup_logging()

from brand_compliance.feedback_store import FeedbackEntry, FeedbackStore
from brand_compliance.agents.rule_extractor import extract_rules
from brand_compliance.graph import BrandState, build_graph, make_llm
from brand_compliance.ml_trainer import MIN_SAMPLES, get_history, model_exists, samples_since_last_training, train
from brand_compliance.models import BrandRule, ComplianceReport, CorrectionSuggestion
from brand_compliance.report_generator import generate_json, generate_pdf
from brand_compliance.rule_editor import CHECK_TYPES, POSITIONS, rows_to_rules, rules_to_rows

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Brand Compliance Checker", layout="centered")
st.title("Brand Compliance Checker")
st.caption("Upload your brand guidelines PDF, review the extracted rules and checks, then check your ad copy.")


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_resource
def get_feedback_store():
    return FeedbackStore()


graph = get_graph()
feedback_store = get_feedback_store()

def holdout_text(entry: dict) -> str:
    """Describe a training result as a small holdout check, not proven performance."""
    if "test_samples" not in entry:
        return f"holdout accuracy {entry['accuracy']:.0%} (holdout size not recorded)"
    return (
        f"holdout accuracy {entry['accuracy']:.0%} on {entry['test_samples']} test samples "
        f"(always guessing the majority label: {entry['baseline_accuracy']:.0%})"
    )


# ── Sidebar: feedback stats + retrain ─────────────────────────────────────
with st.sidebar:
    st.header("Model Training")

    stats     = feedback_store.stats()
    new_count = samples_since_last_training(feedback_store)
    history   = get_history()

    col_a, col_b = st.columns(2)
    col_a.metric("Total Feedback", stats["total"])
    col_b.metric("Correct / Wrong", f"{stats['correct']} / {stats['incorrect']}")

    if model_exists():
        last = history[-1]
        st.success(
            f"Custom model active  \n"
            f"Trained on: **{last['samples']}** samples  \n"
            f"Small holdout check, not proven performance: {holdout_text(last)}"
        )
        if new_count > 0:
            st.info(f"**{new_count}** new samples since last training. Retrain to improve.")
    else:
        st.info(f"No custom model yet. Need {MIN_SAMPLES} feedback samples.")

    can_train = stats["total"] >= MIN_SAMPLES

    if st.button("Retrain Model", type="primary", disabled=not can_train):
        with st.spinner("Training on all feedback..."):
            try:
                result = train(feedback_store)
                logger.info(f"Model retrained | {result}")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training failed | {e}", exc_info=True)

    if not can_train:
        remaining = MIN_SAMPLES - stats["total"]
        st.caption(f"{remaining} more feedback needed.")

    # Training history
    if history:
        st.divider()
        st.markdown("**Training History**")
        for i, h in enumerate(reversed(history), 1):
            date = h["trained_at"][:10]
            st.caption(f"#{len(history)-i+1}  {date}  —  {h['samples']} samples, {holdout_text(h)}")

st.divider()

# ── Main UI ────────────────────────────────────────────────────────────────
pdf_file = st.file_uploader("Brand Guidelines PDF", type=["pdf"])

if st.button("Extract Rules", disabled=not pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    try:
        with st.spinner("Extracting rules from the PDF..."):
            extracted = extract_rules(tmp_path, make_llm())
        st.session_state["rule_rows"] = rules_to_rows(extracted)
        st.session_state["rules_pdf_name"] = pdf_file.name
        st.session_state.pop("rule_editor", None)  # drop edits made to a previous extraction
        logger.info(f"Rules extracted for review | pdf={pdf_file.name} | rules={len(extracted)}")
    except Exception as e:
        logger.error(f"Rule extraction failed | {e}", exc_info=True)
        st.error(f"Rule extraction failed: {e}")
    finally:
        os.unlink(tmp_path)

edited_rows = []
if "rule_rows" in st.session_state:
    st.subheader("Review rules and checks")
    st.caption(
        "Only rules with an exact check are evaluated. Set a check for a rule to have it enforced: "
        "forbidden_words (comma separated list), required_text (exact text, anywhere or on the last line) "
        "or no_all_caps. You can edit, delete or add rows."
    )
    edited = st.data_editor(
        st.session_state["rule_rows"],
        key="rule_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "category": st.column_config.TextColumn("Category"),
            "rule": st.column_config.TextColumn("Rule", width="large"),
            "value": st.column_config.TextColumn("Value"),
            "check": st.column_config.SelectboxColumn("Exact check", options=CHECK_TYPES),
            "words": st.column_config.TextColumn("Forbidden words"),
            "text": st.column_config.TextColumn("Required text"),
            "position": st.column_config.SelectboxColumn("Position", options=POSITIONS),
        },
    )
    edited_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)

ad_text = st.text_area("Advertisement Text", height=150,
                       placeholder="Enter the ad copy you want to check...")

run = st.button("Check Compliance", type="primary",
                disabled=not (edited_rows and ad_text.strip()))

if run:
    # Reset per-violation feedback state on new run
    for key in list(st.session_state.keys()):
        if key.startswith("fb_"):
            del st.session_state[key]

    try:
        reviewed_rules = rows_to_rules(edited_rows)
    except ValueError as e:
        st.error(f"Please fix the rules before checking: {e}")
        st.stop()

    session_id = str(uuid.uuid4())
    st.session_state["last_pdf_name"] = st.session_state.get("rules_pdf_name", "unknown")
    logger.info(f"New analysis | session={session_id} | rules={len(reviewed_rules)} | ad_len={len(ad_text)}")

    try:
        initial_state: BrandState = {
            "pdf_path":          "",
            "ad_text":           ad_text,
            "session_id":        session_id,
            "brand_rules":       [r.model_dump() for r in reviewed_rules],
            "compliance_report": {},
            "corrections":       [],
            "corrections_error": "",
            "error":             "",
        }

        with st.status("Running agents...", expanded=True) as status:
            st.write("**Agent 1** — Using the reviewed rules...")
            st.write("**Agent 2** — Checking compliance...")
            st.write("**Agent 3** — Generating correction suggestions...")

        result = graph.invoke(initial_state)

        if result.get("error"):
            logger.error(f"Graph error | session={session_id} | {result['error']}")
            status.update(label=f"Failed: {result['error']}", state="error")
            st.stop()

        status.update(label="Analysis complete", state="complete")

        report      = ComplianceReport(**result["compliance_report"])
        corrections = [CorrectionSuggestion(**c) for c in result.get("corrections", [])]
        rules       = [BrandRule(**r) for r in result.get("brand_rules", [])]

        st.session_state["last_report"]      = report
        st.session_state["last_corrections"] = corrections
        st.session_state["last_corrections_error"] = result.get("corrections_error", "")
        st.session_state["last_rules"]       = rules
        st.session_state["last_ad_text"]     = ad_text
        st.session_state["last_session_id"]  = session_id

    except Exception as e:
        logger.error(f"Error | session={session_id} | {e}", exc_info=True)
        st.error(f"Error: {e}")
        st.stop()

# ── Results (shown after run or from session state) ────────────────────────
if "last_report" in st.session_state:
    report      = st.session_state["last_report"]
    corrections = st.session_state["last_corrections"]
    corrections_error = st.session_state.get("last_corrections_error", "")
    rules       = st.session_state["last_rules"]
    ad_text_res = st.session_state["last_ad_text"]
    session_id  = st.session_state["last_session_id"]

    st.divider()

    checked = report.checks_evaluated
    col1, col2, col3 = st.columns(3)
    col1.metric("Exact checks passed", f"{checked - len(report.violations)} / {checked}")
    col2.metric("Violations",        len(report.violations))
    col3.metric("Corrections",       len(corrections))

    if report.violations:
        st.error(f"{len(report.violations)} of {checked} exact checks failed.")
    else:
        st.success(f"All {checked} exact checks passed.")
    unchecked = [r for r in rules if not r.check]
    st.caption(
        "This result covers only rules with an exact check. "
        f"{len(unchecked)} of {len(rules)} rules have no exact check and were not evaluated."
    )
    if unchecked:
        with st.expander(f"Rules not evaluated ({len(unchecked)})"):
            for rule in unchecked:
                st.markdown(f"**[{rule.category.upper()}]** {rule.rule}")

    st.divider()

    # Brand rules
    with st.expander(f"Brand Rules Found ({len(rules)})"):
        for rule in rules:
            val = f" — `{rule.value}`" if rule.value else ""
            st.markdown(f"**[{rule.category.upper()}]** {rule.rule}{val}")

    # Violations + feedback buttons
    if report.violations:
        st.subheader(f"Violations ({len(report.violations)})")
        st.caption("Rate each violation to help train the custom model.")

        for i, v in enumerate(report.violations):
            label = "[CRITICAL]" if v.severity == "critical" else "[WARNING]"
            with st.expander(f"{label} {v.violation}"):
                st.write(f"**Category:** {v.rule.category}")
                st.write(f"**Rule:** {v.rule.rule}")
                if v.rule.value:
                    st.write(f"**Required:** `{v.rule.value}`")
                st.write(f"**Severity:** {v.severity}")

                fb_key = f"fb_{session_id}_{i}"
                if fb_key in st.session_state:
                    val = st.session_state[fb_key]
                    st.success(f"Feedback saved: {'Correct' if val else 'Wrong'}")
                else:
                    st.markdown("**Was this violation correctly identified?**")
                    c1, c2, _ = st.columns([1, 1, 4])
                    pdf_name = st.session_state.get("last_pdf_name", "unknown")
                    if c1.button("Correct", key=f"yes_{i}"):
                        feedback_store.add(FeedbackEntry(
                            ad_text=ad_text_res,
                            rule_text=v.rule.rule,
                            rule_category=v.rule.category,
                            is_correct=True,
                            session_id=session_id,
                            source_pdf=pdf_name,
                        ))
                        st.session_state[fb_key] = True
                        st.rerun()
                    if c2.button("Wrong", key=f"no_{i}"):
                        feedback_store.add(FeedbackEntry(
                            ad_text=ad_text_res,
                            rule_text=v.rule.rule,
                            rule_category=v.rule.category,
                            is_correct=False,
                            session_id=session_id,
                            source_pdf=pdf_name,
                        ))
                        st.session_state[fb_key] = False
                        st.rerun()
    else:
        st.success("No violations found.")

    # Corrections
    if corrections_error:
        st.warning(f"{corrections_error}. The violations above are not affected.")
    if corrections:
        st.subheader(f"Suggested Corrections ({len(corrections)})")
        for i, c in enumerate(corrections, 1):
            with st.expander(f"Correction {i}: {c.original[:60]}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original**")
                    st.code(c.original, language=None)
                with col_b:
                    st.markdown("**Corrected**")
                    st.code(c.corrected, language=None)
                st.caption(f"Reason: {c.reason}")

    # Download
    st.divider()
    st.subheader("Download Report")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download PDF",
            data=generate_pdf(ad_text_res, report, corrections, rules),
            file_name="compliance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            label="Download JSON",
            data=generate_json(ad_text_res, report, corrections, rules),
            file_name="compliance_report.json",
            mime="application/json",
            use_container_width=True,
        )
