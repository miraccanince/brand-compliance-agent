import logging
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from logging_config import setup_logging
setup_logging()

from config import QDRANT_URL
from feedback_store import FeedbackEntry, FeedbackStore
from graph import BrandState, build_graph
from ml_trainer import MIN_SAMPLES, get_history, model_exists, samples_since_last_training, train
from models import BrandRule, ComplianceReport, CorrectionSuggestion
from report_generator import generate_json, generate_pdf
from vector_store import RuleVectorStore

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Brand Compliance Checker", layout="centered")
st.title("Brand Compliance Checker")
st.caption("Upload your brand guidelines PDF and check if your ad copy is compliant.")


@st.cache_resource
def get_graph():
    store = RuleVectorStore(url=QDRANT_URL)
    return build_graph(store), store


@st.cache_resource
def get_feedback_store():
    return FeedbackStore()


graph, vector_store = get_graph()
feedback_store = get_feedback_store()

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
            f"Accuracy: **{last['accuracy']:.0%}**  \n"
            f"Trained on: **{last['samples']}** samples"
        )
        if new_count > 0:
            st.info(f"**{new_count}** new samples since last training. Retrain to improve.")
    else:
        st.info(f"No custom model yet. Need {MIN_SAMPLES} feedback samples.")

    # PDF scope selector
    pdf_list = feedback_store.list_pdfs()
    scope_options = ["All PDFs"] + [f"{p} ({n})" for p, n in pdf_list]
    selected_scope = st.selectbox("Train on:", scope_options)
    selected_pdf = None if selected_scope == "All PDFs" else pdf_list[scope_options.index(selected_scope) - 1][0]

    scoped_count = feedback_store.count() if selected_pdf is None else len(feedback_store.get_all(source_pdf=selected_pdf))
    can_train = scoped_count >= MIN_SAMPLES

    if st.button("Retrain Model", type="primary", disabled=not can_train):
        with st.spinner(f"Training on {selected_scope}..."):
            try:
                result = train(feedback_store, source_pdf=selected_pdf)
                st.success(
                    f"Done! Accuracy: **{result['accuracy']:.0%}**  \n"
                    f"Scope: **{result['scope']}** ({result['samples']} samples)"
                )
                logger.info(f"Model retrained | {result}")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training failed | {e}", exc_info=True)

    if not can_train:
        remaining = MIN_SAMPLES - scoped_count
        st.caption(f"{remaining} more feedback needed for selected scope.")

    # Training history
    if history:
        st.divider()
        st.markdown("**Training History**")
        for i, h in enumerate(reversed(history), 1):
            date = h["trained_at"][:10]
            st.caption(f"#{len(history)-i+1}  {date}  —  {h['accuracy']:.0%} acc  ({h['samples']} samples)")

st.divider()

# ── Main UI ────────────────────────────────────────────────────────────────
pdf_file = st.file_uploader("Brand Guidelines PDF", type=["pdf"])
ad_text  = st.text_area("Advertisement Text", height=150,
                         placeholder="Enter the ad copy you want to check...")

run = st.button("Check Compliance", type="primary",
                disabled=not (pdf_file and ad_text.strip()))

if run:
    # Reset per-violation feedback state on new run
    for key in list(st.session_state.keys()):
        if key.startswith("fb_"):
            del st.session_state[key]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    session_id = str(uuid.uuid4())
    st.session_state["last_pdf_name"] = pdf_file.name
    logger.info(f"New analysis | session={session_id} | pdf={pdf_file.name} | ad_len={len(ad_text)}")

    try:
        initial_state: BrandState = {
            "pdf_path":          tmp_path,
            "ad_text":           ad_text,
            "session_id":        session_id,
            "brand_rules":       [],
            "content_analysis":  {},
            "compliance_report": {},
            "ml_scores":         {},
            "corrections":       [],
            "error":             "",
        }

        with st.status("Running agents...", expanded=True) as status:
            st.write("**Agent 1** — Extracting brand rules from PDF...")
            st.write("**Agent 2** — Analyzing advertisement content...")
            st.write("**Agent 3** — ML pre-filter + compliance check...")
            st.write("**Agent 4** — Generating correction suggestions...")

        result = graph.invoke(initial_state)

        if result.get("error"):
            logger.error(f"Graph error | session={session_id} | {result['error']}")
            status.update(label=f"Failed: {result['error']}", state="error")
            st.stop()

        status.update(label="Analysis complete", state="complete")

        report      = ComplianceReport(**result["compliance_report"])
        corrections = [CorrectionSuggestion(**c) for c in result.get("corrections", [])]
        rules       = [BrandRule(**r) for r in result.get("brand_rules", [])]
        ml_scores   = result.get("ml_scores", {})

        st.session_state["last_report"]      = report
        st.session_state["last_corrections"] = corrections
        st.session_state["last_rules"]       = rules
        st.session_state["last_ml_scores"]   = ml_scores
        st.session_state["last_ad_text"]     = ad_text
        st.session_state["last_session_id"]  = session_id

    except Exception as e:
        logger.error(f"Error | session={session_id} | {e}", exc_info=True)
        st.error(f"Error: {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)

# ── Results (shown after run or from session state) ────────────────────────
if "last_report" in st.session_state:
    report      = st.session_state["last_report"]
    corrections = st.session_state["last_corrections"]
    rules       = st.session_state["last_rules"]
    ml_scores   = st.session_state["last_ml_scores"]
    ad_text_res = st.session_state["last_ad_text"]
    session_id  = st.session_state["last_session_id"]

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Compliance Score", f"{report.score:.0f} / 100")
    col2.metric("Violations",        len(report.violations))
    col3.metric("Corrections",       len(corrections))

    if report.compliant:
        st.success("This advertisement is compliant with the brand guidelines.")
    else:
        st.error("This advertisement has brand guideline violations.")

    st.divider()

    # ML pre-filter scores
    if ml_scores:
        with st.expander("ML Pre-filter Scores (Zero-shot NLI)"):
            st.caption(
                "P(violation) per rule from BART-MNLI. "
                "Rules above 0.50 were sent to LLM for analysis."
            )
            for rule_text, score in sorted(ml_scores.items(), key=lambda x: x[1], reverse=True):
                icon = "🔴" if score >= 0.5 else "🟢"
                st.markdown(f"`{score:.2f}` {icon}  {rule_text[:90]}")

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
