# Brand Compliance Checker

Checks advertisement text against the rules in a brand guidelines PDF. A local LLM reads the PDF and proposes rules, the user reviews them, and plain Python checks decide pass or fail. The LLM then suggests rewrites for any violations.

## Problem

Brand guidelines arrive as PDFs, and checking copy against them by hand is slow and inconsistent. Asking a small local LLM to judge compliance directly was not reliable in testing (see Design notes). This project uses the LLM only to read the document and to suggest rewrites. The pass or fail decision comes from deterministic checks, so results are repeatable and each failure names the rule and the matched text.

## How it works

A LangGraph workflow with three nodes. They are fixed functions sharing one state dictionary, not autonomous agents.

1. `extract_rules`: PyMuPDF reads the PDF text. The first 8,000 characters go to a local Ollama model (`llama3.2:3b`, temperature 0, JSON output). It returns rules with a category, rule text, optional value and an optional exact check.
2. `check_compliance`: runs the exact checks in Python. No LLM is involved.
3. `suggest_corrections`: the LLM proposes replacement text for each violation. It is skipped when there are no violations.

An error in node 1 or 2 stops the run. If node 3 fails, the violations are kept and `corrections_error` is set.

In the dashboard, the user reviews the extracted rules in an editable table between nodes 1 and 2. They can fix a check, add one to any rule, or add a rule. The graph skips extraction when rules are supplied. The API has no review step, so it uses the extracted checks as they are.

### Exact checks

The extractor may attach one of three check types to a rule. Its words or text must appear in the PDF, otherwise the check is dropped and the rule is not evaluated.

| Type | Fails when |
|---|---|
| `forbidden_words` | any listed word or phrase appears as a whole word, ignoring case |
| `required_text` | the exact text (spacing normalized, case sensitive) is missing, anywhere or on the last line |
| `no_all_caps` | a line of two or more words is entirely in capitals |

Only rules with an exact check are evaluated. The score is the percentage of exact checks that passed, not model confidence. `compliant` in the API only means that no evaluated check failed. If no rule has an exact check, the run reports an error.

## Example

Input: `samples/test_brand_guidelines.pdf` and the four-line ad in `samples/test.txt`. Output from a real API call, abridged:

```json
{
  "compliant": false, "score": 50.0, "checks_evaluated": 2,
  "violations": [{
    "rule": {"category": "language", "rule": "Forbidden words: cheap, free, guaranteed, limited time, urgent, buy now"},
    "violation": "Contains forbidden word(s): cheap, free, guaranteed, limited time, urgent, buy now",
    "severity": "warning"}],
  "corrections": [
    {"original": "Get it FREE now!", "corrected": "Get it at a special introductory offer now!", "reason": "contains forbidden word: free"},
    {"original": "Limited time offer.", "corrected": "Exclusive offer for a limited time.", "reason": "contains forbidden word: limited time"}
  ],
  "corrections_error": null
}
```

Only two extracted rules had an exact check, so two checks ran: the forbidden-words check failed and the other passed. The call took 26 seconds. Corrections are not re-checked: the second one above still contains a forbidden phrase, and the full response has four more corrections.

## Evaluation

Twelve advertisements written for this project (compliant, single violation, multiple violations, omissions, near-misses) with 12 expected violations. The labels were written for this project and have not been independently reviewed. The checks are deterministic. Only rule extraction used the LLM, in a single run with `llama3.2:3b`.

| Exact checks | Expected violations they cover | Found | Missed | False positives |
|---|---|---|---|---|
| As extracted (forbidden words, all caps) | 5 | 5 | 0 | 0 |
| After manual tagline checks (6 checks) | 10 | 10 | 0 | 1 |

The false positive is an ad with an abbreviated tagline at the bottom, which fails both the wording and the last-line check. The three compliant or near-miss ads had no failed check. Without the manual checks, 7 of the 12 expected violations have no exact check; with them, 2 remain (tone and a conditional disclaimer), and those ads pass. This shows the checks work as designed on a small set. It says nothing about other guidelines, other phrasing or accuracy in general.

## Limitations

- Only text rules with an exact check are evaluated. Logo, color, typography, tone and conditional rules (for example a disclaimer only for financial promotions) are listed as not evaluated.
- With `llama3.2:3b`, extraction produced exact checks for 2 of the 22 or 23 sample rules it returned in three runs. Tagline rules need manual checks in the dashboard.
- Only text is checked, not layout or images. PDFs need selectable text (no OCR), and only the first 8,000 characters are read (`PDF_TEXT_LIMIT`).
- Forbidden words match whole words, so inflections ("guarantees") do not match. In the table, words are comma separated and cannot contain commas.
- Corrections are LLM output and are not validated.
- No authentication, and the API listens on all interfaces with CORS open to any origin. Run it locally only.
- Extraction took 20 to 26 seconds in five runs on the development Mac. Other hardware will differ.

## Feedback and classifier

Each violation in the dashboard can be rated Correct or Wrong. Ratings go to `feedback.db` (SQLite) with the ad text, rule text and PDF name. After 10 ratings with at least 2 of each label, Retrain Model fits a logistic regression on Ollama `nomic-embed-text` embeddings. It reports holdout accuracy with the test size and a majority-label baseline. The model is saved as `custom_classifier.pkl`, but nothing in the compliance workflow uses it, and the amounts of data involved are too small for the accuracy to mean much. Only load pickle files you created.

## Setup and run

Requires Python 3.11 or newer (tested on 3.13) and [Ollama](https://ollama.com).

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text            # only for training the classifier
python -m pip install -e ".[dev]"
streamlit run dashboard.py              # http://localhost:8501
python -m brand_compliance.main         # API on port 8000
python -m pytest                        # 78 offline tests, no Ollama needed
```

```bash
curl -X POST http://localhost:8000/analyze \
  -F "pdf=@samples/test_brand_guidelines.pdf" -F "ad_text=$(cat samples/test.txt)"
```

Endpoints: `GET /health`, `POST /analyze`, and interactive docs at `/docs`. Run from the repository root. The dashboard creates `feedback.db` and `logs/` in the working directory, and training adds `custom_classifier.pkl` and `training_history.json`.

| Variable | Default | Use |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Model for extraction and corrections |
| `PDF_TEXT_LIMIT` | `8000` | Characters of PDF text sent to the model |
| `LOG_DIR` | `logs` | Log directory |
| `PORT`, `ENV` | `8000`, `production` | API port; `ENV=development` enables reload |

## Layout

```
src/brand_compliance/
  agents/            workflow node functions: rule_extractor, compliance_checker, correction_suggester
  graph.py           LangGraph workflow
  rule_editor.py     rules <-> editable table rows
  models.py, config.py, utils.py, logging_config.py
  api.py, main.py    FastAPI app and entry point
  feedback_store.py, ml_trainer.py   SQLite ratings and classifier
  report_generator.py                PDF and JSON export
dashboard.py         Streamlit UI
evaluation/          frozen ads, labels and a small scoring script
samples/  tests/
```

## Design notes

Two components were evaluated and removed. An LLM verdict for each non-exact rule flagged four of the seven LLM-judged rules in every one of the 12 ads, including compliant ones. Qdrant retrieval and a BART-MNLI filter did not change which relevant rules were checked: on an 18-rule extraction, retrieval returned 15 rules and kept every rule with an expected violation, and BART scored every rule below 0.12 and removed none. Both findings come from single runs on this sample.
