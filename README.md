# Multi-Agent Brand Compliance System

A multi-agent AI system that reads a company's brand guidelines PDF and checks whether an advertisement copy is compliant. Violations are detected, explained, and corrected automatically.

---

## How It Works

```
Brand Guidelines PDF + Ad Text
           ↓
  Agent 1 — Extracts brand rules from PDF
           ↓
  Agent 2 — Analyzes ad text (tone, colors, slogans, keywords)
           ↓
  Agent 3 — ML pre-filter (BART-MNLI zero-shot NLI)
            + LLM compliance check on likely violations
           ↓
  Agent 4 — Suggests corrections for each violation
           ↓
  Dashboard — Score, violations, corrections, downloadable report
           ↓
  User rates violations → Feedback collected
           ↓
  Custom classifier trained on feedback (improves over time)
```

---

## Tech Stack

| Component | Purpose |
|---|---|
| **LangGraph** | Orchestrates the 4-agent pipeline |
| **Ollama (llama3.2)** | LLM for rule extraction, analysis, compliance check, corrections |
| **Qdrant** | Stores brand rules as vectors, semantic search |
| **BART-MNLI** | Zero-shot NLI pre-filter (reduces LLM token usage) |
| **Scikit-learn** | Logistic regression trained on user feedback |
| **Streamlit** | Dashboard UI |
| **FastAPI** | REST API for programmatic access |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
OLLAMA_MODEL=llama3.2
QDRANT_URL=http://localhost:6333
```

No API keys required — everything runs locally.

### 4. Qdrant (optional)

Qdrant runs in-memory by default. For persistence, start it with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Running

### Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

### API

```bash
python main.py
# or
ENV=development python main.py   # with hot reload
```

API docs at `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/analyze \
  -F "pdf=@brand_guidelines.pdf" \
  -F "ad_text=Get it FREE now! Buy before it's too late!"
```

---

## Usage

1. Upload your brand guidelines PDF
2. Enter the ad copy you want to check
3. Click **Check Compliance**
4. Review violations and corrections
5. Rate each violation as **Correct** or **Wrong**
6. Once 10+ feedback samples accumulate, click **Retrain Model** in the sidebar

---

## Self-Learning Loop

The system improves over time through user feedback:

```
Violation detected → User rates it → Saved to feedback.db
                                             ↓
                             Logistic Regression retrained
                             (ad_embedding + rule_embedding → violation: bool)
                                             ↓
                             Custom model supplements BART-MNLI in next runs
```

Feedback is tagged by source PDF. You can train a global model (all PDFs) or a company-specific model from the sidebar.

---

## Project Structure

```
├── agents/
│   ├── rule_extractor.py        # Agent 1: PDF → brand rules
│   ├── content_analyzer.py      # Agent 2: ad text analysis
│   ├── compliance_checker.py    # Agent 3: ML filter + LLM check
│   ├── correction_suggester.py  # Agent 4: fix suggestions
│   └── zero_shot_filter.py      # BART-MNLI pre-filter
├── graph.py                     # LangGraph pipeline
├── vector_store.py              # Qdrant wrapper
├── feedback_store.py            # SQLite feedback storage
├── ml_trainer.py                # Logistic regression training
├── report_generator.py          # PDF + JSON report export
├── models.py                    # Pydantic data models
├── config.py                    # Centralized configuration
├── utils.py                     # JSON parsing helpers
├── logging_config.py            # Rotating file + stream logger
├── dashboard.py                 # Streamlit UI
├── api.py                       # FastAPI endpoints
└── main.py                      # Uvicorn entry point
```

---

## Configuration

All settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | LLM model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `USE_ML_FILTER` | `true` | Enable BART-MNLI pre-filter |
| `ML_THRESHOLD` | `0.50` | Violation confidence threshold |
| `PDF_TEXT_LIMIT` | `8000` | Max characters read from PDF |
| `LOG_DIR` | `logs` | Log file directory |
| `ENV` | `production` | Set to `development` for hot reload |

---

## Architecture Deep-Dive

### How agents share context

All four agents communicate through a shared `BrandState` TypedDict managed by LangGraph. Each node reads only what it needs and writes back only what it produces — LangGraph merges the partial updates into the shared state between steps.

```
BrandState
  pdf_path          → Agent 1 reads
  brand_rules       ← Agent 1 writes  →  Agent 3 reads
  content_analysis  ← Agent 2 writes  →  Agent 3 reads
  compliance_report ← Agent 3 writes  →  Agent 4 reads
  ml_scores         ← Agent 3 writes
  corrections       ← Agent 4 writes
  error             ← any agent writes (short-circuits the rest)
```

Agent 3 also queries Qdrant for semantically relevant rules instead of passing the full list through state. This keeps the LLM prompt focused on the rules most likely to be violated by the specific ad text.

---

### Orchestration problems encountered

**Duplicate execution.** The first version had the dashboard calling agents directly and also calling `graph.invoke()`, so every analysis ran twice. Fixed by routing everything through `graph.invoke()` and caching results in `st.session_state`.

**Qdrant API breaking change.** Qdrant v1.18 removed the `.search()` method. Replaced with `.query_points()` and changed result access from `results` to `results.points`. No migration guide existed — found it through the error traceback.

**Vector dimension mismatch.** Switching from OpenAI embeddings (1536 dims) to Ollama nomic-embed-text (768 dims) caused Qdrant to reject new vectors against the existing collection. Solution: `_ensure_collection()` checks the stored dimension on every startup and deletes and recreates the collection if there is a mismatch.

**LLM JSON inconsistency.** llama3.2 without `format="json"` would occasionally return markdown-fenced JSON, arrays wrapped in objects, or malformed output. Fixed with a `parse_json_response()` utility that strips fences, tries `json.loads`, and falls back to `json-repair` as a last resort. An `extract_list()` helper normalises both `{"rules": [...]}` and `[...]` response shapes.

---

### Memory management

**Short-term — session memory.**
Each `graph.invoke()` call receives a UUID `session_id`. Brand rules extracted from the PDF are stored in Qdrant tagged with that session ID. Rule lookups during compliance checking are scoped to the same session ID, so two concurrent analyses never contaminate each other. When Qdrant runs in-memory mode (no Docker), this data is discarded on process exit.

**Long-term — feedback memory.**
User ratings ("Correct" / "Wrong") are written to a SQLite database with the source PDF filename as a tenant tag. This persists across restarts. The logistic regression trainer reads from this store, optionally filtered to a single PDF, and writes the trained model to `custom_classifier.pkl`. The custom model is loaded on the next run and runs in parallel with BART-MNLI.

What is intentionally absent: there is no cross-session conversational memory. Each analysis starts from scratch. If session history were needed, a Redis-backed message store per session would be the straightforward addition.

---

### Zero-shot NLI pre-filter

Agent 3 uses `facebook/bart-large-mnli` to score each brand rule against the ad text before sending anything to the LLM. The hypothesis template is:

```
"This advertisement violates the brand guideline: {rule}"
```

BART-MNLI was fine-tuned on the Multi-Genre NLI dataset (400K+ premise-hypothesis pairs labelled as entailment / contradiction / neutral). Using `multi_label=True` causes each rule to be scored independently rather than in competition. Rules that score below the configured threshold (default 0.50) are dropped before the LLM call.

In practice this reduces the number of rules sent to the LLM by 80-90% on a typical brand guidelines document, with no cloud API cost since the model runs locally on CPU or Apple MPS.

---

### Authorization design

The current implementation is single-tenant and has no authentication layer. In a production deployment serving multiple brands the required additions would be:

- JWT middleware on all FastAPI endpoints, extracting `user_id` and `tenant_id` from the token
- Qdrant collection names scoped per tenant: `brand_rules_{tenant_id}`
- Feedback records tagged with `tenant_id` in addition to `source_pdf`
- The retrain endpoint restricted to users with an admin role for the given tenant

The existing `source_pdf` tagging in the feedback store and the session-scoped Qdrant queries are already structured in a way that maps cleanly onto tenant isolation — the primary change would be replacing the PDF filename with an authenticated tenant identifier.
