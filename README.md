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
