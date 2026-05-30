import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",  "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL     = os.getenv("QDRANT_URL",    "http://localhost:6333")
LOG_DIR        = os.getenv("LOG_DIR",       "logs")
MAX_PDF_MB     = int(os.getenv("MAX_PDF_MB", "20"))
PDF_TEXT_LIMIT = int(os.getenv("PDF_TEXT_LIMIT", "8000"))

# ML / Zero-shot filter
USE_ML_FILTER  = os.getenv("USE_ML_FILTER", "true").lower() == "true"
ML_THRESHOLD   = float(os.getenv("ML_THRESHOLD", "0.50"))
