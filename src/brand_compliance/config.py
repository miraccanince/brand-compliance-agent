import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",  "llama3.2:3b")
LOG_DIR        = os.getenv("LOG_DIR",       "logs")
PDF_TEXT_LIMIT = int(os.getenv("PDF_TEXT_LIMIT", "8000"))
