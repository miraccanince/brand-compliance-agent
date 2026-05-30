import asyncio
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from config import QDRANT_URL
from graph import BrandState, build_graph
from models import AnalysisResponse, ComplianceReport, CorrectionSuggestion
from vector_store import RuleVectorStore
import os

app = FastAPI(title="Brand Compliance System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_vector_store = RuleVectorStore(url=QDRANT_URL)
_graph = build_graph(_vector_store)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    pdf: UploadFile = File(..., description="Brand guidelines PDF"),
    ad_text: str = Form(..., description="Advertisement text to check"),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await pdf.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        initial_state: BrandState = {
            "pdf_path":          tmp_path,
            "ad_text":           ad_text,
            "session_id":        str(uuid.uuid4()),
            "brand_rules":       [],
            "content_analysis":  {},
            "compliance_report": {},
            "ml_scores":         {},
            "corrections":       [],
            "error":             "",
        }

        result = await asyncio.to_thread(_graph.invoke, initial_state)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        report      = ComplianceReport(**result["compliance_report"])
        corrections = [CorrectionSuggestion(**c) for c in result.get("corrections", [])]

        return AnalysisResponse(
            compliant=report.compliant,
            score=report.score,
            violations=report.violations,
            corrections=corrections,
        )
    finally:
        os.unlink(tmp_path)
