import asyncio
import os
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from brand_compliance.graph import BrandState, build_graph
from brand_compliance.models import AnalysisResponse, ComplianceReport, CorrectionSuggestion

app = FastAPI(title="Brand Compliance System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()


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
            "compliance_report": {},
            "corrections":       [],
            "corrections_error": "",
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
            checks_evaluated=report.checks_evaluated,
            violations=report.violations,
            corrections=corrections,
            corrections_error=result.get("corrections_error") or None,
        )
    finally:
        os.unlink(tmp_path)
