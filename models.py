from typing import Literal, Optional
from pydantic import BaseModel, field_validator


class BrandRule(BaseModel):
    category: str
    rule: str
    value: Optional[str] = None

    @field_validator("value", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if v is not None else None


class ComplianceViolation(BaseModel):
    rule: BrandRule
    violation: str
    severity: Literal["critical", "warning"] = "warning"


class ComplianceReport(BaseModel):
    compliant: bool
    violations: list[ComplianceViolation]
    score: float

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        return max(0.0, min(100.0, float(v)))


class CorrectionSuggestion(BaseModel):
    original: str
    corrected: str
    reason: str


class AnalysisResponse(BaseModel):
    compliant: bool
    score: float
    violations: list[ComplianceViolation]
    corrections: list[CorrectionSuggestion]
