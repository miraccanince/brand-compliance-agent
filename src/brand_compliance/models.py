from typing import Literal, Optional
from pydantic import BaseModel, field_validator


class RuleCheck(BaseModel):
    """An objective check that can be run on advertisement text without an LLM."""
    type: Literal["forbidden_words", "required_text", "no_all_caps"]
    words: list[str] = []                                  # forbidden_words
    text: Optional[str] = None                             # required_text
    position: Literal["anywhere", "last_line"] = "anywhere"  # required_text


class BrandRule(BaseModel):
    category: str
    rule: str
    value: Optional[str] = None
    check: Optional[RuleCheck] = None

    @field_validator("value", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if v is not None else None


class ComplianceViolation(BaseModel):
    rule: BrandRule
    violation: str
    severity: Literal["critical", "warning"] = "warning"


class ComplianceReport(BaseModel):
    """
    Everything here comes from exact checks. `compliant` only means that none of the
    `checks_evaluated` exact checks failed. Rules without an exact check are not evaluated.
    """
    compliant: bool
    violations: list[ComplianceViolation]
    score: float
    checks_evaluated: int = 0

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        return max(0.0, min(100.0, float(v)))


class CorrectionSuggestion(BaseModel):
    original: str
    corrected: str
    reason: str


class AnalysisResponse(BaseModel):
    compliant: bool  # True only means no violation among checks_evaluated exact checks
    score: float
    checks_evaluated: int = 0
    violations: list[ComplianceViolation]
    corrections: list[CorrectionSuggestion]
    corrections_error: Optional[str] = None  # set when corrections could not be generated
