import io
import json
from datetime import datetime

import fitz

from models import BrandRule, ComplianceReport, CorrectionSuggestion

# Page (A4)
PW, PH = 595, 842
ML = 40
CW = PW - ML * 2

# Color palette
C_DARK     = (0.13, 0.13, 0.18)
C_WHITE    = (1.00, 1.00, 1.00)
C_BG       = (0.96, 0.97, 0.99)
C_BORDER   = (0.82, 0.84, 0.88)
C_TEXT     = (0.12, 0.12, 0.16)
C_MUTED    = (0.48, 0.50, 0.55)
C_RED      = (0.83, 0.17, 0.17)
C_RED_BG   = (1.00, 0.94, 0.94)
C_RED_BD   = (0.92, 0.75, 0.75)
C_ORANGE   = (0.88, 0.50, 0.08)
C_ORG_BG   = (1.00, 0.97, 0.92)
C_ORG_BD   = (0.92, 0.84, 0.68)
C_GREEN    = (0.13, 0.62, 0.33)
C_GRN_BG   = (0.92, 1.00, 0.95)
C_GRN_BD   = (0.70, 0.90, 0.76)


def _rect(page, x, y, w, h, fill, stroke=None, width=0.5):
    page.draw_rect(fitz.Rect(x, y, x + w, y + h),
                   color=stroke, fill=fill,
                   width=width if stroke else 0)


def _t(page, x, y, text, size=10, color=C_TEXT, bold=False):
    page.insert_text((x, y + size),
                     text, fontsize=size, color=color,
                     fontname="hebo" if bold else "helv")


def _box(page, x, y, w, text, size=10, color=C_TEXT):
    est_h = max(size + 6, (len(text) // max(1, int(w / (size * 0.55))) + 1) * (size + 4) + 4)
    page.insert_textbox(fitz.Rect(x, y, x + w, y + est_h + 20),
                        text, fontsize=size, color=color)
    return est_h


class _PDF:
    def __init__(self):
        self.doc = fitz.open()
        self.p = None
        self.y = 0
        self._new_page()

    def _new_page(self):
        self.p = self.doc.new_page(width=PW, height=PH)
        self.y = 40

    def _need(self, h):
        if self.y + h > PH - 50:
            self._new_page()

    # ── layout primitives ──────────────────────────────────────────────────

    def header(self, title, sub):
        _rect(self.p, 0, 0, PW, 58, fill=C_DARK)
        _t(self.p, ML, 10, title, size=17, color=C_WHITE, bold=True)
        _t(self.p, ML, 32, sub,   size=9,  color=(0.65, 0.65, 0.70))
        self.y = 68

    def gap(self, h=10):
        self.y += h

    def rule(self):
        self.p.draw_line((ML, self.y), (ML + CW, self.y),
                         color=C_BORDER, width=0.5)
        self.y += 10

    def section_title(self, text, color=C_DARK):
        self._need(34)
        _rect(self.p, ML, self.y, CW, 26, fill=C_BG, stroke=C_BORDER)
        _t(self.p, ML + 10, self.y + 5, text, size=10, color=color, bold=True)
        self.y += 34

    def metrics(self, items):
        """items = [(label, value, color), ...]"""
        self._need(50)
        _rect(self.p, ML, self.y, CW, 46, fill=C_BG, stroke=C_BORDER)
        col = CW // len(items)
        for i, (label, val, color) in enumerate(items):
            x = ML + i * col + 12
            _t(self.p, x, self.y + 6,  label, size=7,  color=C_MUTED, bold=True)
            _t(self.p, x, self.y + 20, val,   size=15, color=color,   bold=True)
        self.y += 54

    def body_text(self, text, indent=8, size=10, color=C_TEXT):
        self._need(30)
        h = _box(self.p, ML + indent, self.y, CW - indent * 2, text, size, color)
        self.y += h + 6

    def violation_card(self, v):
        is_crit = v.severity == "critical"
        acc   = C_RED    if is_crit else C_ORANGE
        bg    = C_RED_BG if is_crit else C_ORG_BG
        bd    = C_RED_BD if is_crit else C_ORG_BD

        lines_v = max(1, len(v.violation)  // 72 + 1)
        lines_r = max(1, len(v.rule.rule)  // 80 + 1)
        h = 14 + lines_v * 14 + lines_r * 12 + 16
        self._need(h + 10)

        _rect(self.p, ML,     self.y, CW, h, fill=bg,  stroke=bd)
        _rect(self.p, ML,     self.y, 5,  h, fill=acc)

        y0 = self.y + 8

        # badge
        bw = 72
        _rect(self.p, ML + 12, y0 - 1, bw, 14, fill=acc)
        _t(self.p, ML + 15, y0,      v.severity.upper(), size=8, color=C_WHITE, bold=True)
        _t(self.p, ML + bw + 20, y0, v.rule.category.upper(), size=8, color=acc, bold=True)
        y0 += 16

        _box(self.p, ML + 12, y0, CW - 24, v.violation, size=10, color=C_TEXT)
        y0 += lines_v * 14

        rule_str = f"Rule: {v.rule.rule}" + (f"  ({v.rule.value})" if v.rule.value else "")
        _box(self.p, ML + 12, y0, CW - 24, rule_str, size=9, color=C_MUTED)

        self.y += h + 7

    def correction_card(self, idx, c):
        lines_o = max(1, len(c.original)  // 50 + 1)
        lines_c = max(1, len(c.corrected) // 50 + 1)
        lines_r = max(1, len(c.reason)    // 72 + 1)
        content_h = max(lines_o, lines_c) * 14 + 8
        h = 22 + content_h + lines_r * 12 + 12
        self._need(h + 10)

        _rect(self.p, ML, self.y, CW, h, fill=C_GRN_BG, stroke=C_GRN_BD)
        _rect(self.p, ML, self.y, 5,  h, fill=C_GREEN)

        y0 = self.y + 8
        _t(self.p, ML + 12, y0, f"Correction {idx}", size=9, color=C_GREEN, bold=True)
        y0 += 16

        half = (CW - 28) // 2

        # original box
        _rect(self.p, ML + 12,          y0 - 2, half, content_h + 4, fill=(1, 0.96, 0.96), stroke=C_RED_BD)
        _t(self.p,   ML + 16,           y0,     "ORIGINAL",  size=7, color=C_RED,   bold=True)
        _box(self.p, ML + 16,           y0 + 10, half - 8, c.original,  size=9, color=C_RED)

        # corrected box
        x2 = ML + 16 + half + 8
        _rect(self.p, x2 - 4,           y0 - 2, half, content_h + 4, fill=C_GRN_BG, stroke=C_GRN_BD)
        _t(self.p,   x2,                y0,     "CORRECTED", size=7, color=C_GREEN, bold=True)
        _box(self.p, x2,                y0 + 10, half - 8, c.corrected, size=9, color=C_GREEN)

        y0 += content_h + 8
        _box(self.p, ML + 12, y0, CW - 24, f"Reason: {c.reason}", size=9, color=C_MUTED)

        self.y += h + 7

    def save(self) -> bytes:
        buf = io.BytesIO()
        self.doc.save(buf)
        self.doc.close()
        return buf.getvalue()


# ── public API ─────────────────────────────────────────────────────────────

def generate_pdf(
    ad_text: str,
    report: ComplianceReport,
    corrections: list[CorrectionSuggestion],
    rules: list[BrandRule],
) -> bytes:
    pdf = _PDF()

    pdf.header(
        "BRAND COMPLIANCE REPORT",
        datetime.now().strftime("%B %d, %Y   %H:%M"),
    )

    score_color = C_GREEN if report.score >= 80 else C_ORANGE if report.score >= 50 else C_RED
    status      = "COMPLIANT" if report.compliant else "NOT COMPLIANT"
    pdf.metrics([
        ("COMPLIANCE SCORE", f"{report.score:.0f} / 100", score_color),
        ("STATUS",           status,                       score_color),
        ("VIOLATIONS",       str(len(report.violations)),  C_RED   if report.violations else C_GREEN),
        ("CORRECTIONS",      str(len(corrections)),        C_ORANGE if corrections      else C_GREEN),
    ])
    pdf.gap(6)

    pdf.section_title("ADVERTISEMENT TEXT")
    pdf.body_text(ad_text)
    pdf.gap(4)

    viol_color = C_RED if report.violations else C_GREEN
    pdf.section_title(f"VIOLATIONS  ({len(report.violations)})", color=viol_color)
    if not report.violations:
        pdf.body_text("No violations found.", color=C_GREEN)
    else:
        for v in report.violations:
            pdf.violation_card(v)
    pdf.gap(4)

    pdf.section_title(f"CORRECTIONS  ({len(corrections)})", color=C_GREEN)
    if not corrections:
        pdf.body_text("No corrections needed.", color=C_GREEN)
    else:
        for i, c in enumerate(corrections, 1):
            pdf.correction_card(i, c)
    pdf.gap(4)

    pdf.section_title(f"BRAND RULES EXTRACTED  ({len(rules)})", color=C_MUTED)
    for rule in rules:
        val = f"  →  {rule.value}" if rule.value else ""
        pdf.body_text(f"[{rule.category.upper()}]  {rule.rule}{val}", size=9, color=C_MUTED)

    return pdf.save()


def generate_json(
    ad_text: str,
    report: ComplianceReport,
    corrections: list[CorrectionSuggestion],
    rules: list[BrandRule],
) -> str:
    return json.dumps({
        "generated_at": datetime.now().isoformat(),
        "ad_text": ad_text,
        "summary": {
            "compliant": report.compliant,
            "score": report.score,
            "violation_count": len(report.violations),
            "correction_count": len(corrections),
        },
        "violations": [
            {
                "severity":  v.severity,
                "violation": v.violation,
                "rule": {"category": v.rule.category, "rule": v.rule.rule, "value": v.rule.value},
            }
            for v in report.violations
        ],
        "corrections": [
            {"original": c.original, "corrected": c.corrected, "reason": c.reason}
            for c in corrections
        ],
        "brand_rules_extracted": [
            {"category": r.category, "rule": r.rule, "value": r.value}
            for r in rules
        ],
    }, indent=2, ensure_ascii=False)
