import logging
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DB_PATH = "feedback.db"


@dataclass
class FeedbackEntry:
    ad_text: str
    rule_text: str
    rule_category: str
    is_correct: bool   # True = violation was real, False = false positive
    session_id: str
    source_pdf: str = "unknown"


class FeedbackStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ad_text         TEXT    NOT NULL,
                    rule_text       TEXT    NOT NULL,
                    rule_category   TEXT    NOT NULL,
                    is_correct      INTEGER NOT NULL,
                    session_id      TEXT,
                    source_pdf      TEXT    DEFAULT 'unknown',
                    created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Add source_pdf column if upgrading from old schema
            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN source_pdf TEXT DEFAULT 'unknown'")
            except sqlite3.OperationalError:
                pass  # column already exists
        logger.info(f"Feedback store initialized | db={self.db_path}")

    def add(self, entry: FeedbackEntry) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feedback (ad_text, rule_text, rule_category, is_correct, session_id, source_pdf) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry.ad_text, entry.rule_text, entry.rule_category,
                 int(entry.is_correct), entry.session_id, entry.source_pdf),
            )
        label = "CORRECT" if entry.is_correct else "WRONG"
        logger.info(f"Feedback saved | {label} | rule='{entry.rule_text[:50]}'")

    def get_all(self) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT ad_text, rule_text, rule_category, is_correct FROM feedback"
            ).fetchall()

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def count_since(self, iso_timestamp: str) -> int:
        """Count rows created after an ISO 8601 timestamp (UTC if it has no offset)."""
        # created_at is stored by SQLite as UTC text ("YYYY-MM-DD HH:MM:SS").
        # datetime() normalizes both sides, so the "T" separator and a UTC
        # offset in the argument compare correctly.
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE datetime(created_at) > datetime(?)",
                (iso_timestamp,),
            ).fetchone()[0]

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            correct  = conn.execute("SELECT COUNT(*) FROM feedback WHERE is_correct=1").fetchone()[0]
        return {
            "total":     total,
            "correct":   correct,
            "incorrect": total - correct,
        }
